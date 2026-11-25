"""Percentiles of the ensemble: the GWI's summary CSVs, and the quantiles
(its distribution, finely) that distributions.py writes to netCDF.

The terms, as used throughout:
    ensemble            every drawn member's values, before percentiles
                        (the member axis, N)
    warming definition  a filter along the years, applied to each window
                        of years: the annual value (windows of one year),
                        the mean over a window (AR6's decade, the CGWL's
                        centred mean), the end point of a line fitted
                        through it (SR1.5) or its slope (the AR6 rate)
    summary             the median and the likely and very likely ranges
                        of the ensemble (gwi.py's SIGMAS: the 50th, 17th
                        to 83rd and 5th to 95th percentiles), taken along
                        the member axis: what the CSVs give
    quantiles           the ensemble's percentiles at many levels (e.g.
                        every 0.1 from the 0th to the 100th: gwi.py's
                        --quantile-step), its distribution finely, taken
                        in the same call as the summary's
    level               which percentile, in percent: 5 for the 5th
                        percentile, 99.9 for the 99.9th (np.percentile's q).
                        The summary's levels are SIGMAS (5, 17, 83, 95, 50),
                        the quantiles' e.g. 0, 0.1, ..., 100; every
                        percentile array here has the levels along its first
                        axis
    headlines           the headline years (e.g. the latest, and the years
                        of IPCC assessments), whose values of every
                        definition the GWI_headlines-summary CSV
                        gathers

Every warming definition so far is linear in time, so each member's value
in each window is rebuilt from T_mod and T_obs with the definition applied
first, one block of windows at a time (members.Sample.ensemble_definition).
So the ensemble of every year is never held at once, and the summaries
don't depend on the block size.

How the functions connect:
    percentiles_by_year_and_variable  ensemble (t, variables, N) ->
                                      percentiles (sigmas, t, variables)
    percentiles_as_csv_table          percentiles -> a DataFrame laid out
                                      as the CSVs are (a row per year or
                                      window, a column per (variable,
                                      percentile))
    windowed_percentiles              the first, for any definition linear
                                      in time, over a list of windows, in
                                      blocks
    definition_years                  the years a definition has a value
                                      for, and their windows
    definition_timeseries             ... over every output year whose
                                      window lies within the output years:
                                      each definition's timeseries (gwi.py's
                                      WARMING_DEFINITIONS gives each one's
                                      function and window), at the summary's
                                      levels and the quantiles' together
    select_levels                     those percentiles at some of the
                                      levels: the summary's, for the
                                      GWI_timeseries-summary-*
                                      CSVs (laid out by
                                      percentiles_as_csv_table), or the
                                      quantiles'
    quantiles_dataarray               the quantiles, labelled, for the
                                      quantiles netCDF
    headline_rows                     a timeseries' rows at the headline
                                      years, labelled for a headlines CSV
    observed_timeseries_table         a definition's timeseries for the
                                      observed temperatures: percentiles
                                      across the observed members, no
                                      rebuild
                                      (Obs_timeseries-summary-*)
    correlations                      the correlations between variables
                                      across members, for any number of
                                      members
    fidelity_rows                     how close a release is to the run
                                      (its parent): the
                                      GWI_headlines-fidelity CSV's rows for
                                      one headline year
    prior_timeseries_table,           the prior warming (stage A's T_mod)
    erf_timeseries_table              and the forcing, over every
                                      (parameter set, ERF member) pair and
                                      every ERF member

The arrays' axes are named as in src/method/members.py: t year (or window
of years), and N the members; `variables` is the axis along
sample.output_vars, and `sigmas` the axis along the percentiles asked for.
"""

import numpy as np
import pandas as pd
import xarray as xr

import src.infrastructure.compute_resources as compute_resources
import src.method.attribution as attribution


def years_per_block_limits(sample):
    """The most years (or windows) of the ensemble to rebuild at once that
    the threads, and the memory, allow: (by_threads, by_memory).

    sample: members.Sample
    by_threads: the most years whose (year, variable) percentile
        calculations don't outnumber the threads, ⌊threads / variables⌋,
        and at least one (see years_per_block)
    by_memory: the most years whose block, and the copy the percentiles
        make of it (2 × years × sample.bytes_per_year), fit in the memory
        that was free when the sample was made, less the rebuild's own
        working memory (sample.rebuild_working_bytes), and at least one;
        None if the free memory can't be told (e.g. on macOS)

    This only sizes the blocks; it never stops the run. The measured free
    memory can understate what is free (under Slurm the job's use counts
    file cache the system would reclaim), so if even one year looks short,
    the blocks are one year, and compute_resources.check_memory, which
    checks the expected peak against the job's limit, decides whether the
    run can go ahead.
    """
    # The threads' limit: one (year, variable) calculation per thread
    by_threads = max(1, sample.n_threads // len(sample.output_vars))
    # The memory's limit, if the free memory is known
    if sample.memory_available is None:
        return by_threads, None
    # The memory for the blocks: what was free, less the rebuild's own
    for_blocks = sample.memory_available - sample.rebuild_working_bytes
    # ... in years, each year of a block needing bytes_per_year twice, and
    # at least one
    by_memory = max(1, int(for_blocks // (2 * sample.bytes_per_year)))
    return by_threads, by_memory


def years_per_block(sample):
    """How many years (or windows) of the ensemble to rebuild at once: as
    many as the threads can take percentiles of in one pass, unless the
    memory allows fewer (see years_per_block_limits).

    sample: members.Sample

    Each (year, variable) pair is its own percentile calculation along the
    member axis, so a block of t years holds t × variables calculations,
    which are shared out over the threads. The threads' limit is the most
    years whose calculations don't outnumber the threads, ⌊threads /
    variables⌋, and at least one: e.g. 4 years for 6 variables on 28
    threads, which is 24 calculations for 28 threads.

    Bigger blocks gain nothing. The rebuild is no faster in bigger blocks,
    since each member's scaling factors are summed once for the whole
    sample. Nor are the percentiles, once every thread has a calculation; a
    few calculations more than threads is slower, as the threads that finish
    first then wait while the rest take a second one. Measured at 6 million
    members (the cube at --samples=60, before Phase 2; 6 variables, 28
    threads): 4, 14 and 32 years per block take 35.6, 35.0 and 34.5 s for
    the timeseries, and 1.1,
    3.8 and 8.7 GiB per block (a block's members, and the copy the
    percentiles make of them); 5 years (30 calculations) took 38.7 s.

    For very large ensembles the memory sets the limit instead: e.g. the
    complete ensemble (790 million members, 6 variables) needs ~19 GB per
    year of the ensemble, so four-year blocks need ~150 GB.
    """
    # The threads' and the memory's limits
    by_threads, by_memory = years_per_block_limits(sample)
    # The smaller of the two (the threads' alone if the memory is unknown)
    return by_threads if by_memory is None else min(by_threads, by_memory)


def block_bytes(sample):
    """The most memory an output here holds at once: a block of
    years_per_block years (or windows) of the ensemble, and the copy the
    percentiles make of it, 2 × years_per_block × sample.bytes_per_year.

    sample: members.Sample
    Every warming definition, the annual timeseries and the rates included,
    works in such blocks, so this holds whichever outputs the run gives.
    """
    return 2 * years_per_block(sample) * sample.bytes_per_year


def percentiles_by_year_and_variable(ensemble, sigmas):
    """The percentiles of the attributed warming along the member axis, for
    each year and variable.

    ensemble: array (t, variables, N), every member's value in each year
        (or window), e.g. a block of years of attributed warming
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    Returns an array (sigmas, t, variables): element [s, i, v] is the
    sigmas[s]-th percentile of ensemble[i, v, :], the N members of
    year i and variable v.

    It is np.percentile(ensemble, sigmas, axis=-1) (in xarray,
    a quantile along the member dimension), with the (year, variable) pairs
    shared out over the job's threads, so a block of a few years still uses
    every CPU.
    """
    # The sizes: years, variables, members
    n_years, n_vars, n_members = ensemble.shape
    # The year and variable axes merged into one axis of (year, variable)
    # pairs, pair i · variables + v being year i and variable v, so the
    # threads can share out the pairs (percentile_threaded shares out the
    # first axis): a view, (t · variables, N)
    pairs = ensemble.reshape(n_years * n_vars, n_members)
    # The percentiles of each pair along the member axis, over threads:
    # (sigmas, t · variables)
    pair_percentiles = compute_resources.percentile_threaded(
        pairs, sigmas, axis=1)
    # The pair axis split back into years and variables: (sigmas, t,
    # variables)
    return pair_percentiles.reshape(len(sigmas), n_years, n_vars)


def percentiles_as_csv_table(percentiles, sigmas, output_vars, labels):
    """Percentiles laid out as in the GWI's CSVs: a row for each year (or
    window), a column for each (variable, percentile).

    percentiles: array (sigmas, t, variables)
    sigmas: the percentiles taken, e.g. gwi.py's SIGMAS
    output_vars: the names along the variable axis
    labels: the rows' names in the CSV's Year column, one per position
        along t, e.g. the years, or '2016-2025'
    Returns a DataFrame.
    """
    # One CSV column per (variable, percentile), each along t: (t,)
    columns = {
        (var, sigma): percentiles[sigmas.index(sigma), :, v]
        for v, var in enumerate(output_vars) for sigma in sigmas
    }
    # The table, one CSV row per position along t
    table = pd.DataFrame(columns, index=labels)
    table.columns.names = ['variable', 'percentile']
    table.index.name = 'Year'
    return table


def windowed_percentiles(sample, levels, definition, windows):
    """The percentiles of a definition linear in time, e.g. the mean over a
    decade, in each window of years.

    sample:
        members.Sample
    levels:
        the percentiles to take, e.g. gwi.py's SIGMAS, or those and the
        quantiles' levels together
    definition:
        a warming definition, a function of an array along its first axis,
        the years, and linear (see members.Sample.ensemble_definition), e.g.
        warming_definitions.mean_over_years
    windows:
        the windows, (first_year, last_year) pairs, both included
    Returns an array (levels, windows, variables).

    The windows are taken years_per_block at a time: each block rebuilds
    the ensemble's value in its windows, (windows, variables, N), and takes
    its percentiles in one pass over the threads.
    """
    # How many windows to take at once
    windows_per_block = years_per_block(sample)
    # The percentiles of each block of windows: one (levels, windows,
    # variables) array per block
    block_percentiles = []
    for start in range(0, len(windows), windows_per_block):
        # The ensemble's value in these windows: (windows, variables, N)
        ensemble = sample.ensemble_definition(
            definition, windows[start:start + windows_per_block])
        # Its percentiles: (levels, windows, variables)
        block_percentiles.append(
            percentiles_by_year_and_variable(ensemble, levels))
    # Every window's percentiles: (levels, windows, variables)
    return np.concatenate(block_percentiles, axis=1)


def definition_timeseries(sample, levels, definition, first_output_year,
                          last_output_year, years_before, years_after):
    """A definition's timeseries: its percentiles in every output year
    whose window lies within the output years.

    sample:
        members.Sample
    levels:
        the percentiles to take: gwi.py's SIGMAS, for the summary, or those
        and the quantiles' levels together, in one call (each percentile
        is the same, to the last bit, whatever else is taken with it)
    definition:
        a warming definition linear in time (see windowed_percentiles)
    first_output_year, last_output_year:
        the output years (both included)
    years_before, years_after:
        each year's window, from years_before years before it to years_after
        years after it
    Returns (years, percentiles):
        years: the years, (t,)
        percentiles: array (levels, t, variables)
    """
    # The years whose windows lie within the output years, and each one's
    # window
    years, windows = definition_years(first_output_year, last_output_year,
                                      years_before, years_after)
    # The ensemble's value in each window, and its percentiles
    return years, windowed_percentiles(sample, levels, definition, windows)


def definition_years(first_output_year, last_output_year, years_before,
                     years_after):
    """The years a definition has a value for: those whose window lies
    within the output years.

    first_output_year, last_output_year:
        the output years (both included)
    years_before, years_after:
        each year's window, from years_before years before it to years_after
        years after it
    Returns (years, windows):
        years: array (t,)
        windows: each year's window, a (first_year, last_year) pair
    """
    # The years whose windows lie within the output years
    years = np.arange(first_output_year + years_before,
                      last_output_year - years_after + 1)
    # Stop if there are none
    if len(years) == 0:
        raise ValueError(
            f'No year from {first_output_year} to {last_output_year} has '
            f'its whole window ({years_before} years before it to '
            f'{years_after} after) within those years.')
    # Each year's window
    windows = [(year - years_before, year + years_after) for year in years]
    return years, windows


def select_levels(percentiles, levels, wanted):
    """The percentiles at some of the levels they were taken at, e.g. the
    summary's five from among a thousand.

    percentiles:
        array (levels, ...), e.g. from definition_timeseries: the
        percentiles at each level, along the first axis
    levels:
        the levels they were taken at, in percent, in the order of that
        axis
    wanted:
        the levels to keep, in percent, each one of levels, in the order
        wanted, e.g. gwi.py's SIGMAS for the summary
    Returns an array (wanted, ...).

    For example, percentiles taken at levels [0, 5, 17, 50, 83, 95, 100],
    with wanted [5, 95, 50], gives the 5th, 95th and 50th percentiles:
    percentiles[[1, 5, 3]].
    """
    # Each level's position along the first axis
    position = {level: i for i, level in enumerate(levels)}
    # The wanted levels' positions, in their order
    return percentiles[[position[level] for level in wanted]]


def quantiles_dataarray(percentiles, levels, years, output_vars, name,
                        attributes):
    """A definition's quantiles, labelled, as the quantiles netCDF holds
    them (see distributions.py).

    percentiles:
        array (levels, t, variables), e.g. select_levels(...) at the
        quantiles' levels
    levels:
        the levels along its first axis, in percent
    years:
        the years along t
    output_vars:
        the names along the variable axis
    name:
        the DataArray's name, e.g. 'ANNUAL'
    attributes:
        its attributes: at least long_name and units (see
        distributions.WARMING_ATTRIBUTES)
    Returns a DataArray (year, percentile, output_variable): year first,
    then the distribution, then the variables, as the inputs are laid out.
    """
    # Year first: (t, levels, variables)
    values = percentiles.transpose(1, 0, 2)
    return xr.DataArray(
        values, dims=('year', 'percentile', 'output_variable'),
        coords={'year': years,
                'percentile': ('percentile', np.asarray(levels, dtype=float),
                               {'long_name': 'percentile across the '
                                             'members', 'units': 'percent'}),
                'output_variable': list(output_vars)},
        name=name, attrs=attributes)


def headline_rows(timeseries, headline_years, labels):
    """A definition's timeseries at the headline years, labelled as the
    GWI_headlines-summary CSV labels them.

    timeseries:
        a DataFrame indexed by year, e.g. a timeseries CSV's table
    headline_years:
        the years to take, each in timeseries' index
    labels:
        the rows' names, one per year, e.g. '2016-2025' or '2025 (SR15
        definition)'
    Returns:
        a DataFrame with those rows, indexed by the labels ('Year').
    """
    # The headline years' rows
    rows = timeseries.loc[list(headline_years)].copy()
    # ... named by their labels
    rows.index = pd.Index(labels, name='Year')
    return rows


def observed_timeseries_table(gmt, sigmas, definition, years_before,
                              years_after):
    """A warming definition's timeseries for the observed temperatures: its
    percentiles across the observed members, in every year whose window
    lies within the observed years.

    gmt:
        the observed members' temperatures, a DataArray (year, gmt_member)
        over consecutive years
    sigmas:
        the percentiles to take, e.g. gwi.py's SIGMAS
    definition:
        a warming definition, a function of an array along its first axis, the
        years, e.g. warming_definitions.mean_over_years
    years_before, years_after:
        each year's window, from years_before years before it to years_after
        years after it
    Returns:
        a DataFrame indexed by year ('Year'), with a column for each ('Obs',
        percentile), as the Obs_headlines-summary CSV lays them
        out.

    The observations need no rebuild: the definition is applied to each
    observed member directly, and the percentiles taken across the members.
    """
    # The observed years
    first_obs_year = int(gmt.year[0])
    last_obs_year = int(gmt.year[-1])
    # The years whose windows lie within them: (t,)
    years = np.arange(first_obs_year + years_before,
                      last_obs_year - years_after + 1)
    # Each observed member's value of the definition in each year's window:
    # one (o,) array per year, stacked: (t, o)
    values = np.stack([
        definition(gmt.sel(year=slice(year - years_before,
                                      year + years_after)).to_numpy())
        for year in years])
    # Their percentiles across the members: (sigmas, t)
    percentiles = np.percentile(values, sigmas, axis=1)
    # One CSV column per percentile, each along the years
    columns = {('Obs', sigma): percentiles[sigmas.index(sigma)]
               for sigma in sigmas}
    # The table, one CSV row per year
    table = pd.DataFrame(columns, index=years)
    table.columns.names = ['variable', 'percentile']
    table.index.name = 'Year'
    return table


# The memory of the one float64 copy correlations makes of a chunk of
# members: 128 MiB. What it does and doesn't affect:
# - The result: not at all, beyond the last digit or so of a float64. The
#   chunks only split the sums over members into partial sums, which add up
#   to the same totals in a different order.
# - The memory: the values correlations is given are already in memory
#   (float32, 4 bytes each: 19 GB for `all`, 790 million members × 6
#   variables). Its sums need float64 (8 bytes each), so it copies the
#   values to float64 first. Copying all of them at once would add 38 GB;
#   copying one chunk at a time adds at most this, 128 MiB.
# - The speed: each chunk is one matrix product, so above ~1e5 members a
#   chunk the loop's own overhead is negligible. 128 MiB is ~2.8 million
#   members with 6 variables: 3 chunks in production (6 million), ~280 for
#   `all`. The production run's whole fidelity step takes ~2 s.
# It is a size in bytes, not a number of members, so it adapts to the
# number of variables (as members.EMPIRICALLY_TUNED_REBUILD_BLOCK_BYTES
# does), and is a round number well below any node's memory, not tuned.
CORRELATION_CHUNK_BYTES = 2**27


def correlations(values):
    """The correlations between variables across members.

    values:
        array (variables, members), e.g. every member's value of each
        variable in one year
    Returns an array (variables, variables): element [v, w] is the Pearson
    correlation of variables v and w across the members; NaN where a
    variable has a NaN (e.g. Res after the observations end).

    THE MATHS, step by step (the comments in the code name each step):

    1. The correlation. For two variables x and y over N members m, with
       means x̄ and ȳ:

           cov(x, y) = (1/N) Σₘ (xₘ − x̄)(yₘ − ȳ)   how they move together
           var(x)    = cov(x, x)
           r(x, y)   = cov(x, y) / √(var(x) · var(y))

    2. Why in chunks, and the covariance rewritten for it.
       - Written as in 1, it needs every member at once: first for x̄,
         then to measure each member from it. The values are float32 and
         the sums must be float64 (step 4), so that would mean a float64
         copy of every member, twice the values' size: 38 GB for `all`.
         So the members are taken a chunk at a time (see
         CORRELATION_CHUNK_BYTES), which needs a form of the covariance
         built from sums over members alone.
       - Multiplying out (x̄ and ȳ are constants, so the mean of xₘȳ is
         x̄ȳ):

           (xₘ − x̄)(yₘ − ȳ) = xₘyₘ − xₘȳ − x̄yₘ + x̄ȳ

           cov(x, y) = (1/N) Σₘ xₘyₘ − x̄ȳ − x̄ȳ + x̄ȳ
                     = (1/N) Σₘ xₘyₘ − x̄ȳ,
                       with x̄ = (1/N) Σₘ xₘ and ȳ = (1/N) Σₘ yₘ

         the mean of the products less the product of the means. So r
         needs only two kinds of sum over members: Σₘ xₘ for each variable,
         and Σₘ xₘyₘ for each pair (Σₘ xₘ² being the pair x, x).
       - A sum over all members is the sum of the chunks' sums. So each
         chunk adds its sums to running totals, and the means, covariances
         and correlations are formed once, from the totals, at the end.
         (The chunks' own correlations couldn't be combined like this:
         each sees only the spread within itself, not the spread between
         the chunks' means.)

    3. Every pair's sum of products, by one matrix product. With the chunk
       as a matrix X, (variables, members), X Xᵀ is (variables, variables):

           (X Xᵀ)[v, w] = Σₘ X[v, m] · X[w, m]

       the sum over members of v times w. For two variables:

           X = ⎡x₁ x₂ … xₙ⎤      X Xᵀ = ⎡Σ xₘxₘ  Σ xₘyₘ⎤
               ⎣y₁ y₂ … yₙ⎦             ⎣Σ yₘxₘ  Σ yₘyₘ⎦

       the sums of squares on the diagonal, the sums of cross products off
       it (Σ xₘyₘ = Σ yₘxₘ, so it is symmetric): every sum step 2 needs
       of a product, for every pair at once.

    4. Precision. Step 2 subtracts x̄ȳ from (1/N) Σₘ xₘyₘ, and the two are
       close when the mean is large next to the spread (e.g. Tot: mean
       ~1.4 K, spread ~0.1 K, so (1/N) Σ xₘ² ≈ 1.97 and x̄² ≈ 1.96); every
       digit they share is lost in the difference. Two safeguards:
       - the sums are float64 (~16 significant digits), not float32 (~7);
       - each variable is first shifted by a constant a (its first
         member's value), so the values summed are small differences
         around the distribution and nothing large cancels. A shift
         changes no covariance, since each value's distance from the mean
         is the same:

           (xₘ − a) − mean(x − a) = xₘ − x̄
           so cov(x − a, y − b) = cov(x, y), and r with it.

       Together they keep r within ~1e-14 of np.corrcoef at any chunk
       size; the chunk size changes only the order of the additions.
    """
    # The sizes: variables, members
    n_vars, n_members = values.shape
    # The members in each chunk, so its float64 copy (8 bytes a value) is
    # CORRELATION_CHUNK_BYTES (step 2)
    members_per_chunk = max(1, CORRELATION_CHUNK_BYTES // (8 * n_vars))
    # Step 4: the constant a each variable is shifted by, its first
    # member's value: (v,)
    shift = values[:, 0].astype(np.float64)

    # Step 2: the running totals over members, of Σₘ xₘ for each variable,
    # (v,), and of Σₘ xₘyₘ for each pair, (v, v)
    sums = np.zeros(n_vars)
    product_sums = np.zeros((n_vars, n_vars))
    for start in range(0, n_members, members_per_chunk):
        # This chunk's values, copied to float64 (step 4): X, (v, chunk)
        chunk = values[:, start:start + members_per_chunk].astype(np.float64)
        # ... shifted by a (step 4), in place, so no second copy
        chunk -= shift[:, None]
        # Step 2: add this chunk's Σₘ xₘ for each variable
        sums += chunk.sum(axis=1)
        # Step 3: add this chunk's Σₘ xₘyₘ for every pair at once, X Xᵀ
        product_sums += chunk @ chunk.T

    # Step 2: the means from the totals, x̄ = (1/N) Σₘ xₘ (of the shifted
    # values): (v,)
    means = sums / n_members
    # Step 2: the covariances, (1/N) Σₘ xₘyₘ − x̄ȳ for every pair, the
    # outer product being the grid of x̄ȳ: (v, v), each variable's
    # variance on the diagonal
    covariances = product_sums / n_members - np.outer(means, means)
    # Step 1: the standard deviations, √var(x): (v,)
    deviations = np.sqrt(np.diag(covariances))
    # Step 1: the correlations, cov(x, y) / √(var(x) · var(y)), the outer
    # product being the grid of the deviations' products: (v, v), 1 on
    # the diagonal
    return covariances / np.outer(deviations, deviations)


def fidelity_rows(label, parent_percentiles, parent_correlations, values,
                  sigmas, output_vars, pairs, prefixes):
    """How close a release is to its parent (the run's whole ensemble) in
    one headline year: the GWI_headlines-fidelity CSV's rows for that year.

    label:
        the year's label, as in the headlines CSV (e.g. 2025, '2016-2025')
    parent_percentiles:
        array (sigmas, variables), the parent's percentiles (the summary's)
    parent_correlations:
        array (variables, variables), the parent's correlations (see
        correlations)
    values:
        array (M, variables), the released members' values that year, in
        the release's order
    sigmas:
        the percentiles, e.g. gwi.py's SIGMAS
    output_vars:
        the names along the variable axis
    pairs:
        the pairs of variables whose correlations to compare, e.g.
        [('GHG', 'Nat'), ...]
    prefixes:
        the sizes of the release's first members to compare too (e.g. 100,
        500), each taken only if smaller than the release
    Returns a DataFrame: a row for each (label, variable, statistic), the
    statistics each percentile ('5', '17', ...) of each variable and the
    correlation of each pair ('GHG & Nat'); columns: parent, release,
    release - parent, and first k - parent for each prefix.

    Every number is a statistic across members of their one value for the
    headline year, never along the timeseries. Each member has one value
    per definition for that year, even where the definition draws on
    several years' warming to make it (e.g. SR1.5 in 2025, the end point
    of a trend fitted to 2011-2025).
    Each column takes that statistic over a different set of members, and
    the "- parent" columns are plain differences, in the statistic's own
    units:
    - a percentile row, e.g. (2025, GHG, 5): the 5th percentile of GHG in
      2025 across the parent's members, across the release's, and each
      subset's less the parent's, in K;
    - a correlation row, e.g. (2025, GHG & OHF, correlation): the
      correlation of GHG with OHF in 2025 across members (members with a
      high GHG have a low OHF: -0.95), and each subset's less the
      parent's. In production, the parent has -0.953 and first 100 -
      parent is +0.013, so a user who takes only the first 100 members
      would find -0.941.
    """
    # The sets of members compared with the parent: the release, and its
    # first k members, (k, variables) each, which, in the release's order,
    # are themselves spread across the distribution
    subsets = {'release': values,
               **{f'first {k}': values[:k] for k in prefixes
                  if k < len(values)}}
    # The percentiles across each set's members, (sigmas, variables)
    percentiles = {name: np.percentile(subset, sigmas, axis=0)
                   for name, subset in subsets.items()}
    # ... and the correlations across them, (variables, variables)
    correlation = {name: correlations(subset.T)
                   for name, subset in subsets.items()}

    # A row for each percentile of each variable
    rows, index = [], []
    for v, var in enumerate(output_vars):
        for s, sigma in enumerate(sigmas):
            # The parent's value, the release's, and each set's difference
            # from the parent's
            parent = parent_percentiles[s, v]
            rows.append([parent, percentiles['release'][s, v]]
                        + [percentiles[name][s, v] - parent
                           for name in subsets])
            index.append((label, var, str(sigma)))
    # A row for each pair's correlation, in the same columns
    for first, second in pairs:
        v, w = output_vars.index(first), output_vars.index(second)
        parent = parent_correlations[v, w]
        rows.append([parent, correlation['release'][v, w]]
                    + [correlation[name][v, w] - parent for name in subsets])
        index.append((label, f'{first} & {second}', 'correlation'))
    # The table
    return pd.DataFrame(
        rows, columns=['parent', 'release'] + [f'{name} - parent'
                                               for name in subsets],
        index=pd.MultiIndex.from_tuples(
            index, names=['Year', 'variable', 'statistic']))


def prior_timeseries_table(T_mod, forcing_vars, regress_vars, years, sigmas):
    """The table of the PRIOR_timeseries-summary CSV: the prior
    warming, stage A's modelled warming itself (the warming before the
    observational constraint, every scaling factor 1), with its percentiles
    over every (parameter set, ERF member) pair, each counted once.

    T_mod:
        array (p, e, t, v), stage A's modelled warming
    forcing_vars, regress_vars:
        the forcing and regressed variables (see src/method/members.py)
    years:
        the years of T_mod's t axis
    sigmas:
        the percentiles to take, e.g. gwi.py's SIGMAS
    Returns:
        a DataFrame indexed by year, with every forcing variable plus the
        aggregates (Ant, Tot), as sums of their parts (which needs the emulator
        to be linear in the forcing).
    """
    # Every forcing variable plus the aggregates: (p, e, t, v'), and the
    # names along v'
    T_prior, prior_vars = attribution.add_aggregate_variables(
        T_mod, forcing_vars, regress_vars)
    # The percentiles over every (parameter set, ERF member) pair:
    # (sigmas, t, v')
    percentiles = np.percentile(T_prior, sigmas, axis=(0, 1))
    # The table, one CSV row per year
    return percentiles_as_csv_table(percentiles, sigmas, prior_vars, years)


def erf_timeseries_table(erf, forcing_vars, regress_vars, sigmas):
    """The table of the ERF_timeseries-summary CSV: the forcing of
    every variable, with its percentiles over the ERF members.

    erf:
        the forcing, a DataArray (year, erf_member, forcing_variable)
    forcing_vars, regress_vars:
        the forcing and regressed variables
    sigmas:
        the percentiles to take, e.g. gwi.py's SIGMAS
    Returns:
        a DataFrame indexed by year, with every forcing variable plus the
        aggregates (Ant, Tot), as sums of their parts.
    """
    # The forcing of every variable, for every ERF member: (t, e, v)
    erf_all = erf.to_numpy()
    # ... plus the aggregates: (t, e, v'), and the names along v'
    erf_all, erf_vars = attribution.add_aggregate_variables(
        erf_all, forcing_vars, regress_vars)
    # The percentiles over the ERF members: (sigmas, t, v')
    percentiles = np.percentile(erf_all, sigmas, axis=1)
    # The table, one CSV row per year
    return percentiles_as_csv_table(percentiles, sigmas, erf_vars,
                                    erf.year.to_numpy())
