"""Percentile summaries of the ensemble: the GWI's CSV outputs.

The terms, as used throughout:
    ensemble            every drawn member's values, before percentiles
                        (the member axis, N)
    warming definition  a filter along the years, applied to each window
                        of years: the annual value (windows of one year),
                        the mean over a window (AR6's decade, the CGWL's
                        centred mean), the end point of a line fitted
                        through it (SR1.5) or its slope (the AR6 rate)
    summary             the percentiles of the ensemble, taken along the
                        member axis: what the CSVs give instead of the
                        ensemble itself
    headlines           the headline years (e.g. the latest, and the years
                        of IPCC assessments), whose values of every
                        definition the GWI_results_headlines CSV gathers

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
    windowed_definition_table         both of the above, for any definition
                                      linear in time, over a list of
                                      windows, in blocks
    definition_timeseries_table       a definition's value in every output
                                      year whose window lies within the
                                      output years: each definition's
                                      GWI_results_timeseries-* CSV (gwi.py's
                                      WARMING_DEFINITIONS gives each one's
                                      function and window)
    headline_rows                     a timeseries' rows at the headline
                                      years, labelled for a headlines CSV
    observed_timeseries_table         a definition's timeseries for the
                                      observed temperatures: percentiles
                                      across the observed members, no
                                      rebuild (Obs_results_timeseries-*)
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

import src.infrastructure.compute_resources as compute_resources
import src.method.attribution as attribution
import src.method.emulators as emulators


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
    first then wait while the rest take a second one. Measured at
    --samples=60 (6 million members, 6 variables, 28 threads): 4, 14 and 32
    years per block take 35.6, 35.0 and 34.5 s for the timeseries, and 1.1,
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


def windowed_definition_table(sample, sigmas, definition, windows, labels):
    """The percentiles of a definition linear in time, e.g. the mean over a
    decade, in each window of years: one CSV row per window.

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    definition: a warming definition, a function of an array along its
        first axis, the years, and linear (see
        members.Sample.ensemble_definition), e.g.
        warming_definitions.mean_over_years
    windows: the windows, (first_year, last_year) pairs, both included
    labels: the rows' names, one per window
    Returns a DataFrame (see percentiles_as_csv_table).

    The windows are taken years_per_block at a time: each block rebuilds
    the ensemble's value in its windows, (windows, variables, N), and takes
    its percentiles in one pass over the threads.
    """
    # How many windows to take at once
    windows_per_block = years_per_block(sample)
    # The percentiles of each block of windows: one (sigmas, windows,
    # variables) array per block
    block_percentiles = []
    for start in range(0, len(windows), windows_per_block):
        # The ensemble's value in these windows: (windows, variables, N)
        ensemble = sample.ensemble_definition(
            definition, windows[start:start + windows_per_block])
        # Its percentiles: (sigmas, windows, variables)
        block_percentiles.append(
            percentiles_by_year_and_variable(ensemble, sigmas))
    # Every window's percentiles: (sigmas, windows, variables)
    percentiles = np.concatenate(block_percentiles, axis=1)
    # The table, one CSV row per window
    return percentiles_as_csv_table(percentiles, sigmas,
                                    sample.output_vars, labels)


def definition_timeseries_table(sample, sigmas, definition,
                                first_output_year, last_output_year,
                                years_before, years_after):
    """A definition's timeseries: its percentiles in every output year
    whose window lies within the output years.

    sample:
        members.Sample
    sigmas:
        the percentiles to take, e.g. gwi.py's SIGMAS
    definition:
        a warming definition linear in time (see windowed_definition_table)
    first_output_year, last_output_year:
        the output years (both included)
    years_before, years_after:
        each year's window, from years_before years before it to years_after
        years after it
    Returns:
        a DataFrame indexed by year ('Year').
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
    # The ensemble's value in each window, and its percentiles
    return windowed_definition_table(sample, sigmas, definition, windows,
                                     years)


def headline_rows(timeseries, headline_years, labels):
    """A definition's timeseries at the headline years, labelled as the
    GWI_results_headlines CSV labels them.

    timeseries:
        a DataFrame indexed by year, e.g. from definition_timeseries_table
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


def observed_timeseries_table(df_obs, sigmas, definition, years_before,
                              years_after):
    """A warming definition's timeseries for the observed temperatures: its
    percentiles across the observed members, in every year whose window
    lies within the observed years.

    df_obs:
        DataFrame, the observed members' temperatures, indexed by consecutive
        years, one column per member, (t, o)
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
        percentile), as the Obs_results_headlines CSV lays them out.

    The observations need no rebuild: the definition is applied to each
    observed member directly, and the percentiles taken across the members.
    """
    # The observed years
    first_obs_year, last_obs_year = df_obs.index[0], df_obs.index[-1]
    # The years whose windows lie within them: (t,)
    years = np.arange(first_obs_year + years_before,
                      last_obs_year - years_after + 1)
    # Each observed member's value of the definition in each year's window:
    # one (o,) array per year, stacked: (t, o)
    values = np.stack([
        definition(df_obs.loc[year - years_before:
                              year + years_after].to_numpy())
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


def prior_timeseries_table(T_mod, forcing_vars, regress_vars, years, sigmas):
    """The table of the PRIOR_results_timeseries CSV: the prior warming,
    stage A's modelled warming itself (the warming before the observational
    constraint, every scaling factor 1), with its percentiles over every
    (parameter set, ERF member) pair, each counted once.

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


def erf_timeseries_table(df_forc, forcing_vars, regress_vars, sigmas):
    """The table of the ERF_results_timeseries CSV: the forcing of every
    variable, with its percentiles over the ERF members.

    df_forc:
        the forcing, columns (variable, ensemble), indexed by year
    forcing_vars, regress_vars:
        the forcing and regressed variables
    sigmas:
        the percentiles to take, e.g. gwi.py's SIGMAS
    Returns:
        a DataFrame indexed by year, with every forcing variable plus the
        aggregates (Ant, Tot), as sums of their parts.
    """
    # The forcing of every variable, for every ERF member: (t, e, v)
    erf_all = emulators.forcing_array(
        df_forc, forcing_vars,
        df_forc.columns.get_level_values('ensemble').unique())
    # ... plus the aggregates: (t, e, v'), and the names along v'
    erf_all, erf_vars = attribution.add_aggregate_variables(
        erf_all, forcing_vars, regress_vars)
    # The percentiles over the ERF members: (sigmas, t, v')
    percentiles = np.percentile(erf_all, sigmas, axis=1)
    # The table, one CSV row per year
    return percentiles_as_csv_table(percentiles, sigmas, erf_vars,
                                    df_forc.index.to_numpy())
