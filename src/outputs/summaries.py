"""Percentile summaries of the sample: the GWI's CSV outputs.

Every summary is a reduction along the member axis: for each year (or
window of years) and variable, the percentiles across the sample's N
members (see src/method/members.py). Two kinds:
- the timeseries: each member's attributed warming, rebuilt one block of
  years at a time (members.Sample.attributed_warming);
- the headline definitions and rates, all linear in time (a mean over a
  window, the end point or slope of a line fitted through it): each
  member's summary of each window, rebuilt from T_mod and T_obs summarised
  first, one block of windows at a time
  (members.Sample.attributed_definition).
So the members of every year are never held at once, and the summaries
don't depend on the block size.

How the functions connect:
    percentiles_by_year_and_variable  members (t, variables, N) ->
                                      percentiles (sigmas, t, variables)
    percentiles_as_csv_table          percentiles -> a DataFrame laid out
                                      as the CSVs are (a row per year or
                                      window, a column per (variable,
                                      percentile))
    gwi_timeseries_table              the GWI_results_timeseries CSV: both
                                      of the above, over blocks of years
    windowed_definition_table         the same for any definition linear in
                                      time, over a list of windows
    sr15_headline_table,              the SR1.5, AR6 and CGWL rows of the
    ar6_headline_table,               GWI_results_headlines CSV (which gwi.py
    cgwl_headline_table               joins with the annual rows, taken from
                                      the timeseries), and the
    ar6_rate_table                    GWI_results_rates CSV: each names its
                                      windows, labels and definition, and
                                      calls windowed_definition_table

The arrays' axes are named as in src/method/members.py: t year (or window
of years), and N the members; `variables` is the axis along
sample.output_vars, and `sigmas` the axis along the percentiles asked for.
"""

import numpy as np
import pandas as pd

import src.infrastructure.compute_resources as compute_resources
import src.method.warming_definitions as warming_definitions


def years_per_block(sample):
    """How many years of members to rebuild at once.

    sample: members.Sample

    Each (year, variable) pair is its own percentile calculation along the
    member axis, so a block of t years holds t × variables calculations,
    which are shared out over the threads. The block is the most years
    whose calculations don't outnumber the threads, ⌊threads / variables⌋,
    and at least one: e.g. 4 years for 6 variables on 28 threads, which is
    24 calculations for 28 threads.

    Bigger blocks gain nothing. The rebuild is no faster in bigger blocks,
    since each member's scaling factors are summed once for the whole
    sample. Nor are the percentiles, once every thread has a calculation; a
    few calculations more than threads is slower, as the threads that finish
    first then wait while the rest take a second one. Measured at
    --samples=60 (6 million members, 6 variables, 28 threads): 4, 14 and 32
    years per block take 35.6, 35.0 and 34.5 s for the timeseries, and 1.1,
    3.8 and 8.7 GiB per block (a block's members, and the copy the
    percentiles make of them); 5 years (30 calculations) took 38.7 s.
    """
    return max(1, sample.n_threads // len(sample.output_vars))


def block_bytes(sample):
    """The most memory a summary here holds at once: a block of
    years_per_block years (or windows) of every member, and the copy the
    percentiles make of it, 2 × years_per_block × sample.bytes_per_year.

    sample: members.Sample
    The timeseries, the headline definitions and the rates all work in such
    blocks, so this holds whichever outputs the run gives.
    """
    return 2 * years_per_block(sample) * sample.bytes_per_year


def year_blocks(first_year, last_year, years_per_block):
    """Split the years first_year to last_year into blocks of
    years_per_block years (the last block may be shorter).

    Returns a list of (first, last) pairs, both ends included: e.g.
    year_blocks(1850, 1861, 5) is [(1850, 1854), (1855, 1859), (1860, 1861)].
    """
    blocks = []
    # The first year of the current block
    first = first_year
    while first <= last_year:
        # Its last year: years_per_block years on, or the final year
        last = min(first + years_per_block - 1, last_year)
        blocks.append((first, last))
        # The next block starts the year after
        first = last + 1
    return blocks


def percentiles_by_year_and_variable(members_warming, sigmas):
    """The percentiles of the attributed warming along the member axis, for
    each year and variable.

    members_warming: array (t, variables, N), every member's attributed
        warming, e.g. over a block of years
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    Returns an array (sigmas, t, variables): element [s, i, v] is the
    sigmas[s]-th percentile of members_warming[i, v, :], the N members of
    year i and variable v.

    It is np.percentile(members_warming, sigmas, axis=-1) (in xarray,
    a quantile along the member dimension), with the (year, variable) pairs
    shared out over the job's threads, so a block of a few years still uses
    every CPU.
    """
    # The sizes: years, variables, members
    n_years, n_vars, n_members = members_warming.shape
    # The year and variable axes merged into one axis of (year, variable)
    # pairs, pair i · variables + v being year i and variable v, so the
    # threads can share out the pairs (percentile_threaded shares out the
    # first axis): a view, (t · variables, N)
    pairs = members_warming.reshape(n_years * n_vars, n_members)
    # The percentiles of each pair along the member axis, over threads:
    # (sigmas, t · variables)
    pair_percentiles = compute_resources.percentile_threaded(
        pairs, sigmas, axis=1)
    # The pair axis split back into years and variables: (sigmas, t,
    # variables)
    return pair_percentiles.reshape(len(sigmas), n_years, n_vars)


def gwi_timeseries_table(sample, sigmas, first_year, last_year):
    """The table of the GWI_results_timeseries CSV: the percentiles of the
    attributed warming, in every year from first_year to last_year.

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_year, last_year: the output years (both included)
    Returns a DataFrame indexed by year ('Year'), with a column for each
    (variable, percentile), the variables in the order of
    sample.output_vars.
    """
    # The percentiles of each block of years: one (sigmas, t, variables)
    # array per block
    block_percentiles = []
    for first, last in year_blocks(first_year, last_year,
                                   years_per_block(sample)):
        # Every member's attributed warming over these years: (t,
        # variables, N)
        members_warming = sample.attributed_warming(first, last)
        # Its percentiles: (sigmas, t, variables)
        block_percentiles.append(
            percentiles_by_year_and_variable(members_warming, sigmas))
    # Every year's percentiles, the blocks joined along the year axis:
    # (sigmas, t, variables)
    percentiles = np.concatenate(block_percentiles, axis=1)
    # The table, one CSV row per year
    return percentiles_as_csv_table(percentiles, sigmas, sample.output_vars,
                                    np.arange(first_year, last_year + 1))


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
    definition: a function summarising an array along its first axis, the
        years, and linear (see members.Sample.attributed_definition), e.g.
        warming_definitions.mean_over_years
    windows: the windows, (first_year, last_year) pairs, both included
    labels: the rows' names, one per window
    Returns a DataFrame (see percentiles_as_csv_table).

    The windows are taken years_per_block at a time: each block rebuilds
    every member's summary of its windows, (windows, variables, N), and
    takes their percentiles in one pass over the threads.
    """
    # How many windows to take at once
    windows_per_block = years_per_block(sample)
    # The percentiles of each block of windows: one (sigmas, windows,
    # variables) array per block
    block_percentiles = []
    for start in range(0, len(windows), windows_per_block):
        # Every member's summary in these windows: (windows, variables, N)
        members_summary = sample.attributed_definition(
            definition, windows[start:start + windows_per_block])
        # Their percentiles: (sigmas, windows, variables)
        block_percentiles.append(
            percentiles_by_year_and_variable(members_summary, sigmas))
    # Every window's percentiles: (sigmas, windows, variables)
    percentiles = np.concatenate(block_percentiles, axis=1)
    # The table, one CSV row per window
    return percentiles_as_csv_table(percentiles, sigmas,
                                    sample.output_vars, labels)


def sr15_headline_table(sample, sigmas, headline_years, trend_years):
    """The SR1.5-definition rows of the GWI_results_headlines CSV: the
    warming in each headline year as the value, at the middle of that year,
    of the least-squares line through its last trend_years annual means, the
    year itself included (IPCC SR1.5, Ch. 1, §1.2.1).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    headline_years: the years to give, e.g. [2017, 2025]
    trend_years: how many annual means the line is fitted through, e.g.
        gwi.py's SR15_TREND_YEARS (which explains the choice)
    Returns a DataFrame with a row for each year, labelled '<year> (SR15
    definition)'.

    E.g. for 2025 with trend_years = 15: the annual means of 2011 to 2025
    inclusive, 15 years of data (1 January 2011 to 31 December 2025), with
    points 14 years apart (mid-2011 to mid-2025), read off at mid-2025 (see
    warming_definitions.final_value_of_trend_vectorised). That equals the
    mean of the line extended over the 29 years 2011 to 2039.
    """
    # Each year's window: the year and the trend_years - 1 before it
    windows = [(year - (trend_years - 1), year) for year in headline_years]
    labels = [f'{year} (SR15 definition)' for year in headline_years]
    # The value of each member's fitted line at the end of each window
    return windowed_definition_table(
        sample, sigmas, warming_definitions.final_value_of_trend_vectorised,
        windows, labels)


def ar6_headline_table(sample, sigmas, headline_years, mean_years):
    """The AR6-definition rows of the GWI_results_headlines CSV: the
    warming over the mean_years years ending in each headline year, the mean
    of their annual means (IPCC AR6 WGI's decadal means).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    headline_years: the windows' last years, e.g. [2019, 2025]
    mean_years: how many annual means each window averages, e.g. gwi.py's
        AR6_MEAN_YEARS
    Returns a DataFrame with a row for each window, labelled '<first
    year>-<last year>'.

    E.g. for 2025 with mean_years = 10: the mean of the annual means of 2016
    to 2025 inclusive, 10 years of data (1 January 2016 to 31 December
    2025), labelled '2016-2025'.
    """
    # Each window: the year and the mean_years - 1 before it
    windows = [(year - (mean_years - 1), year) for year in headline_years]
    labels = [f'{first}-{last}' for first, last in windows]
    # Each member's mean over each window
    return windowed_definition_table(
        sample, sigmas, warming_definitions.mean_over_years, windows, labels)


def cgwl_headline_table(sample, sigmas, headline_years, mean_years):
    """The CGWL-definition rows of the GWI_results_headlines CSV: the
    current global warming level (Betts et al., 2023) in each headline
    year, the mean of mean_years annual means centred on that year,
    (mean_years - 1) / 2 either side of it.

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    headline_years: the years to give, e.g. [2025]; each window must lie
        within the sample's years, so after the observations it uses the
        projected forcing
    mean_years: how many annual means each window averages, an odd
        number, e.g. gwi.py's CGWL_MEAN_YEARS (which explains the choice)
    Returns a DataFrame with a row for each year, labelled '<first
    year>-<last year> (CGWL definition)'.

    E.g. for 2025 with mean_years = 19: the mean of the annual means of 2016
    to 2034 inclusive, 19 years of data (1 January 2016 to 31 December
    2034), centred on mid-2025, labelled '2016-2034 (CGWL definition)'.
    """
    # Each window: (mean_years - 1) / 2 years either side of the year
    half = (mean_years - 1) // 2
    windows = [(year - half, year + half) for year in headline_years]
    labels = [f'{first}-{last} (CGWL definition)' for first, last in windows]
    # Each member's mean over each window
    return windowed_definition_table(
        sample, sigmas, warming_definitions.mean_over_years, windows, labels)


def ar6_rate_table(sample, sigmas, first_year, last_year, trend_years):
    """The table of the GWI_results_rates CSV: in every year from
    first_year to last_year, the AR6-definition warming rate, the slope (°C
    per year) of the least-squares line through that year's last
    trend_years annual means, the year itself included (IPCC AR6 WGI, Ch.
    3, Table 3.1).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_year, last_year: the years whose rates to give (both included)
    trend_years: how many annual means each line is fitted through, e.g.
        gwi.py's AR6_RATE_YEARS
    Returns a DataFrame with a row for each year, labelled '<first year of
    its trend>-<year> (AR6 rate definition)', e.g. '2016-2025 (AR6 rate
    definition)'.
    """
    # Each year's window: the year and the trend_years - 1 before it
    years = range(first_year, last_year + 1)
    windows = [(year - (trend_years - 1), year) for year in years]
    labels = [f'{first}-{last} (AR6 rate definition)'
              for first, last in windows]
    # The slope of each member's fitted line through each window
    return windowed_definition_table(
        sample, sigmas, warming_definitions.rate_func_vectorised, windows,
        labels)
