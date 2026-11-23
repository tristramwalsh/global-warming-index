"""Percentile summaries of the sample: the GWI's CSV outputs.

Every summary is a reduction along the member axis: for each year and
variable, the percentiles of that year's and variable's attributed warming
across the sample's N members (see src/method/members.py). The members are
rebuilt one block of years at a time (members.Sample.attributed_warming), so
the members of every year are never held at once. The summaries don't
depend on the block size: each year's and variable's percentiles come from
all N of its members, whichever block the year is in.

The arrays' axes are named as in src/method/members.py: t year, and N the
members; `variables` is the axis along sample.output_vars, and `sigmas` the
axis along the percentiles asked for.
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
    # One CSV column per (variable, percentile), each a timeseries: (t,)
    columns = {
        (var, sigma): percentiles[sigmas.index(sigma), :, v]
        for v, var in enumerate(sample.output_vars) for sigma in sigmas
    }
    # The table, one CSV row per year
    table = pd.DataFrame(columns,
                         index=np.arange(first_year, last_year + 1))
    table.columns.names = ['variable', 'percentile']
    table.index.name = 'Year'
    return table


def headline_row(percentiles, sigmas, output_vars, label):
    """One headline as a one-row table, laid out as in the
    GWI_results_headlines CSV.

    percentiles: array (sigmas, variables), the headline's percentiles
        along the member axis, its variables along output_vars
    sigmas: the percentiles taken, e.g. gwi.py's SIGMAS
    output_vars: the names along the variable axis
    label: the row's name in the CSV's Year column, e.g.
        '2024 (SR15 definition)'
    Returns a DataFrame with one row, and a column for each (variable,
    percentile).
    """
    # One CSV column per (variable, percentile), each a single number
    columns = {
        (var, sigma): percentiles[sigmas.index(sigma), v]
        for v, var in enumerate(output_vars) for sigma in sigmas
    }
    # The one-row table
    row = pd.DataFrame(columns, index=[label])
    row.columns.names = ['variable', 'percentile']
    row.index.name = 'Year'
    return row


def sr15_headline_rows(sample, sigmas, headline_years, trend_years):
    """The SR1.5-definition rows of the GWI_results_headlines CSV: the
    warming in each headline year as the value, at the middle of that year,
    of the least-squares line through its last trend_years annual means, the
    year itself included (IPCC SR1.5, Ch. 1, §1.2.1).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    headline_years: the years to give, e.g. [2017, 2025]
    trend_years: how many annual means the line is fitted through, e.g.
        gwi.py's SR15_TREND_YEARS (which explains the choice)
    Returns a list of one-row DataFrames (see headline_row), one per year,
    labelled '<year> (SR15 definition)'.

    E.g. for 2025 with trend_years = 15: the annual means of 2011 to 2025
    inclusive, 15 years of data (1 January 2011 to 31 December 2025), with
    points 14 years apart (mid-2011 to mid-2025), read off at mid-2025 (see
    warming_definitions.final_value_of_trend_vectorised). That equals the
    mean of the line extended over the 29 years 2011 to 2039.

    Each year's window of members is rebuilt on its own: (trend_years,
    variables, N), about 2.0 GiB for 15 years at --samples=60.
    """
    rows = []
    for year in headline_years:
        # Every member's attributed warming over the window, the year and
        # the trend_years - 1 before it: (trend_years, variables, N)
        window = sample.attributed_warming(year - (trend_years - 1), year)
        # Each member's line through the window, along the year axis, at
        # its final year: (variables, N)
        trend_end = warming_definitions.final_value_of_trend_vectorised(
            window)
        # Its percentiles along the member axis, the variables shared out
        # over threads: (sigmas, variables)
        percentiles = compute_resources.percentile_threaded(
            trend_end, sigmas, axis=1)
        rows.append(headline_row(percentiles, sigmas, sample.output_vars,
                                 f'{year} (SR15 definition)'))
    return rows


def ar6_headline_rows(sample, sigmas, headline_years, mean_years):
    """The AR6-definition rows of the GWI_results_headlines CSV: the
    warming over the mean_years years ending in each headline year, the mean
    of their annual means (IPCC AR6 WGI's decadal means).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    headline_years: the windows' last years, e.g. [2019, 2025]
    mean_years: how many annual means each window averages, e.g. gwi.py's
        AR6_MEAN_YEARS
    Returns a list of one-row DataFrames (see headline_row), one per
    window, labelled '<first year>-<last year>'.

    E.g. for 2025 with mean_years = 10: the mean of the annual means of 2016
    to 2025 inclusive, 10 years of data (1 January 2016 to 31 December
    2025), labelled '2016-2025'.

    Each window of members is rebuilt on its own: (mean_years, variables,
    N), about 1.4 GiB for 10 years at --samples=60.
    """
    rows = []
    for year in headline_years:
        # The window's first year
        first_year = year - (mean_years - 1)
        # Every member's attributed warming over the window: (mean_years,
        # variables, N)
        window = sample.attributed_warming(first_year, year)
        # Each member's mean over the window, along the year axis:
        # (variables, N)
        window_mean = window.mean(axis=0)
        # Its percentiles along the member axis, the variables shared out
        # over threads: (sigmas, variables)
        percentiles = compute_resources.percentile_threaded(
            window_mean, sigmas, axis=1)
        rows.append(headline_row(percentiles, sigmas, sample.output_vars,
                                 f'{first_year}-{year}'))
    return rows


def cgwl_headline_rows(sample, sigmas, headline_years, mean_years):
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
    Returns a list of one-row DataFrames (see headline_row), one per year,
    labelled '<first year>-<last year> (CGWL definition)'.

    E.g. for 2025 with mean_years = 19: the mean of the annual means of 2016
    to 2034 inclusive, 19 years of data (1 January 2016 to 31 December
    2034), centred on mid-2025, labelled '2016-2034 (CGWL definition)'.

    Each window of members is rebuilt on its own: (mean_years, variables,
    N), about 2.6 GiB for 19 years at --samples=60.
    """
    rows = []
    for year in headline_years:
        # The window's first and last years
        first_year = year - (mean_years - 1) // 2
        last_year = year + (mean_years - 1) // 2
        # Every member's attributed warming over the window: (mean_years,
        # variables, N)
        window = sample.attributed_warming(first_year, last_year)
        # Each member's mean over the window, along the year axis:
        # (variables, N)
        window_mean = window.mean(axis=0)
        # Its percentiles along the member axis, the variables shared out
        # over threads: (sigmas, variables)
        percentiles = compute_resources.percentile_threaded(
            window_mean, sigmas, axis=1)
        rows.append(headline_row(
            percentiles, sigmas, sample.output_vars,
            f'{first_year}-{last_year} (CGWL definition)'))
    return rows
