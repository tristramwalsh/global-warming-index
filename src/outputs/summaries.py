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
                                      output years
    gwi_timeseries_table,             each definition's timeseries CSV
    sr15_timeseries_table,            (GWI_results_timeseries, -SR15, -AR6,
    ar6_timeseries_table,             -CGWL) and the GWI_results_rates CSV:
    cgwl_timeseries_table,            each names its window and definition
    ar6_rate_table
    headline_rows                     a timeseries' rows at the headline
                                      years, labelled for the
                                      GWI_results_headlines CSV, which gwi.py
                                      joins from every definition

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
    # The output years: (t,)
    years = np.arange(first_year, last_year + 1)
    # Each year on its own, a window of one year
    windows = [(year, year) for year in years]
    # Each member's value in each year
    return windowed_definition_table(
        sample, sigmas, warming_definitions.annual_value, windows, years)


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

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    definition: a warming definition linear in time (see
        windowed_definition_table)
    first_output_year, last_output_year: the output years (both included)
    years_before, years_after: each year's window, from years_before
        years before it to years_after years after it
    Returns a DataFrame indexed by year ('Year').
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


def sr15_timeseries_table(sample, sigmas, first_output_year,
                          last_output_year, trend_years):
    """The table of the GWI_results_timeseries-SR15 CSV: in every output
    year whose window fits, the warming as the value, at the middle of that
    year, of the least-squares line through its last trend_years annual
    means, the year itself included (IPCC SR1.5, Ch. 1, §1.2.1).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_output_year, last_output_year: the output years (both included)
    trend_years: how many annual means the line is fitted through, e.g.
        gwi.py's SR15_TREND_YEARS (which explains the choice)
    Returns a DataFrame indexed by year ('Year').

    E.g. for 2025 with trend_years = 15: the annual means of 2011 to 2025
    inclusive, 15 years of data (1 January 2011 to 31 December 2025), with
    points 14 years apart (mid-2011 to mid-2025), read off at mid-2025 (see
    warming_definitions.final_value_of_trend_vectorised). That equals the
    mean of the line extended over the 29 years 2011 to 2039.
    """
    # The value of each member's fitted line at the end of each window
    return definition_timeseries_table(
        sample, sigmas, warming_definitions.final_value_of_trend_vectorised,
        first_output_year, last_output_year,
        years_before=trend_years - 1, years_after=0)


def ar6_timeseries_table(sample, sigmas, first_output_year,
                         last_output_year, mean_years):
    """The table of the GWI_results_timeseries-AR6 CSV: in every output
    year whose window fits, the warming over the mean_years years ending in
    it, the mean of their annual means (IPCC AR6 WGI's decadal means).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_output_year, last_output_year: the output years (both included)
    mean_years: how many annual means each window averages, e.g. gwi.py's
        AR6_MEAN_YEARS
    Returns a DataFrame indexed by year ('Year'), each year being its
    window's last.

    E.g. for 2025 with mean_years = 10: the mean of the annual means of 2016
    to 2025 inclusive, 10 years of data (1 January 2016 to 31 December
    2025).
    """
    # Each member's mean over each window
    return definition_timeseries_table(
        sample, sigmas, warming_definitions.mean_over_years,
        first_output_year, last_output_year,
        years_before=mean_years - 1, years_after=0)


def cgwl_timeseries_table(sample, sigmas, first_output_year,
                          last_output_year, mean_years):
    """The table of the GWI_results_timeseries-CGWL CSV: in every output
    year whose window fits, the current global warming level (Betts et al.,
    2023), the mean of mean_years annual means centred on that year,
    (mean_years - 1) / 2 either side of it.

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_output_year, last_output_year: the output years (both included);
        so the series stops (mean_years - 1) / 2 years before the last,
        which, after the observations end, uses the projected forcing
    mean_years: how many annual means each window averages, an odd
        number, e.g. gwi.py's CGWL_MEAN_YEARS (which explains the choice)
    Returns a DataFrame indexed by year ('Year'), each year being its
    window's centre.

    E.g. for 2025 with mean_years = 19: the mean of the annual means of 2016
    to 2034 inclusive, 19 years of data (1 January 2016 to 31 December
    2034), centred on mid-2025.
    """
    # Each window: (mean_years - 1) / 2 years either side of the year
    half = (mean_years - 1) // 2
    # Each member's mean over each window
    return definition_timeseries_table(
        sample, sigmas, warming_definitions.mean_over_years,
        first_output_year, last_output_year,
        years_before=half, years_after=half)


def ar6_rate_table(sample, sigmas, first_output_year, last_output_year,
                   trend_years):
    """The table of the GWI_results_rates CSV: in every output year whose
    window fits, the AR6-definition warming rate, the slope (°C per year)
    of the least-squares line through that year's last trend_years annual
    means, the year itself included (IPCC AR6 WGI, Ch. 3, Table 3.1).

    sample: members.Sample
    sigmas: the percentiles to take, e.g. gwi.py's SIGMAS
    first_output_year, last_output_year: the output years (both included)
    trend_years: how many annual means each line is fitted through, e.g.
        gwi.py's AR6_RATE_YEARS
    Returns a DataFrame with a row for each year, labelled '<first year of
    its trend>-<year> (AR6 rate definition)', e.g. '2016-2025 (AR6 rate
    definition)', as the rates CSV has always been.
    """
    # The slope of each member's fitted line through each window
    rates = definition_timeseries_table(
        sample, sigmas, warming_definitions.rate_func_vectorised,
        first_output_year, last_output_year,
        years_before=trend_years - 1, years_after=0)
    # Each row labelled by its trend's years
    labels = [f'{year - (trend_years - 1)}-{year} (AR6 rate definition)'
              for year in rates.index]
    return headline_rows(rates, rates.index, labels)


def headline_rows(timeseries, headline_years, labels):
    """A definition's timeseries at the headline years, labelled as the
    GWI_results_headlines CSV labels them.

    timeseries: a DataFrame indexed by year, e.g. from sr15_timeseries_table
    headline_years: the years to take, each in timeseries' index
    labels: the rows' names, one per year, e.g. '2016-2025' or '2025 (SR15
        definition)'
    Returns a DataFrame with those rows, indexed by the labels ('Year').
    """
    # The headline years' rows
    rows = timeseries.loc[list(headline_years)].copy()
    # ... named by their labels
    rows.index = pd.Index(labels, name='Year')
    return rows
