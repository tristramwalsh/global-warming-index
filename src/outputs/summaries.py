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
