"""The historical-only GWI: for each year, the values from the regression
that ended in that year. For combine_results_iterations.py.
"""
import src.postprocessing.results_layout as results_layout


def map_headline_to_index(
        headline=None, year=None, index_str=None, invert=False):
    """Map headline year to string format used in headlines dataframe."""

    if not invert:
        mapping = {
            'ANNUAL': str(year),
            'SR15': f'{year} (SR15 definition)',
            'AR6': f'{year-9}-{year}',
            'CGWL': f'{year-9}-{year+10} (CGWL definition)',
        }

        return mapping.get(headline, None)

    elif invert:
        if 'CGWL' in index_str:
            return 'CGWL'
        elif 'SR15' in index_str:
            return 'SR15'
        # Check whether index_str is of the form 'YYYY-YYYY'
        elif '-' in index_str and index_str.replace('-', '').isdigit():
            return 'AR6'
        else:
            return 'ANNUAL'


def get_available_headlines(results_dfs, scen, ens, reg_vars, reg_ranges_all):
    """Determine available headlines from the datasets.

    Note this doesn't work out all the years that each headline is available
    for; we just need to know which headlines are available for any regression
    range for this configuration; the actual years will be handled later in
    combine_historical_only, which handles gaps in the data on a case by case
    basis.
    """
    # Assume to start that we do have headlines available
    available_headlines = set()

    for reg_range in reg_ranges_all:
        if 'headlines' not in results_dfs[scen][ens][reg_vars][reg_range]:
            continue
        else:  # we do have headline file
            df = results_dfs[scen][ens][reg_vars][reg_range]['headlines']
            for index_str in df.index:
                available_headlines.add(
                    map_headline_to_index(
                        index_str=index_str, invert=True)
                )

    # Check if available_headlines is empty:
    if not available_headlines:
        valid_headlines = ['ANNUAL']
        headline_toggle = False
    else:
        valid_headlines = list(available_headlines)
        headline_toggle = True

    print('      Valid headlines: ',  valid_headlines)
    return valid_headlines, headline_toggle


def calculate_historical_only(
    results_dfs
):
    """Derive the historical-only datasets and save them as CSVs.

    For every scenario, selection and set of regressed variables with more
    than one regressed range, and for each headline available (ANNUAL, AR6,
    SR15, CGWL), builds a timeseries in which each year's value comes from
    the regression that ends in that year, and writes it to
    AGGREGATED_FOLDER (combine_historical_only). Plots nothing.

    In the pipeline: step 4 of the script, run with --re-calculate=y. Step 5,
    load_historical_only_dfs, reads the CSVs back for plotting.
    """

    print('\nGenerating historical-only timeseries')
    for scen in sorted(results_dfs.keys()):
        print('INPUTS:', scen)

        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)

            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED-VARIABLES:', reg_vars)

                # Get all available regressed year ranges
                reg_ranges_all = sorted(
                    list(results_dfs[scen][ens][reg_vars].keys()))

                if len(reg_ranges_all) == 1:
                    # No historical years to compare against, so skip to the
                    # next set of variables.
                    print('      Only one regressed range available; ' +
                          'skipping historical-only calculation.')
                    continue

                # The +PRE variant also includes the years before the
                # earliest regressed range, but with the same headline
                # definitions as the historical-only dataset. This is
                # inconsistent with the way the historical-only dataset is
                # calculated, but is included as a reference for plotting.
                # If I really want a full-information (instead of
                # historical-only) dataset using the various definitions, this
                # will need doing inside GWI.py (and could easily be added
                # using a new argv of 'all' alongside 'y' and 'n' in the
                # headline_toggles).

                # Determine which headline definitions to calculate the
                # historical-only timeseries for.

                headlines, headline_toggle = get_available_headlines(
                    results_dfs, scen, ens, reg_vars, reg_ranges_all)

                print(f'      Available headlines: {headlines}')
                print(f'      Using headlines dataframe: {headline_toggle}')

                # Determine whether to pull the headline from the headlines or
                # timeseries dataframe. You can only pull annual years from the
                # both headlines and timeseries dataframes. The value of
                # selecting, is that the headlines are much more
                # computationally expensive to calculate, do you may not always
                # calculate the headlines for all regressed_year ranges.

                for headline in headlines:
                    combine_historical_only(
                        scen, ens, reg_vars, reg_ranges_all,
                        headline, headline_toggle,
                        results_dfs)


def combine_historical_only(scen, ens, reg_vars, reg_ranges_all,
                            headline, headline_toggle, results_dfs):
    """Calculate historical-only timeseries for each headline."""

    # Identift range of regressed years
    min_regressed_range = min(reg_ranges_all)
    max_regressed_range = max(reg_ranges_all)

    print('      Creating historical-only timeseries for '
          f'{headline}: between ' +
          f'{min_regressed_range} and {max_regressed_range}')

    # Prepare empty timeseries for each headline
    df_hist_headline = results_dfs[
        scen][ens][reg_vars][reg_ranges_all[0]]['timeseries'].copy()
    df_hist_headline[:] = 0

    # Collect the appropriate regressed range results
    for reg_range in reg_ranges_all:
        # Extract the relevant headline names for this regressed range
        current_year = int(reg_range.split('-')[1])
        headline_index = map_headline_to_index(headline, current_year)

        # Select the type of dataset to pull the results from.
        res_type = 'headlines' if headline_toggle else 'timeseries'

        # Check availability of this headline time for this configuration
        if headline_index in results_dfs[scen
                                         ][ens
                                           ][reg_vars
                                             ][reg_range
                                               ][res_type
                                                 ].index:
            # Pull out the relevant headline/time data
            _df = results_dfs[scen
                              ][ens
                                ][reg_vars
                                  ][reg_range
                                    ][res_type
                                      ].loc[headline_index]
            df_hist_headline.loc[current_year] = _df

        else:
            pass  # headline not available for this regressed range

    # Remove all years that are not the end of an attribution
    # period to avoid confusion (i.e. the longer earlier years
    # before the historical-only focus period).
    end_years = [int(reg_range.split('-')[1])
                 for reg_range in reg_ranges_all]
    smallest_end_year = min(end_years)
    largest_end_year = max(end_years)

    # Filter the dataframe to only include the years that are
    # relevant for the historical-only dataset: these are years
    # that are >= smallest_end_year and <= largest_end_year.
    df_hist_headline = df_hist_headline.loc[
        smallest_end_year:largest_end_year, :]

    # Remove any rows that contain on the value zero
    df_hist_headline = df_hist_headline.loc[
        (df_hist_headline != 0).any(axis=1)]

    # Check whether the dataframe is empty and save if not. This fixes an issue
    # where the CGWL isn't calculated in some years in some scenarios
    # (the latter years of observed-202x) due to no "future" ERFs being
    # available. In this case the CGWL csv would be saved, but empty, which
    # then produced an error later on in the code.
    if not df_hist_headline.empty:
        df_hist_headline.to_csv(
            f'{results_layout.AGGREGATED_FOLDER}/{scen}/' +
            f'ENSEMBLE-MEMBER--{ens}/' +
            f'VARIABLES--{reg_vars}/'
            f'GWI_results_{headline}_HISTORICAL-ONLY_' +
            f'{scen}_' +
            f'ENSEMBLE-MEMBER--{ens}_' +
            f'VARIABLES--{reg_vars}_' +
            f'REGRESSED-YEARS--{min_regressed_range}' +
            f'_to_{max_regressed_range}.csv')
