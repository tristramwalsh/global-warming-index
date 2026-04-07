import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from highlight_text import ax_text
from PIL import Image
import src.graphing as gr
import src.definitions as defs
import multiprocessing as mp
import functools
from pprint import pprint


PLOT_FOLDER = 'plots/'
AGGREGATED_FOLDER = 'results/aggregated'
ITERATIONS_FOLDER = 'results/iterations'


def get_subdirs(path, prefix):
    """Get sorted list of subdirectories starting with a prefix."""
    if not os.path.exists(path):
        return []
    return sorted([
        d.split(prefix)[1] for d in os.listdir(path)
        if d.startswith(prefix) and os.path.isdir(os.path.join(path, d))
    ])


def setup_plot_params(scen, ens, reg_vars, results_dfs):
    """Extract common plotting parameters."""
    reg_ranges_all = sorted(
        [item for item in results_dfs[scen][ens][reg_vars].keys()
            if item != 'HISTORICAL-ONLY']
        )
    min_regressed_range = reg_ranges_all[0]
    max_regressed_range = reg_ranges_all[-1]
    smallest_end_year = int(min_regressed_range.split('-')[1])
    largest_end_year = int(max_regressed_range.split('-')[1])

    first_range = reg_ranges_all[0]
    plot_vars = results_dfs[scen][ens][reg_vars][first_range][
        'timeseries'].columns.get_level_values(0).unique().to_list()
    all_var_colours, scaling_map = gr.get_dynamic_colours(
        reg_vars, plot_vars, gr.VAR_COLOURS)

    return {
        'min_range': min_regressed_range,
        'max_range': max_regressed_range,
        'start_year': smallest_end_year,
        'end_year': largest_end_year,
        'plot_vars': plot_vars,
        'colours': all_var_colours,
        'scaling_map': scaling_map,
        'reg_ranges_all': reg_ranges_all
    }


def check_headlines_files(scenario, ensemble_selection,
                          regressed_vars, regressed_years_vars):
    """Check if any headline files exist for the given configuration."""
    for reg_year in regressed_years_vars:
        path = (f'{ITERATIONS_FOLDER}/'
                f'SCENARIO--{scenario}/'
                f'ENSEMBLE-MEMBER--{ensemble_selection}/'
                f'VARIABLES--{regressed_vars}/'
                f'REGRESSED-YEARS--{reg_year}/')
        if os.path.exists(path):
            for f in os.listdir(path):
                if 'headlines' in f:
                    return True
    return False


def check_rates_files(scenario, ensemble_selection,
                      regressed_vars, regressed_years_vars):
    """Check if any rates files exist for the given configuration."""
    for reg_year in regressed_years_vars:
        path = (f'{ITERATIONS_FOLDER}/'
                f'SCENARIO--{scenario}/'
                f'ENSEMBLE-MEMBER--{ensemble_selection}/'
                f'VARIABLES--{regressed_vars}/'
                f'REGRESSED-YEARS--{reg_year}/')
        if os.path.exists(path):
            for f in os.listdir(path):
                if 'rates' in f:
                    return True
    return False


def calculate_iteration_averages():
    """Average the timeseries, headlines, and rates iterations."""

    scenarios_all = get_subdirs(ITERATIONS_FOLDER, 'SCENARIO--')
    print(scenarios_all)

    for scenario in scenarios_all:
        print('Calculating SCENARIO:', scenario)

        ensemble_selections = get_subdirs(
            f'{ITERATIONS_FOLDER}/SCENARIO--{scenario}/', 'ENSEMBLE-MEMBER--')

        for ensemble_selection in ensemble_selections:
            print('  Calculating ensemble selection:', ensemble_selection)

            regressed_variables_all = get_subdirs(
                f'{ITERATIONS_FOLDER}/SCENARIO--{scenario}/'
                f'ENSEMBLE-MEMBER--{ensemble_selection}/',
                'VARIABLES--')

            print('    All regressed variables for scenario:',
                  regressed_variables_all)

            for regressed_vars in regressed_variables_all:
                print('      Calculating regressed variables:', regressed_vars)
                _path = (f'{ITERATIONS_FOLDER}/' +
                         f'SCENARIO--{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')

                regressed_years_vars = get_subdirs(_path, 'REGRESSED-YEARS--')

                if defs.check_steps(regressed_years_vars)['check_bool']:
                    print(f'        All regressed years for {regressed_vars}:',
                          defs.check_steps(regressed_years_vars)['range'])

                # Check if headlines are available
                headline_toggle = check_headlines_files(
                    scenario, ensemble_selection,
                    regressed_vars, regressed_years_vars)

                # Check if rates are available
                rate_toggle = check_rates_files(
                    scenario, ensemble_selection,
                    regressed_vars, regressed_years_vars)

                result_types_to_process = ['timeseries']
                if headline_toggle:
                    result_types_to_process.append('headlines')
                if rate_toggle:
                    result_types_to_process.append('rates')

                with mp.Pool(os.cpu_count()) as p:
                    print('        Calculating (parallel regressed_years) ',
                          'for:',
                          scenario, ensemble_selection, regressed_vars)
                    for res_type in result_types_to_process:
                        p.map(
                            functools.partial(
                                combine_repeats,
                                result_type=res_type, scenario=scenario,
                                ensemble_selection=ensemble_selection,
                                regressed_vars=regressed_vars),
                            regressed_years_vars)


def combine_repeats(regressed_years, result_type, scenario, ensemble_selection,
                    regressed_vars):
    """
    Average results across iterations for a specific configuration.

    Args:
        regressed_years: The range of years used for regression.
        result_type: The type of result (e.g., 'timeseries', 'headlines', 'rates').
        scenario: The scenario name.
        ensemble_selection: The ensemble selection name.
        regressed_vars: The regressed variables.

    Returns:
        A tuple containing the averaged DataFrame, a dictionary of all
        iterations, and a dictionary of ensemble sizes, or (None, None, None)
        if no files found.
    """
    dict_iterations = {}
    size_iterations = {}

    base_path = (
        f'{ITERATIONS_FOLDER}/'
        f'SCENARIO--{scenario}/'
        f'ENSEMBLE-MEMBER--{ensemble_selection}/'
        f'VARIABLES--{regressed_vars}/'
        f'REGRESSED-YEARS--{regressed_years}/'
    )

    if not os.path.exists(base_path):
        print(f'Path not found: {base_path}')
        return None, None, None

    iteration_files = [
        f for f in os.listdir(base_path)
        if result_type in f
    ]

    if len(iteration_files) == 0:
        print('No iterations found for:',
              result_type, scenario, ensemble_selection,
              regressed_years, regressed_vars)
        return None, None, None

    # Remove previously averaged dataset in case it already exists

    for iteration in iteration_files:
        fname = os.path.join(base_path, iteration)
        ens_size = int(fname.split('ENSEMBLE-SIZE--')[-1].split('_')[0])
        df_iteration = pd.read_csv(
            fname, index_col=0,  header=[0, 1], skiprows=0)
        dict_iterations[iteration] = df_iteration
        size_iterations[iteration] = ens_size

    # Produce the averaged dataset
    df_avg = (dict_iterations[iteration_files[0]].copy() *
              size_iterations[iteration_files[0]])
    df_avg[:] = 0
    for iteration in iteration_files:
        df_avg += dict_iterations[iteration] * size_iterations[iteration]
    # df_avg /= len(iterations)
    df_avg /= sum(size_iterations.values())

    # Create the specific directory for these regressed years
    out_path = base_path.replace(ITERATIONS_FOLDER, AGGREGATED_FOLDER)
    os.makedirs(out_path, exist_ok=True)

    df_avg.to_csv(
        f'{out_path}' +
        f'GWI_results_{result_type}_' +
        f'SCENARIO--{scenario}_'
        f'ENSEMBLE-MEMBER--{ensemble_selection}_' +
        f'VARIABLES--{regressed_vars}_' +
        f'REGRESSED-YEARS--{regressed_years}_' +
        'AVERAGE.csv')

    return df_avg, dict_iterations, size_iterations


def load_nested_dfs(d):
    """Return nested dictionary with DataFrames instead of file paths."""
    if isinstance(d, dict):
        return {k: load_nested_dfs(v) for k, v in d.items()}
    elif isinstance(d, str):
        if os.path.exists(d):
            return pd.read_csv(d, index_col=0, header=[0, 1], skiprows=0)
        return None
    return d


def load_gwi_priors_erf_obs():
    """Load all averaged datasets."""
    results_files = {}
    priors_files = {}
    erf_files = {}
    obs_files = {}

    scenarios_all = get_subdirs(AGGREGATED_FOLDER, 'SCENARIO--')

    for scenario in scenarios_all:
        results_files.update({scenario: {}})
        priors_files.update({scenario: {}})
        erf_files.update({scenario: {}})
        obs_files.update({scenario: {}})

        _path = f'{AGGREGATED_FOLDER}/SCENARIO--{scenario}/'

        ensembles_seletions_all = get_subdirs(_path, 'ENSEMBLE-MEMBER--')

        for ensemble_selection in ensembles_seletions_all:
            results_files[scenario].update({ensemble_selection: {}})
            priors_files[scenario].update({ensemble_selection: {}})
            erf_files[scenario].update({ensemble_selection: {}})
            obs_files[scenario].update({ensemble_selection: {}})

            # Load priors files
            _path_prior_dir = ('results/priors/' +
                               f'SCENARIO--{scenario}/' +
                               f'ENSEMBLE-MEMBER--{ensemble_selection}/')
            _path_erf_dir = ('results/erfs/' +
                             f'SCENARIO--{scenario}/' +
                             f'ENSEMBLE-MEMBER--{ensemble_selection}/')
            if os.path.exists(_path_prior_dir):
                for f in os.listdir(_path_prior_dir):
                    if f.startswith('PRIOR_results_timeseries_'):
                        priors_files[scenario][ensemble_selection][
                            'timeseries'] = os.path.join(_path_prior_dir, f)
            if os.path.exists(_path_erf_dir):
                for f in os.listdir(_path_erf_dir):
                    if f.startswith('ERF_results_timeseries_'):
                        erf_files[scenario][ensemble_selection][
                            'timeseries'] = os.path.join(_path_erf_dir, f)

            _path = (f'{AGGREGATED_FOLDER}/' +
                     f'SCENARIO--{scenario}/' +
                     f'ENSEMBLE-MEMBER--{ensemble_selection}/')

            regressed_variables_all = get_subdirs(_path, 'VARIABLES--')

            for regressed_vars in regressed_variables_all:
                results_files[scenario][ensemble_selection].update(
                    {regressed_vars: {}})

                _path = (f'{AGGREGATED_FOLDER}/' +
                         f'SCENARIO--{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')

                regressed_years_vars = get_subdirs(_path, 'REGRESSED-YEARS--')

                for regressed_years in regressed_years_vars:
                    # Load GWI results files
                    res_type_dict = {
                        res_type: (
                                f'{AGGREGATED_FOLDER}/' +
                                f'SCENARIO--{scenario}/' +
                                f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                                f'VARIABLES--{regressed_vars}/' +
                                f'REGRESSED-YEARS--{regressed_years}/' +
                                f'GWI_results_{res_type}_' +
                                f'SCENARIO--{scenario}_'
                                f'ENSEMBLE-MEMBER--{ensemble_selection}_' +
                                f'VARIABLES--{regressed_vars}_' +
                                f'REGRESSED-YEARS--{regressed_years}_' +
                                'AVERAGE.csv'
                            )
                        for res_type in ['timeseries', 'headlines', 'rates']
                    }
                    results_files[scenario
                                  ][ensemble_selection
                                    ][regressed_vars
                                      ].update({
                                          regressed_years: res_type_dict
                                          })

                    # Load observations files
                    obs_files[scenario
                              ][ensemble_selection
                                ].update({regressed_years: {}})

                    _path_obs_dir = ('results/observations/' +
                                     f'SCENARIO--{scenario}/' +
                                     f'ENSEMBLE-MEMBER--{ensemble_selection}/'
                                     f'REGRESSED-YEARS--{regressed_years}/')
                    # The observation headlines don't change for different
                    # regression variables, which means that the file only
                    # needs to be loaded once. It is faster to check this than
                    # access os.listdir multiple times and keep overwriting the
                    # file path in the dictionary.
                    if 'headlines' not in obs_files[scenario
                                                    ][ensemble_selection
                                                      ][regressed_years]:
                        if os.path.exists(_path_obs_dir):
                            for f in os.listdir(_path_obs_dir):
                                if f.startswith('Obs_results_headlines_'):
                                    obs_files[scenario
                                              ][ensemble_selection
                                                ][regressed_years
                                                  ]['headlines'
                                                    ] = \
                                        os.path.join(_path_obs_dir, f)

    print('\nLoading all averaged datasets')
    # Load results timeseries and headlines
    results_dfs = load_nested_dfs(results_files)
    # Load priors timeseries
    priors_dfs = load_nested_dfs(priors_files)
    # Load ERF timeseries
    erf_dfs = load_nested_dfs(erf_files)
    # Load observation headlines
    obs_dfs = load_nested_dfs(obs_files)

    # Load temperature observations timeseries
    print('Loading temperature observations timeseries...')
    for scen in results_dfs.keys():
        for ens in results_dfs[scen].keys():
            # Parse ensemble string for GMT
            # e.g. pull the 7 (or similar) out of: GMT-7_ERF-all
            try:
                ens_GMT = {combo.split('-')[0]: combo.split('-')[1]
                           for combo in ens.split('_')
                           }['GMT']

                scen_in = scen.split('_const-ERF')[0]

                df_temp_Obs = defs.load_Temp(
                    scenario=scen_in, ensemble_members=ens_GMT,
                    start_pi=1850, end_pi=1900)

                obs_dfs[scen][ens]['timeseries'] = df_temp_Obs
            except Exception as e:
                print('Warning: Could not load temperature observations for '
                      f'{scen} {ens}: {e}')

    return results_dfs, priors_dfs, erf_dfs, obs_dfs


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
    """Generate historical-only timeseries and plot them."""

    print('\nGenerating historical-only timeseries')
    for scen in sorted(results_dfs.keys()):
        print('SCENARIO:', scen)

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
            f'{AGGREGATED_FOLDER}/SCENARIO--{scen}/' +
            f'ENSEMBLE-MEMBER--{ens}/' +
            f'VARIABLES--{reg_vars}/'
            f'GWI_results_{headline}_HISTORICAL-ONLY_' +
            f'SCENARIO--{scen}_' +
            f'ENSEMBLE-MEMBER--{ens}_' +
            f'VARIABLES--{reg_vars}_' +
            f'REGRESSED-YEARS--{min_regressed_range}' +
            f'_to_{max_regressed_range}.csv')


def load_historical_only_dfs(results_dfs):
    """Load historical-only datasets; add to the results_files dictionary."""
    print('\nLoading historical-only datasets into results_dfs structure')
    # Load historical-only data from csv files if they are available
    # Iterate over the structure identified in results_files
    # This is reasonable because historical-only datasets can only exist in
    # the same directories/configurations as the main results datasets.
    for scen in results_dfs.keys():
        for ens in results_dfs[scen].keys():
            for reg_vars in results_dfs[scen][ens].keys():
                # Find all historical dataset in the results/aggregated folder
                # and add them all to the dictionary.
                _path = (f'{AGGREGATED_FOLDER}/' +
                         f'SCENARIO--{scen}/' +
                         f'ENSEMBLE-MEMBER--{ens}/' +
                         f'VARIABLES--{reg_vars}/')
                hist_files = [
                    f for f in os.listdir(_path)
                    if f.startswith('GWI_results_') and 'HISTORICAL-ONLY' in f]

                for hist_file in hist_files:
                    _df = pd.read_csv(_path + hist_file,
                                      index_col=0, header=[0, 1], skiprows=0)
                    headline_name = hist_file.split('GWI_results_')[1].split(
                        '_HISTORICAL-ONLY')[0]

                    if 'HISTORICAL-ONLY' not in results_dfs[scen
                                                            ][ens
                                                              ][reg_vars]:
                        results_dfs[scen][ens][reg_vars][
                            'HISTORICAL-ONLY'] = {}

                    results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'].update({headline_name: _df})

    return results_dfs


def is_dataset_present(data_dict, required_keys):
    """
    Check if the required datasets are present in the provided dictionary
    and are not None.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing datasets, typically results_dfs[scen][ens][reg_vars][reg_range]
    required_keys : list or str
        List of keys (or single key) that must be present and not None.
        
    Returns:
    --------
    bool
        True if all required datasets are present and not None, False otherwise.
    """
    if isinstance(required_keys, str):
        required_keys = [required_keys]
        
    for key in required_keys:
        if key not in data_dict or data_dict[key] is None:
            return False
    return True


def figure_timeseries(reg_range, scen, ens, reg_vars,
                      results_dfs, df_temp_Obs, params
                      ):
    """Plot single timeseries plots."""
    # print('Creating single timeseries plots for:',
    #       scen, ens, reg_vars, reg_range, end='\r')

    # Get all variables present in the data
    df_ts = results_dfs[scen][ens][reg_vars][reg_range]['timeseries']
    all_data_vars = df_ts.columns.get_level_values(0).unique().to_list()

    # Define major variables (for plumes)
    major_vars = reg_vars.split('-')
    major_vars.extend(defs.extra_vars(major_vars))
    plume_vars = major_vars

    # Define linestyles for all variables
    var_linestyles = gr.get_dynamic_linestyles(all_data_vars)

    # Determine legend location
    sub_vars = [v for v in all_data_vars if v not in plume_vars]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    reg_start = int(reg_range.split('-')[0])
    reg_end = int(reg_range.split('-')[1])
    trunc_start = df_ts.index.min()
    trunc_end = df_ts.index.max()

    # print(results_dfs[scen][ens][reg_vars][reg_range]['timeseries'])

    if df_ts.loc[reg_end:, :].empty:
        print(f'No data for: {reg_range} {scen} {ens} {reg_vars}')
        print(df_ts)

    gr.gwi_timeseries(
        ax, df_temp_Obs, None,
        df_ts.loc[reg_end:, :],
        all_data_vars, params['colours'], hatch='x', linestyle='dashed',
        plume_vars=plume_vars)

    gr.gwi_timeseries(
        ax, df_temp_Obs, None,
        df_ts.loc[reg_start:reg_end, :],
        all_data_vars, params['colours'], linestyle=var_linestyles,
        plume_vars=plume_vars)

    ax.set_ylim(
        np.floor(np.min(df_ts.drop(columns='Res', level=0).values) * 2) / 2,
        # np.ceil(np.max(df_temp_Obs.values) * 2) / 2,
        np.ceil(np.max(df_ts.drop(columns='Res', level=0).values) * 2) / 2
    )
    # ax.set_ylim(-2,5)

    ax.set_xlim(trunc_start, trunc_end+1)

    if sub_vars:
        gr.overall_legend(fig, 'center right', 1,
                          reorder=gr.get_legend_reorder_indices(fig))
        plt.subplots_adjust(right=0.77)
    else:
        gr.overall_legend(fig, 'lower center', 7,
                          reorder=gr.get_legend_reorder_indices(fig))

    # Plot a box around the regressed years if the final truncation year is not
    # the end of the regressed years.
    if int(trunc_end) != int(reg_end):
        ax.axvline(int(reg_range.split('-')[1]),
                   color='darkslategray', linestyle='--')

    # Add title
    fig.text(ax.get_position().x0, ax.get_position().y1+0.02,
             'Global Warming Index Timeseries',
             ha='left',
             fontsize=plt.rcParams['axes.titlesize'],
             fontweight='bold'
             )

    # Add configuration text
    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars} | '
                     f'Regressed range: {reg_range}')
    if configuration:
        fig.text(0.5, 0.01, configuration, ha='center',
                 fontsize='x-small', fontfamily='monospace',
                 )

    plot_path = ('plots/aggregated/' +
                 f'SCENARIO--{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    plot_name = (f'{plot_path}/' +
                 f'Timeseries_Scenario--{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    # plot_names.append(plot_name)
    fig.savefig(plot_name)
    plt.close(fig)
    return plot_name


def figure_rates(reg_range, scen, ens, reg_vars,
                 results_dfs, df_temp_Obs, params
                 ):
    """Plot single rates plots."""
    # Get all variables present in the data
    df_ts = results_dfs[scen][ens][reg_vars][reg_range]['rates']
    all_data_vars = df_ts.columns.get_level_values(0).unique().to_list()

    # Transform the index of df_ts to numeric - it is currently of the form
    # '1941-1950 (AR6 rate definition)' and we want that to be '1950' for
    # plotting.
    df_ts.index = df_ts.index.to_series().apply(
        lambda x: int(x.split('-')[1].split()[0]) if '-' in x else int(x.split()[0])
        )

    # Define major variables (for plumes)
    major_vars = reg_vars.split('-')
    major_vars.extend(
        [v for v in ['Tot', 'Ant', 'Nat', 'Res'] if v in all_data_vars]
        )
    plume_vars = major_vars

    # Define linestyles for all variables
    var_linestyles = gr.get_dynamic_linestyles(all_data_vars)

    # Determine legend location
    sub_vars = [v for v in all_data_vars if v not in plume_vars]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    reg_start = int(reg_range.split('-')[0])
    reg_end = int(reg_range.split('-')[1])
    trunc_start = df_ts.index.min()
    trunc_end = df_ts.index.max()

    if not df_ts.loc[reg_end:, :].empty:
        gr.gwi_timeseries(
            ax, None, None,
            df_ts.loc[reg_end:, :],
            all_data_vars, params['colours'], hatch='x', linestyle='dashed',
            plume_vars=plume_vars, ylabel='Warming Rate')

    gr.gwi_timeseries(
        ax, None, None,
        df_ts.loc[reg_start:reg_end, :],
        all_data_vars, params['colours'], linestyle=var_linestyles,
        plume_vars=plume_vars, ylabel='Warming Rate')

    try:
        if 'Res' in df_ts.columns.get_level_values(0):
            df_for_ylim = df_ts.drop(columns='Res', level=0)
        else:
            df_for_ylim = df_ts
        y_min = np.floor(np.nanmin(df_for_ylim.values) * 10) / 40
        y_max = np.ceil(np.nanmax(df_for_ylim.values) * 10) / 20
        ax.set_ylim(y_min, y_max)
    except Exception:
        pass

    ax.set_xlim(trunc_start, trunc_end+1)

    if sub_vars:
        gr.overall_legend(fig, 'center right', 1,
                          reorder=gr.get_legend_reorder_indices(fig))
        plt.subplots_adjust(right=0.77)
    else:
        gr.overall_legend(fig, 'lower center', 7,
                          reorder=gr.get_legend_reorder_indices(fig))

    if int(trunc_end) != int(reg_end):
        ax.axvline(int(reg_range.split('-')[1]),
                   color='darkslategray', linestyle='--')

    # Add title
    fig.text(ax.get_position().x0, ax.get_position().y1+0.02,
             'Global Warming Index Rates',
             ha='left',
             fontsize=plt.rcParams['axes.titlesize'],
             fontweight='bold'
             )

    # Add configuration text
    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars} | '
                     f'Regressed range: {reg_range}')
    if configuration:
        fig.text(0.5, 0.01, configuration, ha='center',
                 fontsize='x-small', fontfamily='monospace',
                 )

    plot_path = ('plots/aggregated/' +
                 f'SCENARIO--{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    plot_name = (f'{plot_path}/' +
                 f'Rates_Scenario--{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    fig.savefig(plot_name)
    plt.close(fig)
    return plot_name


def figure_spm2(
        reg_range, scen, ens, reg_vars,
        results_dfs, obs_dfs,
        params):
    """Plot single SPM2 bar plot."""

    # Get headlines
    df_headlines = results_dfs[scen][ens][reg_vars][reg_range]['headlines']

    # Get observations headlines
    obs_dict = obs_dfs[scen][ens][reg_range]
    if is_dataset_present(obs_dict, 'headlines'):
        df_obs_headlines = obs_dict['headlines']
    else:
        df_obs_headlines = None

    periods = list(df_headlines.index)
    if not periods:
        return

    # Determine variables for SPM2 panels 2 and 3.
    possible_vars_p2 = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat', 'Res']
    vars_panel2 = [v for v in possible_vars_p2
                   if (v, '50') in df_headlines.columns]
    for period in periods:
        # Panel 3: Components
        vars_panel3 = []
        if defs.SUB_VAR_MAPPING:
            for group in ['GHG', 'OHF', 'Nat']:
                if group in defs.SUB_VAR_MAPPING:
                    # Identify available variables in this group
                    group_vars = [
                        sub_var for sub_var in defs.SUB_VAR_MAPPING[group]
                        if (sub_var, '50') in df_headlines.columns]
                    # Sort by median value (largest to smallest)
                    group_vars.sort(
                        key=lambda v: df_headlines.loc[period, (v, '50')],
                        reverse=True)
                    vars_panel3.extend(group_vars)

        # Calculate grid dimensions based on the number of variables in each
        # panel in order to make the bars in each panel the same visual width.
        # Panel 1 is fixed width of 3 for padding around observations.
        x_width_1 = 3
        x_width_2 = max(len(vars_panel2), 1)
        spacer = 1

        if vars_panel3:
            x_width_3 = max(len(vars_panel3), 1)
            total_width = x_width_1 + spacer + x_width_2 + spacer + x_width_3
        else:
            x_width_3 = 0
            total_width = x_width_1 + spacer + x_width_2

        # Create figure and axes
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot2grid(
            (1, total_width), (0, 0), colspan=x_width_1, fig=fig)
        ax2 = plt.subplot2grid(
            (1, total_width), (0, x_width_1 + spacer),
            colspan=x_width_2, fig=fig)
        axes = [ax1, ax2]
        if vars_panel3:
            ax3 = plt.subplot2grid(
                (1, total_width), (0, x_width_1 + spacer + x_width_2 + spacer),
                colspan=x_width_3, fig=fig)
            axes.append(ax3)

        # Calculate dynamic ylim
        vals_min = []
        vals_max = []
        for _df in [df_headlines, df_obs_headlines]:
            if _df is not None and period in _df.index:
                vals_max.append(_df.loc[period, (slice(None), '95')].max())
                vals_min.append(_df.loc[period, (slice(None), '5')].min())
        lower_ylim = np.floor(min(vals_min) * 2) / 2
        upper_ylim = np.ceil(max(vals_max) * 2) / 2
        ylim = (lower_ylim, upper_ylim)

        # Panel 1: Observed
        if df_obs_headlines is not None and period in df_obs_headlines.index:
            gr.plot_spm2_panel(axes[0], df_obs_headlines, period, ['Obs'],
                               params['colours'], defs.VAR_NAMES,
                               ylim, show_ylabel=True, show_yticklabels=True,
                               xlim=(-1.5, 1.5))

        # Panel 2: Aggregated
        gr.plot_spm2_panel(axes[1], df_headlines, period, vars_panel2,
                           params['colours'], defs.VAR_NAMES,
                           ylim, show_ylabel=False, show_yticklabels=False)

        # Panel 3: Components
        if vars_panel3 and len(axes) > 2:
            gr.plot_spm2_panel(axes[2], df_headlines, period, vars_panel3,
                               params['colours'], defs.VAR_NAMES,
                               ylim, show_ylabel=False,
                               show_yticklabels=False)

        fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.85))

        # Add text
        fig.text(axes[0].get_position().x0, axes[0].get_position().y1+0.08,
                 f'Observed warming and contributions ({period})',
                 fontsize=plt.rcParams['axes.titlesize'],
                 fontweight='bold',
                 )
        fig.text(axes[0].get_position().x0, axes[0].get_position().y1+0.02,
                 '(a) Observed warming',
                 ha='left',
                 fontsize=plt.rcParams['font.size'],
                 fontweight='regular',
                 #  fontstyle='italic'
                 )
        # fig.text(axes[1].get_position().x0, axes[1].get_position().y1+0.08,
        #          ('Contributions to observed warming'),
        #          fontsize=plt.rcParams['axes.titlesize'],
        #          fontweight='bold'
        #          )
        fig.text(axes[1].get_position().x0, axes[1].get_position().y1+0.02,
                 ('(b) Aggregated contributions'),
                 fontsize=plt.rcParams['font.size'],
                 fontweight='regular'
                 )
        if len(axes) > 2:
            fig.text(axes[2].get_position().x0, axes[2].get_position().y1+0.02,
                     ('(c) Component contributions'),
                     fontsize=plt.rcParams['font.size'],
                     fontweight='regular'
                     )

        # Create plot
        configuration = (f'Scenario: {scen} | '
                         f'Ensemble: {ens} | '
                         f'Regressed variables: {reg_vars} | '
                         f'Regressed range: {reg_range}')
        if configuration:
            fig.text(0.5, 0.01, configuration, ha='center',
                     fontsize='x-small', fontfamily='monospace',
                     )

        # Draw arrows for Ant <- GHG + OHF
        y_offsets = {
            'Ant': 0.185,
            'GHG': 0.215,
            'OHF': 0.165
        }
        if set('GHG-OHF-Nat'.split('-')).issubset(set(vars_panel2)):
            gr.draw_grouping_arrow(axes[1], vars_panel2, 'Ant', ['GHG', 'OHF'],
                                   y_offsets=y_offsets, line_y_offset=0.26)

        # Set the grid to the back for the fig
        for ax in axes:
            ax.set_axisbelow(True)

        # Save plot
        plot_path = ('plots/aggregated/' +
                     f'SCENARIO--{scen}/' +
                     f'ENSEMBLE-MEMBER--{ens}/' +
                     f'VARIABLES--{reg_vars}/' +
                     f'REGRESSED-YEARS--{reg_range}/')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)

        period_token = str(period).replace(' ', '-')
        plot_name = (f'{plot_path}/' +
                     f'SPM2_BarPlot_Scenario--{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        fig.savefig(plot_name)
        plt.close(fig)


def figure_waterfall(
        reg_range, scen, ens, reg_vars,
        results_dfs, obs_dfs,
        params):
    """Plot single waterfall plot (Horizontal Design with Subtotals)."""

    # Get headlines
    df_headlines = results_dfs[scen][ens][reg_vars][reg_range]['headlines']
    
    obs_dict = obs_dfs[scen][ens][reg_range]
    if is_dataset_present(obs_dict, 'headlines'):
        df_obs_headlines = obs_dict['headlines']
    else:
        df_obs_headlines = None

    periods = list(df_headlines.index)
    if not periods:
        return

    for period in periods:
        # Helper to get stats
        def get_stats(v, df=df_headlines):
            if (df is not None and period in df.index
                    and (v, '50') in df.columns):
                med = df.loc[period, (v, '50')]
                low = df.loc[period, (v, '5')]
                high = df.loc[period, (v, '95')]
                return med, low, high
            else:
                return 0, 0, 0

        # 1. Identify variables and sort them
        plot_items = []

        # Helper to add sorted components
        def add_components(source_vars):
            # Filter and sort components
            vars_in_group = [v for v in source_vars
                             if (v, '50') in df_headlines.columns]
            # Sort from largest to smallest warming contribution
            vars_in_group.sort(key=lambda v: get_stats(v)[0], reverse=True)

            for v in vars_in_group:
                plot_items.append({'var': v, 'type': 'component'})

        # Define the structure of the waterfall
        # GHG Group
        add_components(defs.SUB_VAR_MAPPING['GHG'])
        plot_items.append({'var': 'GHG', 'type': 'subtotal'})

        # OHF Group
        add_components(defs.SUB_VAR_MAPPING['OHF'])
        plot_items.append({'var': 'OHF', 'type': 'subtotal'})

        # Ant Total
        plot_items.append({'var': 'Ant', 'type': 'total'})

        # Nat Group
        add_components(defs.SUB_VAR_MAPPING['Nat'])
        plot_items.append({'var': 'Nat', 'type': 'subtotal'})

        # Tot Total
        plot_items.append({'var': 'Tot', 'type': 'total'})

        # Res (Components only)
        add_components(['Res'])

        # Obs Total
        plot_items.append({'var': 'Obs', 'type': 'total'})

        # 2. Prepare plot
        # Increase height to accommodate more bars
        fig, ax = plt.subplots(figsize=(13, 13))

        # Initialize limits
        min_val = 0
        max_val = 0

        # Invert Y axis logic: Start from top
        y_pos = 0
        current_left = 0

        bar_height_component = 0.7
        bar_height_aggregate = 0.35
        bar_alpha_component = 0.6
        bar_alpha_aggregate = 1.0
        edge_colour = 'none'
        err_colour = '#444444'

        # Store positions for connecting lines
        component_positions = []  # (y, start_x, end_x)

        # Manually specify yticks and labels to enable arrows to be added to
        # the labels
        yticks = []
        yticklabels = []

        # Iterate and Plot
        for item in plot_items:
            var = item['var']
            label = defs.VAR_NAMES.get(var, var)
            item_type = item['type']

            # Get Data
            if var == 'Obs':
                med, low, high = get_stats(var, df_obs_headlines)
            else:
                med, low, high = get_stats(var)
            neg_err = med - low
            pos_err = high - med

            if item_type == 'component':
                # Waterfall Component
                left = current_left

                # Update limits
                min_val = min(min_val, left + low, left + high)
                max_val = max(max_val, left + low, left + high)

                # Plot Bar
                ax.barh(
                    y_pos, med,
                    left=left,
                    height=bar_height_component,
                    xerr=[[neg_err], [pos_err]],
                    color=params['colours'][var],
                    edgecolor=edge_colour,
                    alpha=bar_alpha_component,
                    error_kw=dict(
                        lw=1, capsize=3, capthick=1, ecolor=err_colour)
                    )

                # Store for lines
                component_positions.append(
                    {'y': y_pos, 'start': left, 'end': left + med})

                # Update accumulator
                current_left += med

                # Label arrow to show direction of flow and aggregation
                yticklabels.append(f"{label}  ↓ ")

            elif item_type in ['subtotal', 'total']:

                # Update limits
                min_val = min(min_val, low)
                max_val = max(max_val, high)

                # Make the axhlne the same colour as the bar to signify
                # aggregate
                ax.axhline(y=y_pos, color=params['colours'][var], linewidth=1.5)

                # Plot Bar
                ax.barh(
                    y_pos, med,
                    left=0,  # Bar starts from the axis
                    height=bar_height_aggregate,
                    xerr=[[neg_err], [pos_err]],
                    color=params['colours'][var],
                    edgecolor=edge_colour,
                    alpha=bar_alpha_aggregate,
                    error_kw=dict(
                        lw=1, capsize=3, capthick=1, ecolor=err_colour)
                    )

                yticklabels.append(label)

                # Add Explanatory Text
                s = ""
                highlight_textprops = []

                if var == 'Ant':
                    s = (
                        f"Sum of <{defs.VAR_NAMES['GHG']}> and "
                        f"<{defs.VAR_NAMES['OHF']}>"
                    )
                    highlight_textprops = [
                        {"color": params['colours']['GHG'],
                         "fontweight": "bold"},
                        {"color": params['colours']['OHF'],
                         "fontweight": "bold"}
                    ]
                elif var == 'Tot':
                    s = (
                        f"Sum of <{defs.VAR_NAMES['Ant']}> and "
                        f"<{defs.VAR_NAMES['Nat']}>"
                    )
                    highlight_textprops = [
                        {"color": params['colours']['Ant'],
                         "fontweight": "bold"},
                        {"color": params['colours']['Nat'],
                         "fontweight": "bold"}
                    ]
                elif var == 'Obs':
                    s = (
                        f"Sum of <{defs.VAR_NAMES['Tot']}> and "
                        f"<{defs.VAR_NAMES['Res']}>"
                    )
                    highlight_textprops = [
                        {"color": params['colours']['Tot'],
                         "fontweight": "bold"},
                        {"color": params['colours']['Res'],
                         "fontweight": "bold"}
                    ]
                else:
                    s = "Sum of <components>"
                    highlight_textprops = [
                        {"color": params['colours'].get(var, 'black')}
                    ]

                if s:
                    ax_text(x=0.02, y=y_pos + bar_height_aggregate/2 + 0.1,
                            s=s,
                            highlight_textprops=highlight_textprops,
                            ax=ax,
                            fontsize=10,
                            fontweight='regular',
                            color='#555555',
                            ha='left',
                            va='bottom')

            yticks.append(y_pos)

            # Add gap after totals
            if item_type in ['subtotal', 'total']:
                y_pos -= 1.7
            else:
                y_pos -= 1.0

        # Add padding and set limits
        x_range = max_val - min_val
        ax.set_xlim(min_val - x_range * 0.1, max_val + x_range * 0.1)

        ################################################
        # Draw Connecting Lines for Waterfall Components
        ################################################

        # We need to connect the *end* of one component to the *start* of the
        # next component. Visually, the waterfall flow should persist across
        # the subtotals.

        # NOTE: The aggregates (subtotals GHG,OHF,Nat,Ant,Tot) will not
        # necessarily line up perfectly with the ends of the component sums
        # due to the the fact that these are percentiles across large
        # ensembles and a multi-run mean of those percentiles. In reality, at
        # the ensemble-member level, the variables will sum up to give the Obs
        # (e.g. Tot + Res = Obs) exactly.

        # Define destinations for the lines starting from each component
        # For component i, the line goes to component i+1.
        # For the last component, the line goes to Obs.
        destinations = [{'y': p['y'], 'h': bar_height_component}
                        for p in component_positions[1:]]
        destinations.append({'y': yticks[-1], 'h': bar_height_aggregate})

        for start_comp, dest in zip(component_positions, destinations):
            x = start_comp['end']
            y1 = start_comp['y'] - bar_height_component/2
            y2 = dest['y'] + dest['h']/2
            ax.plot([x, x], [y1, y2],
                    color='#666666', linewidth=1.0, linestyle=':')

        ######################################
        # Figure details and style adjustments
        ######################################

        # Formatting labels
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        # Style the tick labels (Bold and Colored for Aggregates)
        labels = ax.get_yticklabels()
        for i, label_obj in enumerate(labels):
            # Match label to plot_item
            # Note: yticks and plot_items are in the same order (top to bottom)
            if plot_items[i]['type'] in ['subtotal', 'total']:
                label_obj.set_fontweight('bold')
                label_obj.set_color(params['colours'][plot_items[i]['var']])

        # Remove spines
        for location in ['top', 'left', 'right']:
            ax.spines[location].set_visible(False)  # Clean up the look

        # Add vertical grid
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        # Vertical line at x=0
        ax.axvline(0, color='black', linewidth=0.8)
        # Set the grid to the back for the fig
        ax.set_axisbelow(True)

        ax.set_xlabel(
            'Change in global mean surface temperature relative to 1850-1900 (°C)',
            fontsize=12)

        # Title
        fig.text(0.05, 0.95,
                 f'Attributable contributions to warming ({period})',
                 ha='left', fontsize=16, fontweight='bold')

        # Configuration text
        configuration = (f'Scenario: {scen} | '
                         f'Ensemble: {ens} | '
                         f'Regressed variables: {reg_vars} | '
                         f'Regressed range: {reg_range}')
        fig.text(0.05, 0.93, configuration, ha='left', fontsize=8,
                 fontfamily='monospace', color='#555555')

        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))

        # Save plot
        plot_path = ('plots/aggregated/' +
                     f'SCENARIO--{scen}/' +
                     f'ENSEMBLE-MEMBER--{ens}/' +
                     f'VARIABLES--{reg_vars}/' +
                     f'REGRESSED-YEARS--{reg_range}/')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)

        period_token = str(period).replace(' ', '-')
        plot_name = (f'{plot_path}/' +
                     f'Waterfall_BarPlot_Scenario--{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        fig.savefig(plot_name)
        plt.close(fig)


def figure_priors_timeseries(
        scen, ens, reg_vars,
        priors_dfs, obs_dfs,
        params
):
    """Plot timeseries for PRIOR warming."""
    plot_vars_priors = priors_dfs[
        scen][ens]['timeseries'].columns.get_level_values(
            0).unique().to_list()

    # Define major variables (for plumes)
    plume_vars = [v for v in plot_vars_priors
                  if v in defs.SUB_VAR_MAPPING or v == 'Res']

    # Define linestyles for all variables
    var_linestyles = gr.get_dynamic_linestyles(plot_vars_priors)

    # Determine legend location
    sub_vars = [v for v in plot_vars_priors if v not in plume_vars]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0),
                          rowspan=1, colspan=1)

    gr.gwi_timeseries(
        ax, obs_dfs[scen][ens]['timeseries'], None,
        priors_dfs[scen][ens]['timeseries'],
        plot_vars_priors, params['colours'],
        hatch='x', linestyle=var_linestyles,
        plume_vars=plume_vars)

    if sub_vars:
        legend_loc = 'center right'
        legend_cols = 1
        reorder = gr.get_legend_reorder_indices(fig)
    else:
        legend_loc = 'lower center'
        legend_cols = 7
        reorder = None

    ax.set_ylim(
        np.floor(np.min(
            priors_dfs[scen][ens]['timeseries'].values)
            * 2) / 2,
        np.ceil(np.max(
            priors_dfs[scen][ens]['timeseries'].values)
            * 2) / 2
        )
    # ax.set_ylim(-2,5)
    ax.set_xlim(
        max(1850,
            priors_dfs[scen][ens]['timeseries'].index.min()),
        priors_dfs[scen][ens]['timeseries'].index.max())
    gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)

    if legend_loc == 'center right':
        plt.subplots_adjust(right=0.8)
    fig.suptitle(
        f'Prior Warming Timeseries\n'
        f'Scenario: {scen} | '
        f'Ensemble: {ens} | '
        f'Regressed variables: {reg_vars}')
    plot_path = (
        'plots/priors/' +
        f'SCENARIO--{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plot_name = (
        f'{plot_path}/' +
        f'Prior_Timeseries_Scenario--{scen}_' +
        f'ENSEMBLE-MEMBER--{ens}_' +
        f'VARIABLES--{reg_vars}.png')
    fig.savefig(plot_name)
    plt.close(fig)


def figure_erf_timeseries(
        scen, ens, reg_vars,
        erf_dfs, params
):
    """Plot timeseries for ERF."""
    plot_vars_erf = erf_dfs[
        scen][ens]['timeseries'].columns.get_level_values(
            0).unique().to_list()

    # Define major variables (for plumes)
    plume_vars = [v for v in plot_vars_erf
                  if v in defs.SUB_VAR_MAPPING or v == 'Res']

    # Define linestyles for all variables
    var_linestyles = gr.get_dynamic_linestyles(plot_vars_erf)

    # Determine legend location
    sub_vars = [v for v in plot_vars_erf if v not in plume_vars]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0),
                          rowspan=1, colspan=1)

    gr.gwi_timeseries(
        ax, None, None,
        erf_dfs[scen][ens]['timeseries'],
        plot_vars_erf, params['colours'],
        hatch='x', linestyle=var_linestyles,
        plume_vars=plume_vars,
        ylabel='Effective Radiative Forcing (W m⁻²)')

    if sub_vars:
        legend_loc = 'center right'
        legend_cols = 1
        reorder = gr.get_legend_reorder_indices(fig)
    else:
        legend_loc = 'lower center'
        legend_cols = 7
        reorder = None

    ax.set_ylim(
        np.floor(np.min(
            erf_dfs[scen][ens]['timeseries'].values)
            * 2) / 2,
        np.ceil(np.max(
            erf_dfs[scen][ens]['timeseries'].values)
            * 2) / 2
        )
    # ax.set_ylim(-2,5)
    ax.set_xlim(
        max(1750,
            erf_dfs[scen][ens]['timeseries'].index.min()),
        erf_dfs[scen][ens]['timeseries'].index.max())
    gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)

    if legend_loc == 'center right':
        plt.subplots_adjust(right=0.8)
    fig.suptitle(
        f'Effective Radiative Forcing Timeseries\n'
        f'Scenario: {scen} | '
        f'Ensemble: {ens} | '
        f'Regressed variables: {reg_vars}')
    plot_path = (
        'plots/erfs/' +
        f'SCENARIO--{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plot_name = (
        f'{plot_path}/' +
        f'ERF_Timeseries_Scenario--{scen}_' +
        f'ENSEMBLE-MEMBER--{ens}_' +
        f'VARIABLES--{reg_vars}.png')
    fig.savefig(plot_name)
    plt.close(fig)


def parse_argvs():
    """Parse command line arguments."""
    # Get the command line arguments for which iterations to average across.
    # argv format:
    # --ensemble-size=ensemble_size --regressed-years=regressed_years
    # e.g. --ensemble-size=6048000 --regressed-years=1850-2023:
    # where ensemble_size is the number of samples in the ensemble, and
    # regressed_years is the range of years over which the regression acted.

    if len(sys.argv) > 1:
        # Separate out the names and values for each argv, and place them in
        # a dictionary for later use.
        argvs = sys.argv
        argv_dict = {argv.split('=')[0]: argv.split('=')[1]
                     for argv in argvs
                     if '=' in argv}
    else:
        # Adding this simplifies logic later on, as we can always assume that
        # the dictionary exists, and just check for the presence.
        argv_dict = {}

    return argv_dict


def figure_gif_animation(plot_names, scen, ens, reg_vars, reg_ranges_all):
    """Create a gif animation of timeseries plots.

    Accepts a list of plot names (file paths) to include in the gif.
    """
    print('        Creating gif animation of timeseries plots')

    images_list = [Image.open(plot) for plot in plot_names]
    # calculate the frame number of the last frame (ie the number of
    # images)

    # # create 2 extra copies of the last frame (to make the gif spend
    # # longer on the most recent image)
    # for x in range(0, 2):
    #     images_list.append(images_list[-1])

    # Copy and revserse the list of images, so that the gif goes back and
    # forth between the first and last image.
    images_list += images_list[::-1]

    # save as a gif
    images_list[0].save(
        f'plots/aggregated/SCENARIO--{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/' +
        f'Timeseries-animation_Scenario--{scen}_' +
        f'Ensemble-Members--{ens}_' +
        f'Regressed--{reg_vars}_' +
        f'{min(reg_ranges_all)}_to_{max(reg_ranges_all)}.gif',
        save_all=True, append_images=images_list[1:],
        optimize=False, duration=500, loop=0)


def toggle_single_timeseries(
        ens,
        number_divisor=10):
    """Toggle whether to plot single timeseries or not.

    This is particulatly useful for large ensembles (e.g. SMILEs),
    where plotting all ensemble members would take a long time
    and create a large number of files."""

    ens_values = [
        combo.split('-')[1] for combo in ens.split('_')
        ]
    ens_nums = [
        s for s in ens_values if s.isdigit()
        ]
    if set(ens_values) == {'all'}:
        single_toggle = True
    # If divisible by 10, then plot (i.e. just plot 1/10 of the
    # available ensemble members to save space/time)
    elif any(int(s) % number_divisor == 0 for s in ens_nums):
        single_toggle = True
    else:
        single_toggle = False

    return single_toggle


def figure_historical_only_timeseries(
        scen, ens, reg_vars,
        results_dfs, obs_dfs,
        plot_path, params):
    """Plot historical-only timeseries for each headline."""

    print('      Plotting historical-only timeseries')

    for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
        print('        Plotting:', headline)
        plot_vars = results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][
            headline].columns.get_level_values(0).unique().to_list()

        # Define major variables (for plumes)
        plume_vars = [v for v in plot_vars
                      if v in defs.SUB_VAR_MAPPING or v == 'Res']

        # Define linestyles for all variables
        var_linestyles = gr.get_dynamic_linestyles(plot_vars)

        # Determine legend location
        sub_vars = [v for v in plot_vars if v not in plume_vars]

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0),
                              rowspan=1, colspan=1)

        gr.gwi_timeseries(
            ax, obs_dfs[scen][ens]['timeseries'], None,
            results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline],
            plot_vars, params['colours'],
            sigmas=['5', '95', '50'],
            hatch=None, linestyle=var_linestyles,
            plume_vars=plume_vars
            )

        if sub_vars:
            legend_loc = 'center right'
            legend_cols = 1
            reorder = gr.get_legend_reorder_indices(fig)
        else:
            legend_loc = 'lower center'
            legend_cols = 7
            reorder = None
        ax.set_ylim(
            -1,
            np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2)
        ax.set_xlim(params['start_year'], params['end_year'])
        xticks = list(
            np.arange(params['start_year'], params['end_year'] + 1, 5))
        xticks.append(params['end_year'])
        ax.set_xticks(xticks, xticks)
        ax.set_title(
            'Regressed years range: ' +
            f"{params['min_range']} to {params['max_range']}")
        gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)

        if legend_loc == 'center right':
            plt.subplots_adjust(right=0.8)
        fig.suptitle(f'Historical-only {headline}')

        configuration = (f'Scenario: {scen} | '
                         f'Ensemble: {ens} | '
                         f'Regressed variables: {reg_vars}')
        fig.text(0.5, 0.01, configuration, ha='center',
                 fontsize='x-small', fontfamily='monospace')

        fig.savefig(
            plot_path +
            f'Historical_only_{headline}_' +
            f'{scen}_{ens}_{reg_vars}_' +
            f"{params['min_range']}_to_{params['max_range']}.png")
        plt.close(fig)


def figure_historical_vs_full_timeseries(
        scen, ens, reg_vars,
        results_dfs, obs_dfs, priors_dfs,
        plot_path, params):
    """Plot historical-only vs full dataset timeseries."""

    print('      Plotting historical-only vs full dataset')

    plot_vars = results_dfs[scen][ens][reg_vars][
        'HISTORICAL-ONLY'][
            'ANNUAL'].columns.get_level_values(0).unique().to_list()

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    gr.gwi_timeseries(
        ax, obs_dfs[scen][ens]['timeseries'], None,
        results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY']['ANNUAL'],
        plot_vars, params['colours'], sigmas=['5', '95', '50'],
        hatch='\\', linestyle='dashed')
    var_linestyles = gr.get_dynamic_linestyles(plot_vars)
    gr.gwi_timeseries(
        ax, obs_dfs[scen][ens]['timeseries'], None,
        results_dfs[scen][ens][reg_vars][params['max_range']]['timeseries'],
        plot_vars, params['colours'], sigmas=['5', '95', '50'],
        hatch=None, linestyle=var_linestyles)

    ax.set_ylim(
        -1, np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2)
    ax.set_xlim(params['start_year'], params['end_year'])
    xticks = list(np.arange(params['start_year'], params['end_year'] + 1, 5))
    xticks.append(params['end_year'])
    ax.set_xticks(xticks, xticks)

    ax.set_title(
        'Regressed years range: ' +
        f"{params['min_range']} to {params['max_range']}")
    gr.overall_legend(fig, 'lower center', 6)

    fig.suptitle(
        'Historical-only (dashed) versus Full-information (solid)')

    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace')

    fig.savefig(
        plot_path +
        'ANNUAL_Historical_vs_Full_timeseries_' +
        f'{scen}_{ens}_{reg_vars}_' +
        f"{params['min_range']}_to_{params['max_range']}.png")
    plt.close(fig)


def figure_headlines_comparison(
        scen, ens, reg_vars,
        results_dfs, obs_dfs,
        plot_path, params):
    """Plot comparison of all headlines datasets."""

    print('      Plotting historical-only and full-information headlines:',
          scen, ens, reg_vars)

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0),
                           rowspan=3, colspan=1)
    ax2 = plt.subplot2grid(shape=(4, 2), loc=(3, 0),
                           rowspan=1, colspan=1)
    ax3 = plt.subplot2grid(shape=(4, 2), loc=(0, 1),
                           rowspan=3, colspan=1)
    ax4 = plt.subplot2grid(shape=(4, 2), loc=(3, 1),
                           rowspan=1, colspan=1)

    for ax in [ax1, ax3]:
        gr.gwi_timeseries(
            ax, obs_dfs[scen][ens]['timeseries'], None, None, None,
            params['colours'], sigmas=['5', '95', '50'])

    # Plot the centered 20-year rolling window on the 50th percentile Obs
    df_temp_Obs_20yr = obs_dfs[scen][ens][
        'timeseries'
        ].quantile(
            q=0.5, axis=1
        ).rolling(
            window=20, center=True, axis=0
        ).mean()

    # for headline in headlines:
    for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
        plot_vars_main = params['plot_vars'].copy()
        unwanted_vars = ['GHG', 'OHF', 'Res']
        plot_vars_main = list(set(plot_vars_main) - set(unwanted_vars))
        for vv in plot_vars_main:
            # Determine line style
            ls = gr.HEADLINE_LINE_STYLE.get(vv)
            if ls is None:
                # Try to get style from parent
                parent = params['scaling_map'].get(vv)
                ls = gr.HEADLINE_LINE_STYLE.get(parent, 'solid')

            # Plot the historical only timeseries
            ax1.plot(results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].index,
                     results_dfs[scen][ens][reg_vars][
                         'HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                     label=f'{headline}-{vv}',
                     linestyle=ls,
                     color=gr.HEADLINE_COLOURS[headline]
                     )
            if vv != 'Nat':
                ax2.plot(
                    (results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].loc[:, (vv, '50')]
                     - df_temp_Obs_20yr),
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=gr.HEADLINE_COLOURS[headline]
                )

            # Calculate the full-information timeseries for the headlines

            rolling_mean_map = {
                'ANNUAL': {'window': 1, 'center': False},
                'AR6': {'window': 10, 'center': False},
                'SR15': {'window': 30, 'center': True},
                'CGWL': {'window': 20, 'center': True}
            }
            df_fullinfo_defs = (
                results_dfs[scen][ens][reg_vars][params['max_range']][
                    'timeseries'].loc[:, (vv, '50')].rolling(
                        window=rolling_mean_map[headline]['window'],
                        center=rolling_mean_map[headline]['center']
                    ).mean()
            )

            ax3.plot(df_fullinfo_defs.index, df_fullinfo_defs,
                     label=f'{headline}-{vv}',
                     linestyle=ls,
                     color=gr.HEADLINE_COLOURS[headline]
                     )
            if vv != 'Nat':
                ax4.plot(
                    (df_fullinfo_defs - df_temp_Obs_20yr),
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=gr.HEADLINE_COLOURS[headline]
                )

    # Plotting 20-year running means on observations moved to the
    # end to sort ordering in the legend to look nicer.
    for ax in [ax1, ax3]:
        # Plot the Obs 20-year running mean
        ax.plot(df_temp_Obs_20yr.index, df_temp_Obs_20yr,
                label='Obs 20-year running mean',
                color='black'
                )
    for ax in [ax2, ax4]:
        # Plot the Obs 20-year running mean
        ax.plot((df_temp_Obs_20yr - df_temp_Obs_20yr),
                label='Obs 20-year running mean',
                color='black'
                )

    # Slice the df_temp_Obs using the smallest and largest end years
    min_y = np.floor(np.min(obs_dfs[scen][ens]['timeseries'].loc[
            params['start_year']:params['end_year']
        ].values) * 2) / 2
    min_y = min([-0.5, min_y])
    max_y = np.ceil(np.max(
            obs_dfs[scen][ens][
                'timeseries'].loc[params['start_year']:params['end_year']
                                  ].values) * 2) / 2
    # min_y = np.floor(np.min(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2
    # max_y = np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2
    ax1.set_ylim(min_y, max_y)
    ax3.set_ylim(min_y, max_y)
    for ax in [ax1, ax2]:
        ax.set_xlim(params['start_year'], params['end_year'])
    gr.overall_legend(fig, 'lower center', 5)

    ax2.set_ylabel(r'$\Delta$ vs 20-year obs, ⁰C')
    ax4.set_ylabel(r'$\Delta$ vs 20-year obs, ⁰C')
    ax1.set_title('Historical-only')
    ax3.set_title('Full-information')
    fig.suptitle(
        'Historical-only and Full-information vs 20-year Obs running mean')

    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace')

    fig.savefig(plot_path +
                'Historical_and_full_headlines_' +
                f'{scen}_{ens}_{reg_vars}_' +
                f"{params['min_range']}_to_{params['max_range']}.png")

    plt.close(fig)


def figure_thorne_et_al(
        scen, ens, reg_vars,
        results_dfs, obs_dfs,
        plot_path, params):
    """Create one-off figure for Thorne et al paper."""

    # Plot the centered 20-year rolling window on the 50th percentile Obs
    df_temp_Obs_20yr = obs_dfs[scen][ens][
        'timeseries'
        ].quantile(
            q=0.5, axis=1
        ).rolling(
            window=20, center=True, axis=0
        ).mean()

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))

    gr.gwi_timeseries(
        ax, obs_dfs[scen][ens]['timeseries'], None, None, None,
        params['colours'],
        sigmas=['5', '95', '50']
        # hatch='\\', linestyle='dashed'
    )
    # for headline in headlines:
    plot_vars_main = params['plot_vars'].copy()
    unwanted_vars = ['GHG', 'OHF', 'Res']
    plot_vars_main = list(set(plot_vars_main) - set(unwanted_vars))

    for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
        for vv in plot_vars_main:
            # Determine line style
            ls = gr.HEADLINE_LINE_STYLE.get(vv)
            if ls is None:
                # Try to get style from parent
                parent = params['scaling_map'].get(vv)
                ls = gr.HEADLINE_LINE_STYLE.get(parent, 'solid')

            # Plot the historical only timeseries
            ax.plot(results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].index,
                    results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=gr.HEADLINE_COLOURS[headline]
                    )
    ax.plot(df_temp_Obs_20yr.index, df_temp_Obs_20yr,
            label='Obs 20-year running mean',
            color='black'
            )

    # Slice the df_temp_Obs using the smallest and largest end years
    min_y = np.floor(np.min(obs_dfs[scen][ens]['timeseries'].loc[
        params['start_year']:params['end_year']
            ].values) * 2) / 2
    min_y = min([-0.5, min_y])
    max_y = np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].loc[
        params['start_year']:params['end_year']
            ].values) * 2) / 2

    ax.set_ylim(min_y, max_y)
    ax.set_xlim(params['start_year'], params['end_year'] + 0.5)
    ax.set_title(
        'Calculated as historical-only: annual-mean, AR6 decade-mean, '
        'SR1.5 centered 30-year mean, and CGWL centered 20-year mean')
    gr.overall_legend(fig, 'lower center', 5)

    ax.set_ylabel('Attributable warming relative to 1850–1900 (⁰C)')
    fig.suptitle(
        'Global Warming Index (GWI)'
    )

    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace')

    fig.savefig(plot_path +
                'Historical_only_headlines_' +
                f'{scen}_{ens}_{reg_vars}_' +
                f"{params['min_range']}_to_{params['max_range']}.png")
    plt.close(fig)


def figure_constrained_warming(
        scen, ens, reg_vars,
        results_dfs, priors_dfs,
        plot_path, params):
    """Generate the projected warming for final constrained year."""

    start_years = set([int(reg_range.split('-')[0])
                       for reg_range in params['reg_ranges_all']])
    if len(start_years) == 1:
        start_regress = list(start_years)[0]
    else:
        start_regress = 'VAR'

    plot_vars_priors = priors_dfs[scen][ens][
        'timeseries'].columns.get_level_values(0).unique().to_list()

    print('      Creating constrained results')
    # Calculate how the expected final year of the timeseries changes
    # depending on the years that are regressed. Expect that the attributed
    # values in 2023 (end year of the full timeseries) will have larger
    # uncertainties, the earlier/shorter the range of regressed years is.

    # Create new empty dataframes to store the constrained results:
    # NOTE: you could also do this using maximum of the truncation range
    # if that's what you're interested in (possibly more relevant for
    # SSP projections in future)

    # constrained_year = int(max_regressed_range.split('-')[1])
    constrained_year = params['end_year']

    df_constrained = results_dfs[scen][ens][reg_vars][
        params['reg_ranges_all'][0]]['timeseries'].copy()
    df_constrained[:] = 0

    # For each iteration, add the final row of the dataframe to the new
    # df_hist. The row index it should be inserted at is the same as the
    # second year in the iteration name.
    for reg_range in params['reg_ranges_all']:
        # print(iteration, iteration.split('-')[1], constrained_year)
        df_constrained.loc[int(reg_range.split('-')[1])] = \
            results_dfs[scen][ens][reg_vars][reg_range][
                'timeseries'].loc[constrained_year]

    # Remove all years that are not the end of an attribution period to
    # avoid confusion:
    df_constrained = df_constrained.loc[params['start_year']:, :]

    #######################################################################
    # Plot this dataframe df_constrined in the same way as df_hist

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(
        shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
    ax2 = plt.subplot2grid(
        shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)

    var_linestyles = gr.get_dynamic_linestyles(plot_vars_priors)
    gr.gwi_timeseries(
        ax1, None, None, df_constrained,
        plot_vars_priors, params['colours'], sigmas=['5', '95', '50'],
        linestyle=var_linestyles)

    # ax1.set_ylim(
    #     np.floor(np.min(df_constrained.values) * 2) / 2,
    #     np.ceil(np.max(df_constrained.values) * 2) / 2)
    ax1.set_xlim(params['start_year'], params['end_year'])
    ax1.set_ylabel(f'Warming in {constrained_year} ⁰C')
    ax1.set_xlabel(f'Regressed years: {start_regress}-<year>')
    xticks = list(np.arange(params['start_year'], params['end_year'] + 1, 5))
    xticks.append(params['end_year'])
    ax1.set_xticks(xticks, xticks)
    ax1.set_title(
        f'Constrained: {constrained_year} (with Obs only up to year <year>)')

    # Create box and whisker plot for prior warming in each variable
    bar_width = 0.4

    for vv in plot_vars_priors:
        # Plot the multi-method assessed results for the 2010-2019 period
        med_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year,
                                                            (vv, '50')]
        min_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year,
                                                            (vv, '5')]
        max_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year,
                                                            (vv, '95')]

        ax2.fill_between(
            [params['plot_vars'].index(vv),
             params['plot_vars'].index(vv) + bar_width],
            min_prior, max_prior,
            color=params['colours'][vv],
            alpha=0.6,
            linewidth=0,
            label=vv
        )

        ax2.plot(
            [params['plot_vars'].index(vv),
             params['plot_vars'].index(vv) + bar_width],
            [med_prior, med_prior],
            color=params['colours'][vv],
            lw=2)

    # Remove the xticks in ax2
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    # Get the ylims from ax1
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_title(f'Unconstrained: {constrained_year}')

    gr.overall_legend(fig, 'lower center', 6)

    fig.suptitle(f'Constrained projected warming in {constrained_year}')

    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace')

    fig.savefig(
        plot_path +
        f'Projected_warming_in_{constrained_year}_' +
        f'regressing_{reg_vars}_'
        'constrained_by_regressed_years_' +
        f"{params['min_range']}_to_{params['max_range']}.png")
    plt.close(fig)

    return df_constrained


def figure_delta_contributions(
        scen, ens, reg_vars,
        results_dfs, obs_dfs, df_constrained,
        plot_path, params):
    """Generate timeseries showing source of changes in GWI value each year."""

    start_years = set([int(reg_range.split('-')[0])
                       for reg_range in params['reg_ranges_all']])
    if len(start_years) == 1:
        start_regress = list(start_years)[0]
    else:
        start_regress = 'VAR'

    print('      Creating delta contributions')

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=1, colspan=1)

    # Create a new empty dataframe copied from before:
    df_delta_additional_forcing_year = df_constrained.copy()
    df_delta_revised_previous_year = df_constrained.copy()
    df_delta_additional_forcing_year[:] = 0
    df_delta_revised_previous_year[:] = 0

    differ_years = sorted([r.split('-')[1] for r in params['reg_ranges_all']])
    # switch the sorted order of the list years
    differ_years = differ_years[::-1]
    # remove the smallest year
    differ_years = differ_years[:-1]

    for y in differ_years:
        # delta_new is the change from year Y to Y+1 in the new dataset.
        delta_new = (
            results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}'][
                'timeseries'].loc[int(y)] -
            results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}'][
                'timeseries'].loc[int(y)-1])
        # delta_rev is the change to the year Y from the previous to the
        # new dataset.
        delta_rev = (
            results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}'][
                'timeseries'].loc[int(y)-1] -
            results_dfs[scen][ens][reg_vars][f'{start_regress}-{int(y)-1}'][
                'timeseries'].loc[int(y)-1])
        df_delta_additional_forcing_year.loc[int(y)] = delta_new
        df_delta_revised_previous_year.loc[int(y)] = delta_rev

    #######################################################################
    # Plot the results

    print('        Plotting delta contributions')

    line_alpha = 0.9

    changing_var = 'Ant' if 'Ant' in params['plot_vars'] else 'Tot'

    df_hist = results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY']['ANNUAL']

    ax1.fill_between(
        # df_delta_additional_forcing_year.index,
        df_hist.loc[params['start_year']:, ('Res', '5')].index,
        df_hist.loc[params['start_year']:, ('Res', '5')].values,
        df_hist.loc[params['start_year']:, ('Res', '95')].values,
        color='seagreen', alpha=0.1, lw=0)
    ax1.fill_between(
        df_delta_revised_previous_year.index,
        # df_delta_revised_previous_year.loc[:, (changing_var, '5')].index,
        df_delta_revised_previous_year.loc[:, (changing_var, '5')].values,
        df_delta_revised_previous_year.loc[:, (changing_var, '95')].values,
        color='steelblue', alpha=0.3, lw=0)
    ax1.fill_between(
        df_delta_additional_forcing_year.index,
        # df_delta_additional_forcing_year.loc[:, (changing_var, '5')].index,
        df_delta_additional_forcing_year.loc[:, (changing_var, '5')].values,
        df_delta_additional_forcing_year.loc[:, (changing_var, '95')].values,
        color='indianred', alpha=0.3, lw=0)
    ax1.plot(
        # df_delta_additional_forcing_year.index,
        df_hist.loc[params['start_year']:, ('Res', '50')].index,
        df_hist.loc[params['start_year']:, ('Res', '50')].values,
        label='Residual (internal variability) in year Y+1',
        color='seagreen', ls='dashed', alpha=line_alpha)
    ax1.plot(
        df_delta_revised_previous_year.index,
        # df_delta_revised_previous_year.loc[:, (changing_var, '50')],
        df_delta_revised_previous_year.loc[:, (changing_var, '50')].values,
        label='Revised warming in year Y',
        color='steelblue', alpha=line_alpha)
    ax1.plot(
        df_delta_additional_forcing_year.index,
        # df_delta_additional_forcing_year.loc[:, (changing_var, '50')].index,
        df_delta_additional_forcing_year.loc[:, (changing_var, '50')].values,
        label='Additional warming from year Y to Y+1 in new dataset',
        color='indianred', alpha=line_alpha)

    # Add a zero line for reference
    ax1.axhline(0, color='black', linestyle='solid')
    xticks = list(np.arange(params['start_year'], params['end_year'] + 1, 5))
    xticks.append(params['end_year'])
    ax1.set_xticks(xticks, xticks)
    ax1.set_xlim(int(min(differ_years)), int(max(differ_years))+0.5)
    ax1.set_ylim(-0.3, +0.3)
    ax1.set_ylabel('Interannual warming delta, ⁰C')

    #######################################################################
    # Plot schematic diagram
    years = list(range(params['end_year'], params['end_year']-4, -1))
    # which [2023, 2022, 2021, 2020] when the end year is 2023.
    for year in years:
        df_new = results_dfs[scen][ens][reg_vars][f'{start_regress}-{year}'][
            'timeseries']
        ax2.plot(df_new.loc[:year, :].index,
                 df_new.loc[:year, (changing_var, '50')],
                 label=f'{start_regress}-{year}',
                 color='darkslategray', linestyle='solid',
                 marker='o', markeredgewidth=0,
                 alpha=(year-min(years)+1)/len(years),
                 lw=2
                 #  lw=(year-min(years))/len(years) * 2 + 0.5
                 )
        if year == max(years):
            # Plot the red lines for the new year's extra year of forcing
            ax2.plot([year-1, year],
                     [df_new.loc[year-1, (changing_var, '50')],
                      df_new.loc[year-1, (changing_var, '50')]],
                     color='indianred', linestyle='dashed', lw=2,
                     alpha=(year-min(years)+1)/len(years))
            ax2.plot([year, year],
                     [df_new.loc[year-1, (changing_var, '50')],
                      df_new.loc[year, (changing_var, '50')]],
                     color='indianred', linestyle='solid', lw=2,
                     alpha=(year-min(years)+1)/len(years))
            # Plot the blue line for the previous year's revision
            df_old = results_dfs[scen][ens][reg_vars][
                f'{start_regress}-{year-1}']['timeseries']

            ax2.plot([year-1, year],
                     [df_old.loc[year-1, (changing_var, '50')],
                      df_old.loc[year-1, (changing_var, '50')]],
                     color='steelblue', linestyle='dashed', lw=2,
                     alpha=(year-min(years)+1)/len(years))
            ax2.plot([year, year],
                     [df_new.loc[year-1, (changing_var, '50')],
                      df_old.loc[year-1, (changing_var, '50')]],
                     color='steelblue', linestyle='solid', lw=2,
                     alpha=(year-min(years)+1)/len(years))

    ax2.plot(df_hist.index,
             df_hist.loc[:, (changing_var, '50')].values,
             label='Historical-only GWI',
             color='slateblue', ls='dashed', alpha=line_alpha, lw=1.5,
             marker='o', markeredgewidth=0)

    #######################################################################
    # Plot observations scatter with error:
    # Plot the observations
    err_pos = (obs_dfs[scen][ens]['timeseries'].quantile(q=0.95, axis=1) -
               obs_dfs[scen][ens]['timeseries'].quantile(q=0.5, axis=1))
    err_neg = (obs_dfs[scen][ens]['timeseries'].quantile(q=0.5, axis=1) -
               obs_dfs[scen][ens]['timeseries'].quantile(q=0.05, axis=1))
    ax2.errorbar(
        obs_dfs[scen][ens]['timeseries'].index, obs_dfs[scen][ens][
            'timeseries'].quantile(q=0.5, axis=1),
        yerr=(err_neg, err_pos),
        fmt='o', color=params['colours']['Obs'], ms=2.5, lw=1,
        label='Reference Temp: HadCRUT5')

    ax2.set_ylabel('Global Warming, ⁰C')
    ax2.set_xticks(years, years)
    # ax2.set_xlim(2019.5, 2023.5)
    ax2.set_xlim(params['end_year']-4+0.5, params['end_year']+0.5)

    fig.suptitle(
        f'Contributions to the change in {changing_var} warming ' +
        'each year Y → Y+1')
    configuration = (f'Scenario: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace')
    gr.overall_legend(fig, 'lower center', 3)
    fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])

    fig.savefig(
        plot_path +
        'Historical_delta_contributions_' +
        f"{reg_vars}_{params['min_range']}_to_{params['max_range']}.png")
    plt.close(fig)

    # Compare variation between internal variation (using Residual as a
    # proxy for this, because ideally speaking, all forced warming is
    # accounted for, so the remaining should largely be internal
    # variability). Use RMS:
    delta_rms = np.sqrt(
        np.mean(df_delta_revised_previous_year.loc[:, (changing_var, '50')
                                                   ].values**2))
    residual_rms = np.sqrt(
        np.mean(df_hist.loc[params['start_year']:, ('Res', '50')].values**2))

    print(f'          Revision RMS for {reg_vars}: {delta_rms}')
    print(f'          Residual RMS for {reg_vars}: {residual_rms}')
    print(f'          Average fractional variation for {reg_vars}:',
          delta_rms / residual_rms)


def overarching_base_result_plotter(
    results_dfs,
    obs_dfs,
    erf_dfs,
    priors_dfs
):
    """Plot figures of base results."""

    print('\nPlotting single-run timeseries')
    for scen in results_dfs.keys():
        print('SCENARIO:', scen)
        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)
            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED_VARIABLES:', reg_vars)

                params = setup_plot_params(
                    scen, ens, reg_vars, results_dfs)
                reg_ranges_all = params['reg_ranges_all']

                # Check if all years are available
                if defs.check_steps(reg_ranges_all)['check_bool']:
                    print('      All years available for: ',
                          defs.check_steps(reg_ranges_all)['range'])

                ###############################################################
                # 1. Plot GWI Timeseries

                single_toggle = toggle_single_timeseries(ens, 10)

                if single_toggle:
                    valid_ranges_ts = [
                        r for r in reg_ranges_all
                        if is_dataset_present(
                            results_dfs[scen][ens][reg_vars][r], 'timeseries')
                        ]
                    if valid_ranges_ts:
                        with mp.Pool(os.cpu_count()) as p:
                            print('        Plotting figure_timeseries for GWI')
                            # print('  in parallel for:', valid_ranges_ts)
                            plot_names = p.map(
                                functools.partial(
                                    figure_timeseries,
                                    scen=scen, ens=ens, reg_vars=reg_vars,
                                    results_dfs=results_dfs,
                                    df_temp_Obs=obs_dfs[scen][ens]['timeseries'],
                                    params=params
                                    ),
                                valid_ranges_ts
                            )

                    valid_ranges_rates = [r for r in reg_ranges_all if is_dataset_present(results_dfs[scen][ens][reg_vars][r], 'rates')]
                    if valid_ranges_rates:
                        with mp.Pool(os.cpu_count()) as p:
                            print('        Plotting figure_rates for GWI')
                            p.map(
                                functools.partial(
                                    figure_rates,
                                    scen=scen, ens=ens, reg_vars=reg_vars,
                                    results_dfs=results_dfs,
                                    df_temp_Obs=obs_dfs[scen][ens]['timeseries'],
                                    params=params
                                    ),
                                valid_ranges_rates
                            )

                    ###########################################################
                    # 2. Create GIF of Timeseries Plots

                    # Add a toggle, because this is quite slow for the SMILE
                    # ensembles (e.g. where we have an entirely different
                    # set of results for a different ensemble member).
                    gif_toggle = True
                    if gif_toggle and valid_ranges_ts:
                        figure_gif_animation(
                            plot_names, scen, ens, reg_vars, valid_ranges_ts)

                ###############################################################
                # 3. Plot Priors Timeseries
                print('        Plotting figure_timeseries for PRIORS')
                figure_priors_timeseries(
                    scen, ens, reg_vars, priors_dfs, obs_dfs,  params)

                ###############################################################
                # 3b. Plot ERF Timeseries
                print('        Plotting figure_timeseries for ERF')
                figure_erf_timeseries(
                    scen, ens, reg_vars, erf_dfs, params)

                ###############################################################
                # 4. Plot SPM2 Bar Plot
                valid_ranges_headlines = [
                    r for r in reg_ranges_all
                    if is_dataset_present(
                        results_dfs[scen][ens][reg_vars][r], 'headlines')
                    ]
                if valid_ranges_headlines:
                    print('        Plotting SPM2 for GWI in parallel')
                    with mp.Pool(os.cpu_count()) as p:
                        p.map(
                            functools.partial(
                                figure_spm2,
                                scen=scen, ens=ens, reg_vars=reg_vars,
                                results_dfs=results_dfs,
                                obs_dfs=obs_dfs,
                                params=params
                            ),
                            valid_ranges_headlines
                        )

                ###############################################################
                # 5. Plot Waterfall Plot
                if valid_ranges_headlines:
                    print('        Plotting Waterfall for GWI in parallel')
                    with mp.Pool(os.cpu_count()) as p:
                        p.map(
                            functools.partial(
                                figure_waterfall,
                                scen=scen, ens=ens, reg_vars=reg_vars,
                                results_dfs=results_dfs,
                                obs_dfs=obs_dfs,
                                params=params
                            ),
                            valid_ranges_headlines
                        )


def overarching_historical_only_plotter(
    results_dfs,
    obs_dfs,
    priors_dfs
):
    """Plot figures of historical-only results."""

    print('\nGenerating historical-only timeseries')
    for scen in sorted(results_dfs.keys()):
        print('SCENARIO:', scen)

        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)

            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED-VARIABLES:', reg_vars)

                if not is_dataset_present(results_dfs[scen][ens][reg_vars], 'HISTORICAL-ONLY'):
                    print('      No historical-only datasets available; skipping historical-only plotting.')
                    continue

                plot_path = f'{PLOT_FOLDER}aggregated/' + \
                    f'SCENARIO--{scen}/' + \
                    f'ENSEMBLE-MEMBER--{ens}/' + \
                    f'VARIABLES--{reg_vars}/'
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)

                params = setup_plot_params(
                    scen, ens, reg_vars, results_dfs)

                # 1. Plot historical-only timeseries for each headline
                figure_historical_only_timeseries(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 2. Plot historical-only vs full dataset timeseries
                figure_historical_vs_full_timeseries(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs, priors_dfs,
                    plot_path, params
                )

                # 3. Plot comparison of all headlines datasets
                figure_headlines_comparison(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 4. Create one-off figure for Thorne et al paper
                figure_thorne_et_al(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 5. Generate the projected warming for final constrained year
                df_constrained = figure_constrained_warming(
                    scen, ens, reg_vars,
                    results_dfs, priors_dfs,
                    plot_path, params
                )

                # 6. Generate timeseries showing source of changes in GWI value
                # each year
                figure_delta_contributions(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs, df_constrained,
                    plot_path, params,
                )


if __name__ == '__main__':

    # NOTE:
    # results_files[reg_scen][reg_vars][reg_range][result_type].keys():
    # results_files[reg_scen][reg_vars][reg_range][result_type].keys():
    # Where result_type is timeseries, headlines
    # And reg_range is the range of years that the regression was performed
    # over, or 'historical-only', which is the range of years that the
    # historical-only dataset was calculated over.

    argv_dict = parse_argvs()
    print(argv_dict)

    # Configuration
    if '--re-calculate' in argv_dict:
        re_calculate = argv_dict['--re-calculate'] == 'y'  # True/False y/n
    else:
        re_calculate = True  # Default to re-calculate if not specified

    print(f"Re-calculate results: {re_calculate}")

    # Ensure directoriesfor plots and results exist
    for folder in [PLOT_FOLDER, AGGREGATED_FOLDER, ITERATIONS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # 1.Calculate gwi iteration averages
    if re_calculate:
        calculate_iteration_averages()

    # 2. Load results (gwi, priors, obs) into dataframes
    results_dfs, priors_dfs, erf_dfs, obs_dfs = load_gwi_priors_erf_obs()

    # 3. Plot the basic results
    overarching_base_result_plotter(
        results_dfs, obs_dfs, erf_dfs, priors_dfs)

    # 4. Generate historical-only timeseries and save to CSVs.
    if re_calculate:
        calculate_historical_only(results_dfs)

    # 5. Load the historical-only data into dataframes
    results_dfs = load_historical_only_dfs(results_dfs)

    # 6. Plot the historical-only results
    overarching_historical_only_plotter(
        results_dfs, obs_dfs, priors_dfs)
