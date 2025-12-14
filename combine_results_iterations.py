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
import pprint


# Constants

VAR_COLOURS = {
    'Tot': '#d7827e',
    'Ant': '#b4637a',
    'GHG': '#907aa9',
    'Nat': '#56949f',
    'OHF': '#ea9d34',
    'Res': '#9893a5',
    'Obs': '#797593',
    'PiC': '#cecacd'
}


VAR_NAMES = {
    'Obs': 'Observed warming',
    'Tot': 'Total forced warming',
    'Ant': 'Human-induced warming',
    'GHG': 'Well-mixed greenhouse gases',
    'OHF': 'Other human forcings',
    'Nat': 'Solar and volcanic drivers',
    'Res': 'Residual (Internal variability)',
    'co2': 'Carbon dioxide',
    'ch4': 'Methane',
    'n2o': 'Nitrous oxide',
    'halogen': 'Halogenated gases',
    'aerosol-radiation_interactions': 'Aerosol-radiation interactions',
    'aerosol-cloud_interactions': 'Aerosol-cloud interactions',
    'land_use': 'Land-use reflectance',
    'bc_snow': 'Black carbon on snow',
    'h2o_strat': 'Stratospheric water vapour',
    'o3': 'Ozone',
    'solar': 'Solar',
    'volcanic': 'Volcanic',
    'contrails': 'Aviation contrails'
}

HEADLINE_COLOURS = {
    'ANNUAL': '#5BA2D0',
    'SR15': '#9CCFD8',
    'AR6': '#EE8679',
    'CGWL': '#A88BFA'
}
HEADLINE_LINE_STYLE = {
    'Tot': 'solid',
    'Ant': 'dashed',
    'Nat': 'dotted'
}

PLOT_FOLDER = 'plots/'
AGGREGATED_FOLDER = 'results/aggregated'
ITERATIONS_FOLDER = 'results/iterations'


def calculate_iterations(
        re_calculate, headline_toggle,
        iterations_folder, aggregated_folder
):
    """Average the timeseries and headlines iterations."""

    # Skip if not re-calculating
    if not re_calculate:
        return

    scenarios_all = sorted(
        [d.split('SCENARIO--')[1] for d in os.listdir(iterations_folder)])
    print(scenarios_all)

    for scenario in scenarios_all:
        print('Calculating SCENARIO:', scenario)

        ensemble_selections = sorted(
            [d.split('ENSEMBLE-MEMBER--')[1]
             for d
             in os.listdir(f'{iterations_folder}/SCENARIO--{scenario}/')
             ])
        for ensemble_selection in ensemble_selections:
            print('  Calculating ensemble selection:', ensemble_selection)

            regressed_variables_all = sorted(
                [d.split('VARIABLES--')[1]
                 for d in os.listdir(
                     f'{iterations_folder}/' +
                     f'SCENARIO--{scenario}/' +
                     f'ENSEMBLE-MEMBER--{ensemble_selection}/')
                 ])

            print('    All regressed variables for scenario:',
                  regressed_variables_all)

            for regressed_vars in regressed_variables_all:
                print('      Calculating regressed variables:', regressed_vars)
                _path = (f'{iterations_folder}/' +
                         f'SCENARIO--{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')
                regressed_years_vars = sorted(
                    [d.split('REGRESSED-YEARS--')[1]
                     for d in os.listdir(_path)
                     if os.path.isdir(f'{_path}{d}')
                     ])

                if defs.check_steps(regressed_years_vars)['check_bool']:
                    print(f'        All regressed years for {regressed_vars}:',
                          defs.check_steps(regressed_years_vars)['range'])

                with mp.Pool(os.cpu_count()) as p:
                    print('        Calculating (parallel regressed_years) ',
                          'for:',
                          scenario, ensemble_selection, regressed_vars)
                    p.map(
                        functools.partial(
                            combine_repeats,
                            result_type='timeseries', scenario=scenario,
                            ensemble_selection=ensemble_selection,
                            regressed_vars=regressed_vars,
                            iterations_folder=iterations_folder,
                            aggregated_folder=aggregated_folder),
                        regressed_years_vars)
                    if headline_toggle:
                        p.map(
                            functools.partial(
                                combine_repeats,
                                result_type='headlines', scenario=scenario,
                                ensemble_selection=ensemble_selection,
                                regressed_vars=regressed_vars,
                                iterations_folder=iterations_folder,
                                aggregated_folder=aggregated_folder),
                            regressed_years_vars)


def combine_repeats(regressed_years, result_type, scenario, ensemble_selection,
                    regressed_vars, iterations_folder, aggregated_folder):
    """
    Average results across iterations for a specific configuration.

    Args:
        regressed_years: The range of years used for regression.
        result_type: The type of result (e.g., 'timeseries', 'headlines').
        scenario: The scenario name.
        ensemble_selection: The ensemble selection name.
        regressed_vars: The regressed variables.
        iterations_folder: Path to the iterations folder.
        aggregated_folder: Path to the aggregated folder.

    Returns:
        A tuple containing the averaged DataFrame, a dictionary of all
        iterations, and a dictionary of ensemble sizes, or (None, None, None)
        if no files found.
    """
    dict_iterations = {}
    size_iterations = {}

    base_path = (
        f'{iterations_folder}/'
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
    out_path = base_path.replace(iterations_folder, aggregated_folder)
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


def load_all_data(aggregated_folder):
    """Load all averaged datasets."""
    results_files = {}
    priors_files = {}
    obs_files = {}

    scenarios_all = sorted(
            [d.split('SCENARIO--')[1] for d in os.listdir(aggregated_folder)])
    print(scenarios_all)

    for scenario in scenarios_all:
        results_files.update({scenario: {}})
        priors_files.update({scenario: {}})
        obs_files.update({scenario: {}})

        _path = f'{aggregated_folder}/SCENARIO--{scenario}/'

        ensembles_seletions_all = sorted(
            [d.split('ENSEMBLE-MEMBER--')[1] for d in os.listdir(_path)])

        for ensemble_selection in ensembles_seletions_all:
            results_files[scenario].update({ensemble_selection: {}})
            priors_files[scenario].update({ensemble_selection: {}})
            obs_files[scenario].update({ensemble_selection: {}})

            # Load priors files
            _path_prior_dir = ('results/priors/' +
                               f'SCENARIO--{scenario}/' +
                               f'ENSEMBLE-MEMBER--{ensemble_selection}/')

            if os.path.exists(_path_prior_dir):
                for f in os.listdir(_path_prior_dir):
                    if f.startswith('PRIOR_results_timeseries_'):
                        priors_files[scenario][ensemble_selection][
                            'timeseries'] = os.path.join(_path_prior_dir, f)

            _path = (f'{aggregated_folder}/' +
                     f'SCENARIO--{scenario}/' +
                     f'ENSEMBLE-MEMBER--{ensemble_selection}/')

            regressed_variables_all = sorted(
                    [d.split('VARIABLES--')[1] for d in os.listdir(_path)])

            for regressed_vars in regressed_variables_all:
                results_files[scenario][ensemble_selection].update(
                    {regressed_vars: {}})

                _path = (f'{aggregated_folder}/' +
                         f'SCENARIO--{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')

                regressed_years_vars = sorted(
                        [d.split('REGRESSED-YEARS--')[1] for d in
                         os.listdir(_path) if os.path.isdir(f'{_path}{d}')])

                for regressed_years in regressed_years_vars:
                    # Load GWI results files
                    res_type_dict = {
                        res_type: (
                                f'{aggregated_folder}/' +
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
                        for res_type in ['timeseries', 'headlines']
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
    results_dfs = load_nested_dfs(results_files)
    priors_dfs = load_nested_dfs(priors_files)
    obs_dfs = load_nested_dfs(obs_files)

    # Load temperature observations
    print('Loading temperature observations...')
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

    return results_dfs, priors_dfs, obs_dfs


def historical_only(scen, ens, reg_vars, reg_ranges_all,
                    headline, headline_toggle, results_dfs):
    """Calculate historical-only timeseries for each headline."""
    # Prepare empty timeseries for each headline
    df_hist_headline = results_dfs[
        scen][ens][reg_vars][reg_ranges_all[0]]['timeseries'].copy()
    df_hist_headline[:] = 0

    for reg_range in reg_ranges_all:
        # Extract the relevant headline values for this regressed range
        current_year = int(reg_range.split('-')[1])
        if headline == 'ANNUAL':
            headline_time = (
                # headlines index string-y; timeseries index integer-y
                str(current_year) if headline_toggle else current_year)
        elif headline == 'AR6':
            headline_time = f'{current_year-9}-{current_year}'
        elif headline == 'SR15':
            headline_time = f'{current_year} (SR15 definition)'
        elif headline == 'CGWL':
            headline_time = (
                f'{current_year-9}-{current_year+10} (CGWL definition)'
                )

        # Determine whether to pull the headline from the headlines or
        # timeseries dataframe. You can only pull annual years from the
        # both headlines and timeseries dataframes. The value of
        # selecting, is that the headlines are much more
        # computationally expensive to calculate, do you may not always
        # calculate the headlines for all regressed_year ranges.
        res_type = 'headlines' if headline_toggle else 'timeseries'

        if headline_time in results_dfs[scen
                                        ][ens
                                          ][reg_vars
                                            ][reg_range
                                              ][res_type
                                                ].index:

            _df = results_dfs[scen
                              ][ens
                                ][reg_vars
                                  ][reg_range
                                    ][res_type
                                      ].loc[headline_time]

            df_hist_headline.loc[current_year] = _df

        else:
            pass
            # print(f'FAILURE: Headline time {headline} {headline_time} not ' +
            #       f'found in {scen} {reg_vars} {reg_range} {res_type}')

    # Remove all years that are not the end of an attribution
    # period to avoid confusion (i.e. the longer earlier years
    # before the historical-only focus period).
    end_years = [int(reg_range.split('-')[1])
                 for reg_range in reg_ranges_all]
    smallest_end_year = min(end_years)
    largest_end_year = max(end_years)
    start_years = set([int(reg_range.split('-')[0])
                       for reg_range in reg_ranges_all])
    if len(start_years) == 1:
        start_regress = list(start_years)[0]
    else:
        raise ValueError('Multiple start years in regressed ranges')

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

    return df_hist_headline, None


def figure_timeseries(reg_range, scen, ens, reg_vars,
                      results_dfs, df_temp_Obs,
                      VAR_COLOURS
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
        all_data_vars, VAR_COLOURS, hatch='x', linestyle='dashed',
        plume_vars=plume_vars)

    gr.gwi_timeseries(
        ax, df_temp_Obs, None,
        df_ts.loc[reg_start:reg_end, :],
        all_data_vars, VAR_COLOURS, linestyle=var_linestyles,
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
        gr.overall_legend(fig, 'lower center', 7, legend_cols,
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
        os.makedirs(plot_path)

    plot_name = (f'{plot_path}/' +
                 f'Timeseries_Scenario--{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    # plot_names.append(plot_name)
    fig.savefig(plot_name)
    plt.close(fig)
    return plot_name


def figure_spm2(
        reg_range, scen, ens, reg_vars,
        results_dfs, obs_dfs,
        var_colours, var_names):
    """Plot single SPM2 bar plot."""

    # Check if headlines exist for this range
    if 'headlines' not in results_dfs[scen][ens][reg_vars][reg_range]:
        print(f'            No headlines found for {reg_range}, skipping.')
        return

    df_headlines = results_dfs[scen][ens][reg_vars][reg_range]['headlines']

    # Get observations headlines
    if 'headlines' in obs_dfs[scen][ens][reg_range]:
        df_obs_headlines = obs_dfs[scen][ens][reg_range]['headlines']
    else:
        df_obs_headlines = None

    # Determine period (last year)
    years = [idx for idx in df_headlines.index if str(idx).isdigit()]
    if years:
        period = years[-1]
    else:
        period = df_headlines.index[-1]

    # Determine variables for SPM2 panels 2 and 3.
    possible_vars_p2 = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat', 'Res']
    vars_panel2 = [v for v in possible_vars_p2
                   if (v, '50') in df_headlines.columns]

    # Panel 3: Components
    vars_panel3 = []
    if defs.SUB_VAR_MAPPING:
        for group in ['GHG', 'OHF', 'Nat']:
            if group in defs.SUB_VAR_MAPPING:
                for sub_var in defs.SUB_VAR_MAPPING[group]:
                    if (sub_var, '50') in df_headlines.columns:
                        vars_panel3.append(sub_var)

    # Calculate grid dimensions based on the number of variables in each panel
    # in order to make the bars in each panel the same visual width.
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
        (1, total_width), (0, x_width_1 + spacer), colspan=x_width_2, fig=fig)
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
                           var_colours, var_names,
                           ylim, show_ylabel=True, show_yticklabels=True,
                           xlim=(-1.5, 1.5))

    # Panel 2: Aggregated
    gr.plot_spm2_panel(axes[1], df_headlines, period, vars_panel2,
                       var_colours, var_names,
                       ylim, show_ylabel=False, show_yticklabels=False)

    # Panel 3: Components
    if vars_panel3 and len(axes) > 2:
        gr.plot_spm2_panel(axes[2], df_headlines, period, vars_panel3,
                           var_colours, var_names,
                           ylim, show_ylabel=False, show_yticklabels=False)

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

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.85))

    # Save plot
    plot_path = ('plots/aggregated/' +
                 f'SCENARIO--{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plot_name = (f'{plot_path}/' +
                 f'SPM2_BarPlot_Scenario--{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    fig.savefig(plot_name)
    plt.close(fig)


def figure_waterfall(
        reg_range, scen, ens, reg_vars,
        results_dfs, obs_dfs,
        VAR_COLOURS, VAR_NAMES):
    """Plot single waterfall plot (Horizontal Design with Subtotals)."""

    # Get headlines
    df_headlines = results_dfs[scen][ens][reg_vars][reg_range]['headlines']
    df_obs_headlines = obs_dfs[scen][ens][reg_range]['headlines']

    # Determine period (last year)
    years = [idx for idx in df_headlines.index if str(idx).isdigit()]
    if years:
        period = years[-1]
    else:
        period = df_headlines.index[-1]

    # Helper to get stats
    def get_stats(v, df=df_headlines):
        if (v, '50') in df.columns:
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

    # Manually specify yticks and labels to enable arrows to be added to the
    # labels
    yticks = []
    yticklabels = []

    # Iterate and Plot
    for item in plot_items:
        var = item['var']
        label = VAR_NAMES.get(var, var)
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
                color=VAR_COLOURS[var],
                edgecolor=edge_colour,
                alpha=bar_alpha_component,
                error_kw=dict(lw=1, capsize=3, capthick=1, ecolor=err_colour)
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

            # Make the axhlne the same colour as the bar to signify aggregate
            ax.axhline(y=y_pos, color=VAR_COLOURS[var], linewidth=1.5)

            # Plot Bar
            ax.barh(
                y_pos, med,
                left=0,  # Bar starts from the axis
                height=bar_height_aggregate,
                xerr=[[neg_err], [pos_err]],
                color=VAR_COLOURS[var],
                edgecolor=edge_colour,
                alpha=bar_alpha_aggregate,
                error_kw=dict(lw=1, capsize=3, capthick=1, ecolor=err_colour)
                )

            yticklabels.append(label)

            # Add Explanatory Text
            s = ""
            highlight_textprops = []

            if var == 'Ant':
                s = f"Sum of <{VAR_NAMES['GHG']}> and <{VAR_NAMES['OHF']}>"
                highlight_textprops = [
                    {"color": VAR_COLOURS['GHG'], "fontweight": "bold"},
                    {"color": VAR_COLOURS['OHF'], "fontweight": "bold"}
                ]
            elif var == 'Tot':
                s = f"Sum of <{VAR_NAMES['Ant']}> and <{VAR_NAMES['Nat']}>"
                highlight_textprops = [
                    {"color": VAR_COLOURS['Ant'], "fontweight": "bold"},
                    {"color": VAR_COLOURS['Nat'], "fontweight": "bold"}
                ]
            elif var == 'Obs':
                s = f"Sum of <{VAR_NAMES['Tot']}> and <{VAR_NAMES['Res']}>"
                highlight_textprops = [
                    {"color": VAR_COLOURS['Tot'], "fontweight": "bold"},
                    {"color": VAR_COLOURS['Res'], "fontweight": "bold"}
                ]
            else:
                s = "Sum of <components>"
                highlight_textprops = [
                    {"color": VAR_COLOURS.get(var, 'black')}
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

    # We need to connect the *end* of one component to the *start* of the next
    # component. Visually, the waterfall flow should persist across the
    # subtotals.

    # NOTE: The aggregates (subtotals GHG,OHF,Nat,Ant,Tot) will not necessarily
    # line up perfectly with the ends of the component sums due to the the fact
    # that these are percentiles across large ensembles and a multi-run mean
    # of those percentiles. In reality, at the ensemble-member level, the
    # variables will sum up to give the Obs (e.g. Tot + Res = Obs) exactly.

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
            label_obj.set_color(VAR_COLOURS[plot_items[i]['var']])

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
    fig.text(0.05, 0.95, f'Attributable contributions to warming ({period})',
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
        os.makedirs(plot_path)

    plot_name = (f'{plot_path}/' +
                 f'Waterfall_BarPlot_Scenario--{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    fig.savefig(plot_name)
    plt.close(fig)


def figure_priors_timeseries(
        scen, ens, reg_vars,
        priors_dfs, obs_dfs,
        var_colours):
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
        plot_vars_priors, var_colours,
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
        os.makedirs(plot_path)
    plot_name = (
        f'{plot_path}/' +
        f'Prior_Timeseries_Scenario--{scen}_' +
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


def figure_gif_animation(plot_names):
    """Create a gif animation of timeseries plots.

    Accepts a list of plot names (file paths) to include in the gif.
    """
    print('  Creating gif of timeseries plots for:',
          scen, ens, reg_vars)

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


def overarching_base_result_plotter(
    results_dfs,
    obs_dfs,
    priors_dfs,
    var_colours,
    var_names
):
    """Plot figures of base results."""

    print('\nPlotting single-run timeseries')
    for scen in results_dfs.keys():
        print('SCENARIO:', scen)
        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)
            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED_VARIABLES:', reg_vars)

                reg_ranges_all = sorted(
                    list(results_dfs[scen][ens][reg_vars].keys()))

                # Get all variables present in the data (from the first
                # available range)
                first_range = reg_ranges_all[0]
                plot_vars = results_dfs[scen
                                        ][ens
                                          ][reg_vars
                                            ][first_range
                                              ]['timeseries'
                                                ].columns.get_level_values(0).unique().to_list()

                # Get dynamic colours for variables present
                current_var_colours, scaling_map = gr.get_dynamic_colours(
                    reg_vars, plot_vars, var_colours)

                # Check if all years are available
                if defs.check_steps(reg_ranges_all)['check_bool']:
                    print('      All years available for: ',
                          defs.check_steps(reg_ranges_all)['range'])

                ###############################################################
                # 1. Plot GWI Timeseries

                single_toggle = toggle_single_timeseries(ens, 10)

                if single_toggle:
                    with mp.Pool(os.cpu_count()) as p:
                        print('        Plotting figure_timeseries for GWI')
                        # print('  in parallel for:', reg_ranges_all)
                        plot_names = p.map(
                            functools.partial(
                                # figure_timeseries,
                                # scen=scen, ens=ens, reg_vars=reg_vars
                                figure_timeseries,
                                scen=scen, ens=ens, reg_vars=reg_vars,
                                results_dfs=results_dfs,
                                df_temp_Obs=obs_dfs[scen][ens]['timeseries'],
                                VAR_COLOURS=current_var_colours
                                ),
                            reg_ranges_all
                        )

                    ###########################################################
                    # 2. Create GIF of Timeseries Plots

                    # Add a toggle, because this is quite slow for the SMILE
                    # ensembles (e.g. where we have an entirely different
                    # set of results for a different ensemble member).
                    gif_toggle = False
                    if gif_toggle:
                        figure_gif_animation(plot_names)

                ###############################################################
                # 3. Plot Priors Timeseries
                print('        Plotting figure_timeseries for PRIORS')
                figure_priors_timeseries(
                    scen, ens, reg_vars, priors_dfs, obs_dfs,
                    current_var_colours)

                ###############################################################
                # 4. Plot SPM2 Bar Plot
                print('        Plotting SPM2 for GWI in parallel')
                with mp.Pool(os.cpu_count()) as p:
                    p.map(
                        functools.partial(
                            figure_spm2,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            results_dfs=results_dfs,
                            obs_dfs=obs_dfs,
                            VAR_COLOURS=current_var_colours,
                            VAR_NAMES=var_names
                        ),
                        reg_ranges_all
                    )

                ###############################################################
                # 5. Plot Waterfall Plot
                print('        Plotting Waterfall for GWI in parallel')
                with mp.Pool(os.cpu_count()) as p:
                    p.map(
                        functools.partial(
                            figure_waterfall,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            results_dfs=results_dfs,
                            obs_dfs=obs_dfs,
                            VAR_COLOURS=current_var_colours,
                            VAR_NAMES=var_names
                        ),
                        reg_ranges_all
                    )


if __name__ == '__main__':

    argv_dict = parse_argvs()
    print(argv_dict)

    # Configuration
    if '--include-headlines' in argv_dict:
        headline_toggle = argv_dict['--include-headlines'] == 'y'
    else:
        headline_toggle = input('Include headlines? (y/n): ') == 'y'

    if '--re-calculate' in argv_dict:
        re_calculate = argv_dict['--re-calculate'] == 'y'
    else:
        re_calculate = input('Re-calculate? (y/n): ') == 'y'

    # Ensure directoriesfor plots and results exist
    for folder in [PLOT_FOLDER, AGGREGATED_FOLDER, ITERATIONS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # 1. Calculate Iterations
    calculate_iterations(
        re_calculate, headline_toggle, ITERATIONS_FOLDER, AGGREGATED_FOLDER)

    # 2. Load Data
    results_dfs, priors_dfs, obs_dfs = load_all_data(AGGREGATED_FOLDER)

    # NOTE:
    # results_files[reg_scen][reg_vars][reg_range][result_type].keys():
    # results_files[reg_scen][reg_vars][reg_range][result_type].keys():
    # Where result_type is timeseries, headlines
    # And reg_range is the range of years that the regression was performed
    # over, or 'historical-only', which is the range of years that the
    # historical-only dataset was calculated over.

    # 3. Plot the basic results
    overarching_base_result_plotter(
        results_dfs, obs_dfs, priors_dfs,
        VAR_COLOURS, VAR_NAMES)

    # 4. Generate historical-only timeseries and plot them.

    print('\nGenerating historical-only timeseries')
    for scen in sorted(results_dfs.keys()):
        print('SCENARIO:', scen)

        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)


            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED-VARIABLES:', reg_vars)
                # Create a new empty dataframe to store the historical-only results:
                reg_ranges_all = sorted(list(results_dfs[scen][ens][reg_vars].keys()))

                # Define the colours for the sub-variables
                # Get all variables present in the data (from the first available range)
                first_range = reg_ranges_all[0]
                plot_vars = results_dfs[scen][ens][reg_vars][first_range]['timeseries'].columns.get_level_values(0).unique().to_list()
                
                current_var_colours, scaling_map = gr.get_dynamic_colours(reg_vars, plot_vars, VAR_COLOURS)

                min_regressed_range = min(reg_ranges_all)
                max_regressed_range = max(reg_ranges_all)
                print(f'      Creating historical-only timeseries for {reg_vars}: between ' +
                    min_regressed_range + ' and ' + max_regressed_range)

                results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'] = {}
                results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY-PREHIST'] = {}
                # The prehist variant also includes the years before the earliest
                # regressed range, but with the same headline definitions as the
                # historical-only dataset. This is inconsistent with the way the
                # historical-only dataset is calculated, but is included as a
                # reference for plotting.
                # TODO: If I really want a full-information (instead of
                # historical-only) dataset using the various definitions, this will
                # need doing inside GWI.py (and could easily be added using a new argv
                # of 'all' alongside 'y' and 'n' in the headline_toggle).

                if headline_toggle:
                    headlines = ['ANNUAL', 'SR15', 'AR6', 'CGWL']
                else:
                    headlines = ['ANNUAL']
                for headline in headlines:
                    print(f'        Calculating historical-only for {headline}')
                    df_results_headlines, df_results_headlines_prehist = historical_only(
                        scen, ens, reg_vars, reg_ranges_all,
                        headline, headline_toggle,
                        results_dfs)

                    if not df_results_headlines.empty:
                        results_dfs[scen][ens][reg_vars][
                            'HISTORICAL-ONLY'].update({headline: df_results_headlines})
                        results_dfs[scen][ens][reg_vars][
                            'HISTORICAL-ONLY-PREHIST'].update({headline: df_results_headlines_prehist})

                smallest_end_year = min([int(reg_range.split('-')[1])
                                        for reg_range in reg_ranges_all])
                largest_end_year = max([int(reg_range.split('-')[1])
                                        for reg_range in reg_ranges_all])
                start_years = set([int(reg_range.split('-')[0])
                                for reg_range in reg_ranges_all])
                if len(start_years) == 1:
                    start_regress = list(start_years)[0]

                ###################################################################
                # Create directory

                # Create the overarching directory for the plots
                out_path = f'plots/aggregated/' + \
                    f'SCENARIO--{scen}/' + \
                    f'ENSEMBLE-MEMBER--{ens}/' + \
                    f'VARIABLES--{reg_vars}/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                ###################################################################
                # Plot each headline historical-only timeseries as its own plot
                print('      Plotting historical-only timeseries for:', scen, ens, reg_vars)
                for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
                    print('        Plotting:', headline)
                    plot_vars = results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][
                        headline].columns.get_level_values(0).unique().to_list()
                    
                    # Define major variables (for plumes)
                    plume_vars = [v for v in plot_vars if v in defs.SUB_VAR_MAPPING or v == 'Res']
                    
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
                        plot_vars, current_var_colours,
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
                    ax.set_ylim(-1, np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2)
                    ax.set_xlim(smallest_end_year, largest_end_year)
                    xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 5))
                    xticks.append(largest_end_year)
                    ax.set_xticks(xticks, xticks)
                    ax.set_title(
                        'Regressed years range: ' +
                        f'{min_regressed_range} to {max_regressed_range}')
                    gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)
                    
                    if legend_loc == 'center right':
                        plt.subplots_adjust(right=0.8)
                    fig.suptitle(
                        f'Historical-only {headline}\n' +
                        f'Scenario: {scen} | Ensemble: {ens} | Regressed variables: {reg_vars}')
                    fig.savefig(
                        'plots/aggregated/' +
                        f'SCENARIO--{scen}/' +
                        f'ENSEMBLE-MEMBER--{ens}/' +
                        f'VARIABLES--{reg_vars}/' +
                        f'Historical_only_{headline}_' +
                        f'{scen}_{ens}_{reg_vars}_' +
                        f'{min_regressed_range}_to_{max_regressed_range}.png')
                    plt.close(fig)

                #######################################################################
                # Plot the historical-only vs full dataset using gr.gwi_timeseries

                print('      Plotting historical-only vs full dataset for:',
                    scen, ens, reg_vars)
                
                # results_dfs are a very nested dictionary - please print the
                # nested keys for all levels, except the data at the bottom level:
                pprint.pprint({k: list(v.keys()) for k, v in results_dfs[scen][ens][reg_vars].items()})

                plot_vars = results_dfs[scen][ens][reg_vars][
                    'HISTORICAL-ONLY'][
                        'ANNUAL'].columns.get_level_values(0).unique().to_list()
                plot_vars_priors = priors_dfs[scen][ens][
                    'timeseries'].columns.get_level_values(0).unique().to_list()

                fig = plt.figure(figsize=(12, 8))
                ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

                gr.gwi_timeseries(
                    ax, obs_dfs[scen][ens]['timeseries'], None,
                    results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY']['ANNUAL'],
                    plot_vars, current_var_colours, sigmas=['5', '95', '50'],
                    hatch='\\', linestyle='dashed')
                var_linestyles = gr.get_dynamic_linestyles(plot_vars)
                gr.gwi_timeseries(
                    ax, obs_dfs[scen][ens]['timeseries'], None,
                    results_dfs[scen][ens][reg_vars][max_regressed_range]['timeseries'],
                    plot_vars, current_var_colours, sigmas=['5', '95', '50'],
                    hatch=None, linestyle=var_linestyles)

                ax.set_ylim(-1, np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2)
                ax.set_xlim(smallest_end_year, largest_end_year)
                xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 5))
                xticks.append(largest_end_year)
                ax.set_xticks(xticks, xticks)

                ax.set_title(
                    'Regressed years range: ' +
                    f'{min_regressed_range} to {max_regressed_range}')
                gr.overall_legend(fig, 'lower center', 6)

                fig.suptitle(
                    'Historical-only (dashed) versus Full-information (solid)\n' +
                    f'Scenario: {scen} | Ensemble: {ens} | Regressed variables: {reg_vars}')
                fig.savefig(
                    f'plots/aggregated/SCENARIO--{scen}/' +
                    f'ENSEMBLE-MEMBER--{ens}/' +
                    f'VARIABLES--{reg_vars}/' +
                    'ANNUAL_Historical_vs_Full_timeseries_' +
                    f'{scen}_{ens}_{reg_vars}_' +
                    f'{min_regressed_range}_to_{max_regressed_range}.png')
                plt.close(fig)

                #######################################################################
                # Plot comparison of all headlines datasets

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
                        current_var_colours, sigmas=['5', '95', '50'])

                # Plot the centered 20-year rolling window on the 50th percentile Obs
                df_temp_Obs_20yr = obs_dfs[scen][ens]['timeseries'].quantile(q=0.5, axis=1).rolling(
                    window=20, center=True, axis=0).mean()

                # for headline in headlines:
                for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
                    plot_vars_main = plot_vars.copy()
                    unwanted_vars = ['GHG', 'OHF', 'Res']
                    plot_vars_main = list(set(plot_vars_main) - set(unwanted_vars))
                    for vv in plot_vars_main:
                        # Determine line style
                        ls = HEADLINE_LINE_STYLE.get(vv)
                        if ls is None:
                            # Try to get style from parent
                            parent = scaling_map.get(vv)
                            ls = HEADLINE_LINE_STYLE.get(parent, 'solid')

                        # Plot the historical only timeseries
                        ax1.plot(results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline].index,
                                results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                                label=f'{headline}-{vv}',
                                linestyle=ls,
                                color=HEADLINE_COLOURS[headline]
                                )
                        if vv != 'Nat':
                            ax2.plot(
                                (results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline].loc[:, (vv, '50')]
                                - df_temp_Obs_20yr),
                                label=f'{headline}-{vv}',
                                linestyle=ls,
                                color=HEADLINE_COLOURS[headline]
                            )

                        # Calculate the full-information timeseries for the headlines

                        if headline == 'ANNUAL':
                            # Use the full-information timeseries for the annual headline
                            df_fullinfo_defs = results_dfs[scen][ens][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')]
                        elif headline == 'AR6':
                            # Calculate rolling 10-year mean, lagged
                            df_fullinfo_defs = results_dfs[scen][ens][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=10, center=False).mean()
                        elif headline == 'SR15':
                            # Calcualte the rolling 30-year mean, centered
                            df_fullinfo_defs = results_dfs[scen][ens][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=30, center=True).mean()
                        elif headline == 'CGWL':
                            # Calculate the rolling 20-year mean, centered
                            df_fullinfo_defs = results_dfs[scen][ens][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=20, center=True).mean()

                        ax3.plot(df_fullinfo_defs.index, df_fullinfo_defs,
                                label=f'{headline}-{vv}',
                                linestyle=ls,
                                color=HEADLINE_COLOURS[headline]
                                )
                        if vv != 'Nat':
                            ax4.plot(
                                (df_fullinfo_defs - df_temp_Obs_20yr),
                                label=f'{headline}-{vv}',
                                linestyle=ls,
                                color=HEADLINE_COLOURS[headline]
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
                min_y = np.floor(np.min(obs_dfs[scen][ens]['timeseries'].loc[smallest_end_year:largest_end_year].values) * 2) / 2
                min_y = min([-0.5, min_y])
                max_y = np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].loc[smallest_end_year:largest_end_year].values) * 2) / 2
                # min_y = np.floor(np.min(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2
                # max_y = np.ceil(np.max(obs_dfs[scen][ens]['timeseries'].values) * 2) / 2
                ax1.set_ylim(min_y, max_y)
                ax3.set_ylim(min_y, max_y)
                for ax in [ax1, ax2]:
                    ax.set_xlim(smallest_end_year, largest_end_year)
                gr.overall_legend(fig, 'lower center', 5)

                ax2.set_ylabel('$\Delta$ vs 20-year obs, ⁰C')
                ax4.set_ylabel('$\Delta$ vs 20-year obs, ⁰C')
                ax1.set_title('Historical-only')
                ax3.set_title('Full-information')
                fig.suptitle(
                    f'Historical-only and Full-information vs 20-year Obs running mean\n' +
                    f'Scenario: {scen} | Ensemble: {ens} | Regressed variables: {reg_vars}'
                )
                fig.savefig(f'plots/aggregated/SCENARIO--{scen}/' +
                            f'ENSEMBLE-MEMBER--{ens}/' +
                            f'VARIABLES--{reg_vars}/' +
                            'Historical_and_full_headlines_' +
                            f'{scen}_{ens}_{reg_vars}_' +
                            f'{min_regressed_range}_to_{max_regressed_range}.png')

                plt.close(fig)

                # Create one-off figure for Thorne et al paper
                fig = plt.figure(figsize=(12,8))
                ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))

                gr.gwi_timeseries(
                    ax, obs_dfs[scen][ens]['timeseries'], None, None, None,
                    current_var_colours,
                    sigmas=['5', '95', '50']
                    # hatch='\\', linestyle='dashed'
                )
                # for headline in headlines:
                for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
                    for vv in plot_vars_main:
                        # Determine line style
                        ls = HEADLINE_LINE_STYLE.get(vv)
                        if ls is None:
                            # Try to get style from parent
                            parent = scaling_map.get(vv)
                            ls = HEADLINE_LINE_STYLE.get(parent, 'solid')

                        # Plot the historical only timeseries
                        ax.plot(results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline].index,
                                results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                                label=f'{headline}-{vv}',
                                linestyle=ls,
                                color=HEADLINE_COLOURS[headline]
                                )
                ax.plot(df_temp_Obs_20yr.index, df_temp_Obs_20yr,
                        label='Obs 20-year running mean',
                        color='black'
                        )

                # Slice the df_temp_Obs using the smallest and largest end years
                ax.set_ylim(min_y, max_y)
                ax.set_xlim(smallest_end_year, largest_end_year + 0.5)
                ax.set_title('Calculated as historical-only: annual-mean, AR6 decade-mean, SR1.5 centered 30-year mean, and CGWL centered 20-year mean')
                gr.overall_legend(fig, 'lower center', 5)

                ax.set_ylabel('Attributable warming relative to 1850–1900 (⁰C)')
                fig.suptitle(
                    'Global Warming Index (GWI)'
                )

                fig.savefig(f'plots/aggregated/SCENARIO--{scen}/' +
                            f'ENSEMBLE-MEMBER--{ens}/' +
                            f'VARIABLES--{reg_vars}/' +
                            'Historical_only_headlines_' +
                            f'{scen}_{ens}_{reg_vars}_' +
                            f'{min_regressed_range}_to_{max_regressed_range}.png')
                plt.close(fig)


        ###############################################################################
        # Generate the projected warming for final constrained year ###################
        ###############################################################################
                # TODO: Add to constrained warming dictionary.
                # TODO: Move calculation higher up in script.

                print('      Creating constrained results for:', reg_vars)
                # Calculate how the expected final year of the timeseries changes
                # depending on the years that are regressed. Expect that the attributed
                # values in 2023 (end year of the full timeseries) will have larger
                # uncertainties, the earlier/shorter the range of regressed years is.

                # Create new empty dataframes to store the constrained results:
                # NOTE: you could also do this using maximum of the truncation range
                # if that's what you're interested in (possibly more relevant for
                # SSP projections in future)

                # constrained_year = int(max_regressed_range.split('-')[1])
                constrained_year = largest_end_year

                df_constrained = results_dfs[scen][ens][reg_vars][reg_ranges_all[0]]['timeseries'].copy()
                df_constrained[:] = 0

                # For each iteration, add the final row of the dataframe to the new
                # df_hist. The row index it should be inserted at is the same as the
                # second year in the iteration name.
                for reg_range in reg_ranges_all:
                    # print(iteration, iteration.split('-')[1], constrained_year)
                    df_constrained.loc[int(reg_range.split('-')[1])] = \
                        results_dfs[scen][ens][reg_vars][reg_range]['timeseries'].loc[constrained_year]

                # Remove all years that are not the end of an attribution period to
                # avoid confusion:
                df_constrained = df_constrained.loc[smallest_end_year:, :]

                #######################################################################
                # Plot this dataframe df_constrined in the same way as df_hist

                print('        Plotting constrained results for:', reg_vars)
                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot2grid(
                    shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
                ax2 = plt.subplot2grid(
                    shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)
                
                var_linestyles = gr.get_dynamic_linestyles(plot_vars_priors)
                gr.gwi_timeseries(
                    ax1, None, None, df_constrained,
                    plot_vars_priors, current_var_colours, sigmas=['5', '95', '50'],
                    linestyle=var_linestyles)

                # ax1.set_ylim(
                #     np.floor(np.min(df_constrained.values) * 2) / 2,
                #     np.ceil(np.max(df_constrained.values) * 2) / 2)
                ax1.set_xlim(smallest_end_year, largest_end_year)
                ax1.set_ylabel(f'Warming in {constrained_year} ⁰C')
                ax1.set_xlabel(f'Regressed years: {start_regress}-<year>')
                ax1.set_xticks(xticks, xticks)
                ax1.set_title(f'Constrained: {constrained_year} (with Obs only up to year <year>)')


                # Create box and whisker plot for prior warming in each variable
                bar_width = 0.4

                for vv in plot_vars_priors:
                    # Plot the multi-method assessed results for the 2010-2019 period
                    med_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year, (vv, '50')]
                    min_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year, (vv, '5')]
                    max_prior = priors_dfs[scen][ens]['timeseries'].loc[constrained_year, (vv, '95')]

                    ax2.fill_between(
                        [plot_vars.index(vv), plot_vars.index(vv) + bar_width],
                        min_prior, max_prior,
                        color=current_var_colours[vv],
                        alpha=0.6,
                        linewidth=0,
                        label=vv
                        )

                    ax2.plot(
                        [plot_vars.index(vv), plot_vars.index(vv) + bar_width],
                        [med_prior, med_prior],
                        color=current_var_colours[vv],
                        lw=2)

                    # Add horizontal lines in the variable colours for the min,
                    # med, and max values of each variable in ax1
                    # for val in [min_prior, med_prior, max_prior]:
                    #     ax1.hlines(
                    #         y=val, xmin=smallest_end_year,
                    #         xmax=largest_end_year,
                    #         colors=VAR_COLOURS[vv], linestyles='dotted', lw=1)
                    
                # Remove the xticks in ax2
                ax2.set_xticks([])
                ax2.set_yticklabels([])
                # Get the ylims from ax1
                ax2.set_ylim(ax1.get_ylim())
                ax2.set_title(f'Unconstrained: {constrained_year}')

                gr.overall_legend(fig, 'lower center', 6)


                fig.suptitle(f'Constrained projected warming in {constrained_year}\n' +
                             f'Scenario: {scen} | Ensemble: {ens} | Regressed variables: {reg_vars}')
                fig.savefig(
                    f'plots/aggregated/SCENARIO--{scen}/' +
                    f'ENSEMBLE-MEMBER--{ens}/' +
                    f'VARIABLES--{reg_vars}/' +
                    f'Projected_warming_in_{constrained_year}_' +
                    f'regressing_{reg_vars}_'
                    'constrained_by_regressed_years_' +
                    f'{min_regressed_range}_to_{max_regressed_range}.png')
                plt.close(fig)


        ###############################################################################
        # Generate timeseries showing source of changes in GWI value each year ########
        ###############################################################################
                # TODO: This whole multi-plot figure needs fixing, revising,
                # sorting, etc

                # From year Y to year Y+1, you have contributions from:
                # 1. the change in temp in year Y in the old dataset to year Y in the
                # new dataset
                # 2. the change in temp in year Y+1 in the old dataset to the temp in
                # year Y in the new dataset.
                # 3. any changes in historical forcing in the new dataset
                # 4. any changes in HadCRUT temperatures in the new dataset
                # Only factor 1 and 2 are considered in this calcualtion. The other
                # factors may be added later, but sourcing historical T and ERF data is
                # significantly more wrangling.

                print('      Creating delta contributions for:', scen, reg_vars)

                fig = plt.figure(figsize=(12, 10))
                ax1 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=1)
                ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=1, colspan=1)

                # Create a new empty dataframe copied from before:
                df_delta_additional_forcing_year = df_constrained.copy()
                df_delta_revised_previous_year = df_constrained.copy()
                df_delta_additional_forcing_year[:] = 0
                df_delta_revised_previous_year[:] = 0

                differ_years = sorted([r.split('-')[1] for r in reg_ranges_all])
                # switch the sorted order of the list years
                differ_years = differ_years[::-1]
                # remove the smallest year
                differ_years = differ_years[:-1]

                for y in differ_years:
                    # delta_new is the change from year Y to Y+1 in the new dataset.
                    delta_new = (
                        results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)] -
                        results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)-1])
                    # delta_rev is the change to the year Y from the previous to the
                    # new dataset.
                    delta_rev = (
                        results_dfs[scen][ens][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)-1] -
                        results_dfs[scen][ens][reg_vars][f'{start_regress}-{int(y)-1}']['timeseries'].loc[int(y)-1])
                    df_delta_additional_forcing_year.loc[int(y)] = delta_new
                    df_delta_revised_previous_year.loc[int(y)] = delta_rev
                
                #######################################################################
                # Plot the results

                # The red line is the additional warming in year Y+1 relative to year
                # Y in the new dataset.
                # The blue line is the revised warming in year Y calculated in the year
                # Y+1 dataset relative to the year Y calculated in the year Y dataset.
                # The green dashed line is the residual warming in year Y relative in
                # the dataset for year Y. That is so say, this line comes from the
                # historical-only dataset.
                print('        Plotting delta contributions for:', reg_vars)

                line_alpha = 0.9

                changing_var = 'Ant' if 'Ant' in plot_vars else 'Tot'
                # Plot the residual in the

                df_hist = results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY']['ANNUAL']

                # Plot the 
                ax1.fill_between(
                    # df_delta_additional_forcing_year.index,
                    df_hist.loc[smallest_end_year:, ('Res', '5')].index,
                    df_hist.loc[smallest_end_year:, ('Res', '5')].values,
                    df_hist.loc[smallest_end_year:, ('Res', '95')].values,
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
                    df_hist.loc[smallest_end_year:, ('Res', '50')].index,
                    df_hist.loc[smallest_end_year:, ('Res', '50')].values,
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
                xticks.append(int(min(differ_years)))
                ax1.set_xticks(xticks, xticks)
                ax1.set_xlim(int(min(differ_years)), int(max(differ_years))+0.5)
                ax1.set_ylim(-0.3, +0.3)
                ax1.set_ylabel('Interannual warming delta, ⁰C')

                # #######################################################################
                # # Plot correlation between interannual delta and residual warming
                # TODO: Complete this plot - commented out to keep commit clean.

                # # Calculate the correlation between the residual warming and the
                # # interannual delta in the new dataset.
                # ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 1), rowspan=1, colspan=1)

                # ax3.plot(
                #     df_hist.loc[smallest_end_year:, ('Res', '50')].values,
                #     df_delta_revised_previous_year.loc[:, (changing_var, '50')].values,
                #     # label='Residual (internal variability) in year Y+1',
                #     # color='seagreen', ls='dashed', alpha=line_alpha)
                # )
                # ax3.set_ylabel('Revised warming in year Y')
                # ax3.set_xlabel('Residual (internal variability) in year Y+1')

                # # Calculate the correlation between these two, and plot the line
                # corr = np.corrcoef(
                #     df_hist.loc[smallest_end_year:, ('Res', '50')].values,
                #     df_delta_revised_previous_year.loc[:, (changing_var, '50')].values)
                # print(corr)
                # # ax3.plot(df_hist.loc[smallest_end_year:, ('Res', '50')].values,
                # #          np.poly1d(np.polyfit(
                # #              df_hist.loc[smallest_end_year:, ('Res', '50')].values,
                # #              df_delta_revised_previous_year.loc[:, (changing_var, '50')].values,
                # #              1)))

                # ax3.set_title(f'Correlation: {corr[0, 1]:.2f}')

                #######################################################################
                # Plot schematic diagram
                years = list(range(largest_end_year, largest_end_year-4, -1))
                # which [2023, 2022, 2021, 2020] when the end year is 2023.
                for year in years:
                    df_new = results_dfs[scen][ens][reg_vars][f'{start_regress}-{year}']['timeseries']
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
                ax2.errorbar(obs_dfs[scen][ens]['timeseries'].index, obs_dfs[scen][ens]['timeseries'].quantile(q=0.5, axis=1),
                            yerr=(err_neg, err_pos),
                            fmt='o', color=current_var_colours['Obs'], ms=2.5, lw=1,
                            label='Reference Temp: HadCRUT5')

                ax2.set_ylabel('Global Warming, ⁰C')
                ax2.set_xticks(years, years)
                # ax2.set_xlim(2019.5, 2023.5)
                ax2.set_xlim(largest_end_year-4+0.5, largest_end_year+0.5)
                # ax2.set_ylim(
                #     np.floor(obs_dfs[scen][ens]['timeseries'].loc[largest_end_year-4:, :].min().min() * 10) / 10,
                #     np.ceil(obs_dfs[scen][ens]['timeseries'].loc[largest_end_year-4:, :].min().min() * 10) / 10
                #     # 1.1, 1.5
                #     )

                fig.suptitle(
                    f'Contributions to the change in {changing_var} warming ' +
                    'each year Y → Y+1')
                gr.overall_legend(fig, 'lower center', 3)
                fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
                fig.savefig(
                    f'plots/aggregated/SCENARIO--{scen}/' +
                    f'ENSEMBLE-MEMBER--{ens}/' +
                    f'VARIABLES--{reg_vars}/' +
                    'Historical_delta_contributions_' +
                    f'{reg_vars}_{min_regressed_range}_to_{max_regressed_range}.png')
                plt.close(fig)

                # Compare variation between internal variation (using Residual as a
                # proxy for this, because ideally speaking, all forced warming is
                # accounted for, so the remaining should largely be internal
                # variability). Use RMS:
                delta_rms = np.sqrt(
                    np.mean(df_delta_revised_previous_year.loc[:, (changing_var, '50')
                                                            ].values**2))
                residual_rms = np.sqrt(
                    np.mean(df_hist.loc[smallest_end_year:, ('Res', '50')].values**2))

                print(f'          Revision RMS for {reg_vars}: {delta_rms}')
                print(f'          Residual RMS for {reg_vars}: {residual_rms}')
                print(f'          Average fractional variation for {reg_vars}:',
                    delta_rms / residual_rms)
