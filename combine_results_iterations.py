import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import src.graphing as gr
import src.definitions as defs
import json
import multiprocessing as mp
import functools

# NOTE:
# results_files[reg_scen][reg_vars][reg_range][result_type].keys():
# results_files[reg_scen][reg_vars][reg_range][result_type].keys():
# Where result_type is timeseries, headlines
# And reg_range is the range of years that the regression was performed over,
# or 'historical-only', which is the range of years that the historical-only
# dataset was calculated over.



# Get the command line arguments for which iterations to average across.
# argv format: --ensemble-size=ensemble_size --regressed-years=regressed_years
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

print(argv_dict)

var_colours = {'Tot': '#d7827e',
               'Ant': '#b4637a',
               'GHG': '#907aa9',
               'Nat': '#56949f',
               'OHF': '#ea9d34',
               'Res': '#9893a5',
               'Obs': '#797593',
               'PiC': '#cecacd'}

# Removed this for now: instead average across all iterations, weighting by
# the number of samples in each iteration.
# # Specify which ensemble size to average across
# if '--ensemble-size' in argv_dict:
#     ensemble_size = argv_dict['--ensemble-size']
# else:
#     ensemble_size = input('Sample size to average across (int): ')

# Removed this for now: instead automativally calculate for all choices for the
# choice of regressed years.
# # Specify which regressed years to average across
# if '--regressed-years' in argv_dict:
#     regressed_years = argv_dict['--regressed-years']
# else:
#     regressed_years = input('Regressed years to average across (y1-yn): ')

# Specify whether to include headline results
if '--include-headlines' in argv_dict:
    headline_toggle = argv_dict['--include-headlines']
    headline_toggle = True if headline_toggle == 'y' else False
else:
    headline_toggle = input('Include headlines? (y/n): ')
    headline_toggle = True if headline_toggle == 'y' else False

if '--re-calculate' in argv_dict:
    re_calculate = argv_dict['--re-calculate']
    re_calculate = True if re_calculate == 'y' else False
else:
    re_calculate = input('Re-calculate? (y/n): ')
    re_calculate = True if re_calculate == 'y' else False

plot_folder = 'plots/'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
aggregated_folder = 'results/aggregated'
if not os.path.exists(aggregated_folder):
    os.makedirs(aggregated_folder)
iterations_folder = 'results/iterations'
if not os.path.exists(iterations_folder):
    os.makedirs(iterations_folder)


# AVERAGE THE TIMESERIES AND HEADLINES ITERATIONS #############################
def combine_repeats(result_type, scenario, regressed_years, regressed_vars):
    dict_iterations = {}
    size_iterations = {}

    iteration_files = [
        f for f in os.listdir(f'{iterations_folder}/SCENARIO--{scenario}/' +
                              f'VARIABLES--{regressed_vars}/' +
                              f'REGRESSED-YEARS--{regressed_years}/')
        if result_type in f]
    if len(iteration_files) == 0:
        print('No iterations found for:',
              result_type, scenario, regressed_years, regressed_vars)
        return None, None, None

    # Remove previously averaged dataset in case it already exists
    iterations_dates = [f.split('_DATE-CALCULATED--')[-1].split('.')[0]
                        for f in iteration_files]
    for iteration in iteration_files:

        fname = f'{iterations_folder}/SCENARIO--{scenario}/' + \
            f'VARIABLES--{regressed_vars}/' + \
            f'REGRESSED-YEARS--{regressed_years}/' + \
            f'{iteration}'

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

    out_path = f'{aggregated_folder}/' + \
               f'SCENARIO--{scenario}/' + \
               f'VARIABLES--{regressed_vars}/' + \
               f'REGRESSED-YEARS--{regressed_years}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    df_avg.to_csv(
        f'{out_path}' +
        f'GWI_results_{result_type}_' +
        f'SCENARIO--{scenario}_'
        f'VARIABLES--{regressed_vars}_' +
        f'REGRESSED-YEARS--{regressed_years}_' +
        'AVERAGE.csv')

    return df_avg, dict_iterations, size_iterations


# Loop through all regressed years that are available.
if re_calculate:
    scenarios_all = sorted(
        [d.split('SCENARIO--')[1] for d in os.listdir(iterations_folder)])

    for scenario in scenarios_all:
        print('Calculating SCENARIO:', scenario)

        regressed_variables_all = sorted(
            [d.split('VARIABLES--')[1]
             for d in os.listdir(f'{iterations_folder}/SCENARIO--{scenario}/')
             ])

        print('All regressed variables for scenario:',
              regressed_variables_all)

        for regressed_vars in regressed_variables_all:
            _path = (f'{iterations_folder}/SCENARIO--{scenario}/' +
                     f'VARIABLES--{regressed_vars}/')
            regressed_years_vars = sorted(
                [d.split('REGRESSED-YEARS--')[1]
                 for d in os.listdir(_path)
                 if os.path.isdir(f'{_path}{d}')
                 ])

            print(f'Regressed years for {regressed_vars}:',
                  regressed_years_vars)

            for regressed_years in regressed_years_vars:
                print('Calculating:', scenario, regressed_years, regressed_vars)
                # Timeseries averaging
                df_avg, dict_iterations, size_iterations = combine_repeats(
                    'timeseries', scenario, regressed_years, regressed_vars)

                # Headlines may not have been calculated if only timeseries
                # were needed
                if headline_toggle:
                    combine_repeats(
                        'headlines', scenario, regressed_years, regressed_vars)

                # Get the variable names from the dataframe:
                vars_all = df_avg.columns.get_level_values(
                    0).unique().to_list()

                # Plot each iteration of the data to check
                fig = plt.figure(figsize=(15, 10))
                ax1 = plt.subplot2grid(
                    shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)
                ax2 = plt.subplot2grid(
                    shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)
                for sigma in ['5', '95', '50']:
                    for var in vars_all:
                        for iteration in dict_iterations.keys():
                            ax1.plot(df_avg.index,
                                     dict_iterations[iteration][(var, sigma)],
                                     label=f'{iteration} {var}',
                                     color=var_colours[var],
                                     alpha=1 if sigma == '50' else 0.5
                                     #  linestyle=linestyles[iteration]
                                     )
                        ax1.plot(df_avg.index, df_avg[(var, sigma)],
                                 label=f'Avg {var}', color='black')
                # plt.legend()
                ax1.set_ylabel('Iteration results, ⁰C')
                ax1.set_title('Multiple iterations of samples,' +
                              '5th, 50th, 95th percentiles\n' +
                              'ensembles sizes: ' +
                              str(list(size_iterations.values())))

                # Plot difference between the data in each iteration and the
                # average
                for iteration in dict_iterations.keys():
                    for sigma in ['5', '95', '50']:
                        for var in vars_all:
                            ax2.plot(
                                df_avg.index,
                                (dict_iterations[iteration][(var, sigma)]
                                 - df_avg[(var, sigma)]),
                                label=f'{iteration} {var}',
                                color=var_colours[var],
                                alpha=1 if sigma == '50' else 0.5
                                #  linestyle=linestyles[iteration]
                                )
                # plt.legend()
                ax2.set_ylabel('Iteration minus average, ⁰C')
                ax2.set_title('Difference between iterations and average, ' +
                              '5th, 50th, 95th percentiles')
                fig.suptitle(
                    'Iterations for regressed years: ' +
                    f'{regressed_years} and regressed vars: {regressed_vars}')
                gr.overall_legend(fig, 'lower center', 6)

                plot_path = f'{plot_folder}/iterations/' + \
                            f'SCENARIO--{scenario}/' + \
                            f'VARIABLES--{regressed_vars}/' + \
                            f'REGRESSED-YEARS--{regressed_years}/'
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                fig.savefig(
                    f'{plot_path}Compare_iterations_' +
                    f'{scenario}_{regressed_vars}_{regressed_years}.png')
                plt.close(fig)


###############################################################################
# Load all averaged datasets
###############################################################################
# Get a list of all files with 'AVERAGE' in them:
results_files = {}

scenarios_all = sorted(
        [d.split('SCENARIO--')[1] for d in os.listdir(aggregated_folder)])
for scenario in scenarios_all:
    results_files.update({scenario: {}})
    regressed_variables_all = sorted(
            [d.split('VARIABLES--')[1] for d in
             os.listdir(f'{aggregated_folder}/SCENARIO--{scenario}/')])

    for regressed_vars in regressed_variables_all:
        results_files[scenario].update({regressed_vars: {}})
        _path = (f'{aggregated_folder}/SCENARIO--{scenario}/' +
                 f'VARIABLES--{regressed_vars}/')
        regressed_years_vars = sorted(
                [d.split('REGRESSED-YEARS--')[1] for d in
                 os.listdir(_path) if os.path.isdir(f'{_path}{d}')])
        for regressed_years in regressed_years_vars:
            results_files[scenario][regressed_vars].update({
                regressed_years: {
                    res_type: (
                        f'{aggregated_folder}/SCENARIO--{scenario}/' +
                        f'VARIABLES--{regressed_vars}/' +
                        f'REGRESSED-YEARS--{regressed_years}/' +
                        f'GWI_results_{res_type}_' +
                        f'SCENARIO--{scenario}_'
                        f'VARIABLES--{regressed_vars}_' +
                        f'REGRESSED-YEARS--{regressed_years}_' +
                        'AVERAGE.csv'
                    )
                    for res_type in ['timeseries', 'headlines']
                }
            })
# print(json.dumps(results_files, indent=4))


print('Loading all averaged datasets')
results_dfs = results_files.copy()
for reg_scen in results_files.keys():
    for reg_vars in results_files[reg_scen].keys():
        for reg_range in results_files[reg_scen][reg_vars].keys():
            for res_type in results_files[reg_scen][reg_vars][reg_range].keys():
                df_ = pd.read_csv(
                        results_files[reg_scen
                                      ][reg_vars
                                        ][reg_range
                                          ][res_type],
                        index_col=0,  header=[0, 1], skiprows=0)
                results_dfs[reg_scen][reg_vars][reg_range][res_type] = df_


###############################################################################
# Plot results ################################################################
###############################################################################

def single_timeseries(reg_range, scen, reg_vars):
    """Plot single timeseries plots."""
    print('Creating single timeseries plots for:',
          scen, reg_vars, reg_range, end='\r')
    plot_vars = results_dfs[
        scen][reg_vars][reg_range]['timeseries'].columns.get_level_values(
            0).unique().to_list()

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

    reg_start = int(reg_range.split('-')[0])
    reg_end = int(reg_range.split('-')[1])
    trunc_start = results_dfs[scen][reg_vars][reg_range]['timeseries'].index.min()
    trunc_end = results_dfs[scen][reg_vars][reg_range]['timeseries'].index.max()

    gr.gwi_timeseries(
        ax, df_temp_Obs, None,
        results_dfs[scen][reg_vars][reg_range]['timeseries'].loc[reg_end:, :],
        plot_vars, var_colours, hatch='x', linestyle='dashed')

    gr.gwi_timeseries(
        ax, df_temp_Obs, None,
        results_dfs[scen][reg_vars][reg_range]['timeseries'].loc[
            reg_start:reg_end, :],
        plot_vars, var_colours)

    ax.set_ylim(-1, np.ceil(np.max(df_temp_Obs.values) * 2) / 2)
    ax.set_xlim(trunc_start, trunc_end)
    gr.overall_legend(fig, 'lower center', 6)

    # Plot a box around the regressed years
    ax.axvline(int(reg_range.split('-')[1]),
               color='darkslategray', linestyle='--')

    fig.suptitle(f'Scenario--{scen} Regressed--{reg_vars}_{reg_range}'
                 + 'Timeseries Plot')

    plot_path = ('plots/aggregated/' +
                 f'SCENARIO--{scen}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plot_name = (f'{plot_path}/' +
                 f'Timeseries_Scenario--{scen}_' +
                 f'Regressed--{reg_vars}_{reg_range}.png')
    # plot_names.append(plot_name)
    fig.savefig(plot_name)
    plt.close(fig)
    return plot_name


for scen in results_dfs.keys():
    df_temp_Obs = defs.load_Temp(scenario=scen, start_pi=1850, end_pi=1900)
    # df_temp_Obs = defs.load_Temp(scenario=scen, start_pi=1980, end_pi=2010)

    for reg_vars in sorted(results_dfs[scen].keys()):
        ###################################################################
        # Plot the timeseries for each iteration ##########################
        ###################################################################

        reg_ranges_all = sorted(list(results_dfs[scen][reg_vars].keys()))

        # for reg_range in reg_ranges_all:
        # This code was just the code inside the single_timeseries function
        # above, separated in order to parallelise to speed up code.
        with mp.Pool(os.cpu_count()) as p:
            print('multiprocessing for:', scen, reg_vars)
            plot_names = p.map(
                functools.partial(
                    single_timeseries, scen=scen, reg_vars=reg_vars),
                reg_ranges_all)

        print('')
        print('All years complete: ', reg_ranges_all)

        #######################################################################
        # Create a gif of the timeseries plots
        #######################################################################
        # Add a toggle, because this is quite slow for the SMILE ensembles.
        gif_toggle = False
        if gif_toggle:
            print('Creating gif of timeseries plots for:', reg_vars)

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
                f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
                f'Timeseries-animation_Scenario--{scen}_Regressed--{reg_vars}_' +
                f'{min(reg_ranges_all)}_to_{max(reg_ranges_all)}.gif',
                save_all=True, append_images=images_list[1:],
                optimize=False, duration=500, loop=0)


###############################################################################
# Generate the historical-only timeseries #####################################
###############################################################################

for scen in sorted(results_dfs.keys()):
    df_temp_Obs = defs.load_Temp(scenario=scen, start_pi=1850, end_pi=1900)
    # df_temp_Obs = defs.load_Temp(scenario=scen, start_pi=1980, end_pi=2010)

    for reg_vars in sorted(results_dfs[scen].keys()):
        # Create a new empty dataframe to store the historical-only results:
        reg_ranges_all = sorted(list(results_dfs[scen][reg_vars].keys()))

        min_regressed_range = min(reg_ranges_all)
        max_regressed_range = max(reg_ranges_all)
        print(f'Creating historical-only timeseries for {reg_vars}: between ' +
              min_regressed_range + ' and ' + max_regressed_range)

        results_dfs[scen][reg_vars]['HISTORICAL-ONLY'] = {}
        results_dfs[scen][reg_vars]['HISTORICAL-ONLY-PREHIST'] = {}
        # The prehist variant also includes the years before the earliest
        # regressed range, but with the same headline definitions as the
        # historical-only dataset. This is inconsistent with the way the
        # historical-only dataset is calculated, but is included as a
        # reference for plotting.
        # TODO: If I really want a full-information (instead of
        # historical-only) dataset using the various definitions, this will
        # need doing inside GWI.py (and could easily be added using a new argv
        # of 'all' alongside 'y' and 'n' in the headline_toggle).

        def historical_only(scen, reg_vars, reg_ranges_all,
                            headline, headline_toggle, results_dfs):
            """Calculate historical-only timeseries for each headline."""
            print(f'Calculating historical-only for {headline}')
            # Prepare empty timeseries for each headline
            df_hist_headline = results_dfs[
                scen][reg_vars][reg_ranges_all[0]]['timeseries'].copy()
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

                if headline_time in results_dfs[
                    scen][reg_vars][reg_range][res_type].index:

                    # print(f'SUCCESS: Headline time {headline} {headline_time} found in ' +
                    #       f'{scen} {reg_vars} {reg_range} {res_type}')

                    _df = results_dfs[
                        scen][reg_vars][reg_range][res_type
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

            # Save the dataframe to a csv file
            df_hist_headline.to_csv(
                f'{aggregated_folder}/SCENARIO--{scen}/VARIABLES--{reg_vars}/'
                f'GWI_results_{headline}_HISTORICAL-ONLY_' +
                f'SCENARIO--{scen}_VARIABLES--{reg_vars}_' +
                f'__REGRESSED-YEARS--{min_regressed_range}' +
                f'_to_{max_regressed_range}.csv')

            ###################################################################
            # NOTE: commented out for now, as I have no use for this now that
            # checks are complete.

            # # Concatenate the previous timeperiods before the earliest
            # # regressed range.

            # # NOTE: This technically is inconsistent with the way the the full
            # # historical-only timeseries is calcualted, since the headlines for
            # # the full historical-only timeseries are calculated within the 
            # # entire GWI ensemble. This pre-historical-only extension is
            # # included as a optional historical reference, e.g. for plotting.

            # # Select the timeseries from the earliest regressed range
            # historical_piece = results_dfs[
            #     scen][reg_vars][min(reg_ranges_all)]['timeseries']
            # print('\nHere is the pre-filtered index for historical piece')
            # print(historical_piece.index)
            # # Apply the headlie-specific running means over this piece
            # if headline == 'AR6':
            #     historical_piece_filtered = historical_piece.rolling(
            #         window=10, center=False).mean()
            # elif headline == 'SR15':
            #     historical_piece_filtered = historical_piece.rolling(
            #         window=30, center=True).mean()
            # elif headline == 'CGWL':
            #     historical_piece_filtered = historical_piece.rolling(
            #         window=20, center=True).mean()
            # else:
            #     historical_piece_filtered = historical_piece
            # historical_piece_filtered = historical_piece_filtered.loc[
            #         :smallest_end_year-1, :]
            # print('\nHere is the post-filtered index for historical piece:')
            # print(historical_piece_filtered.index)
            # # Concatenate the historical piece with the historical-only
            # # dataset

            # df_hist_headline_prehist = pd.concat(
            #     [historical_piece_filtered, df_hist_headline] ,
            #     axis=0).sort_index()
            # print('\nHere is the index for the full historical-only dataset:')
            # print(df_hist_headline_prehist.index)

            # # Check whether there are any rows that are duplicates
            # # (if overlapping)
            # if df_hist_headline_prehist.index.duplicated().any():
            #     df_hist_headline_prehist = df_hist_headline_prehist.drop_duplicates()

            # return df_hist_headline, df_hist_headline_prehist
            ###################################################################

            return df_hist_headline, None

        if headline_toggle:
            headlines = ['ANNUAL', 'SR15', 'AR6', 'CGWL']
        else:
            headlines = ['ANNUAL']
        for headline in headlines:
            df_results_headlines, df_results_headlines_prehist = historical_only(
                scen, reg_vars, reg_ranges_all,
                headline, headline_toggle,
                results_dfs)

            results_dfs[scen][reg_vars][
                'HISTORICAL-ONLY'].update({headline: df_results_headlines})
            results_dfs[scen][reg_vars][
                'HISTORICAL-ONLY-PREHIST'].update({headline: df_results_headlines_prehist})

        smallest_end_year = min([int(reg_range.split('-')[1])
                                 for reg_range in reg_ranges_all])
        largest_end_year = max([int(reg_range.split('-')[1])
                                for reg_range in reg_ranges_all])
        start_years = set([int(reg_range.split('-')[0])
                           for reg_range in reg_ranges_all])
        if len(start_years) == 1:
            start_regress = list(start_years)[0]

        #######################################################################
        # Plot each headline historical-only timeseries as its own plot
        print('Plotting historical-only timeseries for:', scen, reg_vars)
        for headline in results_dfs[scen][reg_vars]['HISTORICAL-ONLY'].keys():
            print('Plotting:', headline)
            plot_vars = results_dfs[scen][reg_vars]['HISTORICAL-ONLY'][
                headline].columns.get_level_values(0).unique().to_list()
            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0),
                                  rowspan=1, colspan=1)

            gr.gwi_timeseries(
                ax, df_temp_Obs, None,
                results_dfs[scen][reg_vars]['HISTORICAL-ONLY'][headline],
                plot_vars, var_colours,
                sigmas=['5', '95', '50'],
                # hatch='\\', linestyle='dashed'
                hatch=None, linestyle='solid'
                )
            ax.set_ylim(-1, np.ceil(np.max(df_temp_Obs.values) * 2) / 2)
            ax.set_xlim(smallest_end_year, largest_end_year)
            xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 5))
            xticks.append(largest_end_year)
            ax.set_xticks(xticks, xticks)
            ax.set_title(
                'Regressed years range: ' +
                f'{min_regressed_range} to {max_regressed_range}')
            gr.overall_legend(fig, 'lower center', 6)
            fig.suptitle(
                f'Historical-only {headline}\n' +
                f'Scenario: {scen} | Regressed variables: {reg_vars}')
            fig.savefig(
                f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
                f'Historical_only_{headline}_' +
                f'{scen}_{reg_vars}_' +
                f'{min_regressed_range}_to_{max_regressed_range}.png')
            plt.close(fig)

        #######################################################################
        # Plot the historical-only vs full dataset using gr.gwi_timeseries

        print('Plotting historical-only vs full dataset for:', scen, reg_vars)
        plot_vars = results_dfs[scen][reg_vars][
            'HISTORICAL-ONLY'][
                'ANNUAL'].columns.get_level_values(0).unique().to_list()

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

        gr.gwi_timeseries(
            ax, df_temp_Obs, None,
            results_dfs[scen][reg_vars]['HISTORICAL-ONLY']['ANNUAL'],
            plot_vars, var_colours, sigmas=['5', '95', '50'],
            hatch='\\', linestyle='dashed')
        gr.gwi_timeseries(
            ax, df_temp_Obs, None,
            results_dfs[scen][reg_vars][max_regressed_range]['timeseries'],
            plot_vars, var_colours, sigmas=['5', '95', '50'],
            hatch=None, linestyle='solid')

        ax.set_ylim(-1, np.ceil(np.max(df_temp_Obs.values) * 2) / 2)
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
            f'Scenario: {scen} | Regressed variables: {reg_vars}')
        fig.savefig(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
            'ANNUAL_Historical_vs_Full_timeseries_' +
            f'{scen}_{reg_vars}_' +
            f'{min_regressed_range}_to_{max_regressed_range}.png')
        plt.close(fig)

        #######################################################################
        # Plot comparison of all headlines datasets

        print('Plotting historical-only and full-information headlines:',
              scen, reg_vars)

        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), rowspan=3, colspan=1)
        ax2 = plt.subplot2grid(shape=(4, 2), loc=(3, 0), rowspan=1, colspan=1)
        ax3 = plt.subplot2grid(shape=(4, 2), loc=(0, 1), rowspan=3, colspan=1)
        ax4 = plt.subplot2grid(shape=(4, 2), loc=(3, 1), rowspan=1, colspan=1)

        headline_colours = {
            'ANNUAL': '#5BA2D0',
            'SR15': '#9CCFD8',
            'AR6': '#EE8679',
            'CGWL': '#A88BFA'
        }

        for ax in [ax1, ax3]:
            gr.gwi_timeseries(
                ax, df_temp_Obs, None, None, None,
                var_colours, sigmas=['5', '95', '50'])
            # Plot the centered 20-year rolling window on the 50th percentile Obs
            df_temp_Obs_20yr = df_temp_Obs.quantile(q=0.5, axis=1).rolling(
                window=20, center=True, axis=0).mean()
            ax.plot(df_temp_Obs_20yr.index, df_temp_Obs_20yr,
                    label='Obs 20-year running mean',
                    color='black'
                    )

        for headline in headlines:
            for vv in ['Tot', 'Ant']:
                # Plot the historical only timeseries
                ax1.plot(results_dfs[scen][reg_vars]['HISTORICAL-ONLY'][headline].index,
                         results_dfs[scen][reg_vars]['HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                         label=f'{headline}-{vv}',
                         linestyle=('solid' if vv == 'Tot' else 'dashed'),
                         color=headline_colours[headline]
                         )
                ax2.plot(
                    (results_dfs[scen][reg_vars]['HISTORICAL-ONLY'][headline].loc[:, (vv, '50')]
                     - df_temp_Obs_20yr),
                    label=f'{headline}-{vv}',
                    linestyle=('solid' if vv == 'Tot' else 'dashed'),
                    color=headline_colours[headline]
                )
                ax2.hlines(0, smallest_end_year, largest_end_year, color='black')

                # Calculate the full-information timeseries for the headlines

                if headline == 'ANNUAL':
                    # Use the full-information timeseries for the annual headline
                    df_fullinfo_defs = results_dfs[scen][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')]
                elif headline == 'AR6':
                    # Calculate rolling 10-year mean, lagged
                    df_fullinfo_defs = results_dfs[scen][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=10, center=False).mean()
                elif headline == 'SR15':
                    # Calcualte the rolling 30-year mean, centered
                    df_fullinfo_defs = results_dfs[scen][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=30, center=True).mean()
                elif headline == 'CGWL':
                    # Calculate the rolling 20-year mean, centered
                    df_fullinfo_defs = results_dfs[scen][reg_vars][max_regressed_range]['timeseries'].loc[:, (vv, '50')].rolling(window=20, center=True).mean()

                ax3.plot(df_fullinfo_defs.index, df_fullinfo_defs,
                         label=f'{headline}-{vv}',
                         linestyle=('solid' if vv == 'Tot' else 'dashed'),
                         color=headline_colours[headline]
                         )
                ax4.plot(
                    (df_fullinfo_defs - df_temp_Obs_20yr),
                    label=f'{headline}-{vv}',
                    linestyle=('solid' if vv == 'Tot' else 'dashed'),
                    color=headline_colours[headline]
                )
                ax4.hlines(0, smallest_end_year, largest_end_year, color='black')

        # Slice the df_temp_Obs using the smallest and largest end years
        min_y = np.floor(np.min(df_temp_Obs.loc[smallest_end_year:largest_end_year].values) * 2) / 2
        max_y = np.ceil(np.max(df_temp_Obs.loc[smallest_end_year:largest_end_year].values) * 2) / 2
        # min_y = np.floor(np.min(df_temp_Obs.values) * 2) / 2
        # max_y = np.ceil(np.max(df_temp_Obs.values) * 2) / 2
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
            f'Scenario: {scen} | Regressed variables: {reg_vars}'
        )
        fig.savefig(f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
                    'Historical_only_headlines_' +
                    f'{scen}_{reg_vars}_' +
                    f'{min_regressed_range}_to_{max_regressed_range}.png')

        plt.close(fig)

###############################################################################
# Generate the projected warming for final constrained year ###################
###############################################################################
        # TODO: Add to constrained warming dictionary.
        # TODO: Move calculation higher up in script.

        print('Creating constrained results for:', reg_vars)
        # Calculate how the expected final year of the timeseries changes
        # depending on the years that are regressed. Expect that the attributed
        # values in 2023 (end year of the full timeseries) will have larger
        # uncertainties, the earlier/shorter the range of regressed years is.

        # Create new empty dataframes to store the constrained results:
        # NOTE: you could also do this using maximum of the truncation range
        # if that's what you're interested in (possibly more relevant for
        # SSP projections in future)
        constrained_year = int(max_regressed_range.split('-')[1])

        df_constrained = results_dfs[scen][reg_vars][reg_ranges_all[0]]['timeseries'].copy()
        df_constrained[:] = 0

        # For each iteration, add the final row of the dataframe to the new
        # df_hist. The row index it should be inserted at is the same as the
        # second year in the iteration name.
        for reg_range in reg_ranges_all:
            # print(iteration, iteration.split('-')[1], constrained_year)
            df_constrained.loc[int(reg_range.split('-')[1])] = \
                results_dfs[scen][reg_vars][reg_range]['timeseries'].loc[constrained_year]

        # Remove all years that are not the end of an attribution period to
        # avoid confusion:
        df_constrained = df_constrained.loc[smallest_end_year:, :]

        #######################################################################
        # Plot this dataframe df_constrined in the same way as df_hist

        print('Plotting constrained results for:', reg_vars)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
        gr.gwi_timeseries(
            ax, None, None, df_constrained,
            plot_vars, var_colours, sigmas=['5', '95', '50'])

        ax.set_ylim(-1, np.ceil(np.max(df_constrained.loc[:, ('Tot', '50')].values) * 2) / 2)
        ax.set_xlim(smallest_end_year, largest_end_year)
        ax.set_ylabel(f'Warming in {constrained_year} ⁰C')
        ax.set_xlabel(f'Regressed years: {start_regress}-<year>')
        ax.set_xticks(xticks, xticks)
        gr.overall_legend(fig, 'lower center', 6)

        fig.suptitle(f'Projected warming in year {constrained_year}, ' +
                     'constrained by differing regressed years')
        fig.savefig(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
            f'Projected_warming_in_{constrained_year}_' +
            f'regressing_{reg_vars}_'
            'constrained_by_regressed_years_' +
            f'{min_regressed_range}_to_{max_regressed_range}.png')
        plt.close(fig)

###############################################################################
# Generate timeseries showing source of changes in GWI value each year ########
###############################################################################
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

        print('Creating delta contributions for:', scen, reg_vars)

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
                results_dfs[scen][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)] -
                results_dfs[scen][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)-1])
            # delta_rev is the change to the year Y from the previous to the
            # new dataset.
            delta_rev = (
                results_dfs[scen][reg_vars][f'{start_regress}-{y}']['timeseries'].loc[int(y)-1] -
                results_dfs[scen][reg_vars][f'{start_regress}-{int(y)-1}']['timeseries'].loc[int(y)-1])
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
        print('Plotting delta contributions for:', reg_vars)

        line_alpha = 0.9

        changing_var = 'Ant' if 'Ant' in plot_vars else 'Tot'
        # Plot the residual in the

        df_hist = results_dfs[scen][reg_vars]['HISTORICAL-ONLY']['ANNUAL']

        # Plot the 
        ax1.fill_between(
            df_delta_additional_forcing_year.index,
            # df_hist.loc[smallest_end_year:, ('Res', '5')].index,
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
            df_delta_additional_forcing_year.index,
            # df_hist.loc[smallest_end_year:, ('Res', '50')].index,
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
            df_new = results_dfs[scen][reg_vars][f'{start_regress}-{year}']['timeseries']
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
                df_old = results_dfs[scen][reg_vars][
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
        err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
                   df_temp_Obs.quantile(q=0.5, axis=1))
        err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
                   df_temp_Obs.quantile(q=0.05, axis=1))
        ax2.errorbar(df_temp_Obs.index, df_temp_Obs.quantile(q=0.5, axis=1),
                     yerr=(err_neg, err_pos),
                     fmt='o', color=var_colours['Obs'], ms=2.5, lw=1,
                     label='Reference Temp: HadCRUT5')

        ax2.set_ylabel('Global Warming, ⁰C')
        ax2.set_xticks(years, years)
        # ax2.set_xlim(2019.5, 2023.5)
        ax2.set_xlim(largest_end_year-4+0.5, largest_end_year+0.5)
        # ax2.set_ylim(
        #     np.floor(df_temp_Obs.loc[largest_end_year-4:, :].min().min() * 10) / 10,
        #     np.ceil(df_temp_Obs.loc[largest_end_year-4:, :].min().min() * 10) / 10
        #     # 1.1, 1.5
        #     )

        fig.suptitle(
            f'Contributions to the change in {changing_var} warming ' +
            'each year Y → Y+1')
        gr.overall_legend(fig, 'lower center', 3)
        fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
        fig.savefig(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
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

        print(f'Revision RMS for {reg_vars}: {delta_rms}')
        print(f'Residual RMS for {reg_vars}: {residual_rms}')
        print(f'Average fractional variation for {reg_vars}:',
              delta_rms / residual_rms)
