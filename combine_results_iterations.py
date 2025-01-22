import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import src.graphing as gr
import src.definitions as defs
import json

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

    # Remove previously averaged dataset in case it already exists
    iterations_dates = [f.split('_DATE-CALCULATED--')[-1].split('.')[0]
                        for f in iteration_files]
    print(f'{regressed_vars} {regressed_years} iterations: {iterations_dates}')

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

        print('Alll regressed variables for scenario:',
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
                # Timeseries averaging
                df_avg, dict_iterations, size_iterations = combine_repeats(
                    'timeseries', scenario, regressed_years, regressed_vars)

                # Headlines may not have been calculated if only timeseries
                # were needed
                if headline_toggle:
                    combine_repeats(
                        'headlines', scenario, regressed_years, regressed_vars)

                # Get the variable names from the dataframe:
                vars_all = df_avg.columns.get_level_values(0).unique().to_list()

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
                    'Iterations for regressed years: '+
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
                regressed_years: (
                    f'{aggregated_folder}/SCENARIO--{scenario}/' +
                    f'VARIABLES--{regressed_vars}/' +
                    f'REGRESSED-YEARS--{regressed_years}/' +
                    'GWI_results_timeseries_' +
                    f'SCENARIO--{scenario}_'
                    f'VARIABLES--{regressed_vars}_' +
                    f'REGRESSED-YEARS--{regressed_years}_' +
                    'AVERAGE.csv'
                    )
                    })
# print(json.dumps(results_files_new, indent=4))

print('Loading all averaged datasets')
results_dfs = results_files.copy()
for reg_scen in results_files.keys():
    for reg_vars in results_files[reg_scen].keys():
        for reg_years in results_files[reg_scen][reg_vars].keys():
            df_ = pd.read_csv(
                    results_files[reg_scen][reg_vars][reg_years],
                    index_col=0,  header=[0, 1], skiprows=0)
            results_dfs[reg_scen][reg_vars][reg_years] = df_


###############################################################################
# Plot the timeseries for each iteration ######################################
###############################################################################
start_pi = 1850
end_pi = 1900

# TODO: Handle the next two lines in a scenario-general way
# start_yr = int(max(regressed_years_all).split('-')[0])
# end_yr = int(max(regressed_years_all).split('-')[1])
start_yr = 1850
end_yr = 2023

df_temp_Obs = defs.load_HadCRUT(start_pi, end_pi, start_yr, end_yr)

for scen in results_dfs.keys():
    for reg_vars in sorted(results_dfs[scen].keys()):
        reg_years_all = sorted(list(results_dfs[scen][reg_vars].keys()))
        plot_names = []

        for reg_years in reg_years_all:
            print('Creating timeseries plots for:',
                  scen, reg_vars, reg_years, end='\r')
            plot_vars = results_dfs[scen][reg_vars][reg_years
                ].columns.get_level_values(0).unique().to_list()

            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0),
                                  rowspan=1, colspan=1)

            reg_start = int(reg_years.split('-')[0])
            reg_end = int(reg_years.split('-')[1])

            gr.gwi_timeseries(
                ax, df_temp_Obs, None,
                results_dfs[scen][reg_vars][reg_years].loc[reg_end:, :],
                plot_vars, var_colours, hatch='x', linestyle='dashed')

            gr.gwi_timeseries(
                ax, df_temp_Obs, None,
                results_dfs[scen][reg_vars][reg_years].loc[reg_start:reg_end, :],
                plot_vars, var_colours)

            ax.set_ylim(-1, 2)
            ax.set_xlim(1850, 2023)
            gr.overall_legend(fig, 'lower center', 6)

            # Plot a box around the regressed years
            ax.axvline(int(reg_years.split('-')[1]),
                       color='darkslategray', linestyle='--')

            ax.set_ylim(-1, 2)
            ax.set_xlim(1850, 2023)

            fig.suptitle(f'Scenario--{scen} Regressed--{reg_vars}_{reg_years} Timeseries Plot')
            plot_path = f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/REGRESSED-YEARS--{reg_years}/'
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            fig.savefig(f'{plot_path}/Timeseries_Scenario--{scen}_Regressed--{reg_vars}_{reg_years}.png')
            plt.close(fig)
            plot_names.append(
                f'{plot_path}/Timeseries_Scenario--{scen}_Regressed--{reg_vars}_{reg_years}.png')
        print('')
        print('All years complete: ', reg_years_all)

        #######################################################################
        # Create a gif of the timeseries plots
        #######################################################################
        print('Creating gif of timeseries plots for:', reg_vars)

        images_list = [Image.open(plot) for plot in plot_names]
        # calculate the frame number of the last frame (ie the number of images)

        # # create 2 extra copies of the last frame (to make the gif spend longer on
        # # the most recent image)
        # for x in range(0, 2):
        #     images_list.append(images_list[-1])

        # Copy and revserse the list of images, so that the gif goes back and forth
        # between the first and last image.
        images_list += images_list[::-1]

        # save as a gif
        images_list[0].save(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
            f'Timeseries-animation_Scenario--{scen}_Regressed--{reg_vars}_' +
            f'{min(reg_years_all)}_to_{max(reg_years_all)}.gif',
            save_all=True, append_images=images_list[1:],
            optimize=False, duration=500, loop=0)

###############################################################################
# Generate the historical-only timeseries #####################################
###############################################################################
for scen in sorted(results_dfs.keys()):
    for reg_vars in sorted(results_dfs[scen].keys()):
        # Create a new empty dataframe to store the historical-only results:
        reg_years_all = sorted(list(results_dfs[scen][reg_vars].keys()))
        min_regressed_range = min(reg_years_all)
        max_regressed_range = max(reg_years_all)
        print(f'Creating historical-only timeseries for {reg_vars}: between ' +
              min_regressed_range + ' and ' + max_regressed_range)
        df_hist = results_dfs[scen][reg_vars][reg_years_all[0]].copy()
        df_hist[:] = 0

        # For each iteration, add the row that corresponds to the final year of
        # the regressed years to the new df_hist. The row indedx it should be
        # inserted at is the same as the second year in the iteration name.
        for reg_years in reg_years_all:
            df_hist.loc[int(reg_years.split('-')[1])] = \
                results_dfs[scen][reg_vars][reg_years].loc[
                    int(reg_years.split('-')[1])]
        # Remove all years that are not the end of an attribution period to
        # avoid confusion (i.e. the longer earlier years before the
        # historical-only focus period).
        end_years = [int(reg_years.split('-')[1])
                     for reg_years in reg_years_all]
        smallest_end_year = min(end_years)
        largest_end_year = max(end_years)

        df_hist = df_hist.loc[smallest_end_year:, :]
        results_dfs[scen][reg_vars]['HISTORICAL-ONLY'] = df_hist
        df_hist.to_csv(
            f'{aggregated_folder}/SCENARIO--{scen}/VARIABLES--{reg_vars}/'
            'GWI_results_timeseries_HISTORICAL-ONLY' +
            f'SCENARIO--{scen}_VARIABLES--{reg_vars}_' +
            f'__REGRESSED-YEARS--{min_regressed_range}' +
            f'_to_{max_regressed_range}.csv')

        #######################################################################
        # Plot the historical-only vs full dataset using gr.gwi_timeseries
        #######################################################################
        print('Plotting historical-only vs full dataset for:', scen, reg_vars)
        plot_vars = df_hist.columns.get_level_values(0).unique().to_list()

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)

        gr.gwi_timeseries(
            ax, df_temp_Obs, None,
            results_dfs[scen][reg_vars]['HISTORICAL-ONLY'],
            plot_vars, var_colours, sigmas=['5', '95', '50'],
            hatch='\\', linestyle='dashed')
        gr.gwi_timeseries(
            ax, df_temp_Obs, None,
            results_dfs[scen][reg_vars][max_regressed_range],
            plot_vars, var_colours, sigmas=['5', '95', '50'],
            hatch=None, linestyle='solid')

        ax.set_ylim(-1, 2)
        ax.set_xlim(smallest_end_year, largest_end_year)
        xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 5))
        xticks.append(largest_end_year)
        ax.set_xticks(xticks, xticks)

        ax.set_title(
            'Regressed years range: '+
            f'{min_regressed_range} to {max_regressed_range}')
        gr.overall_legend(fig, 'lower center', 6)

        fig.suptitle('Historical-only (dashed) versus Full-information (solid)\n' +
                    f'Scenario: {scen} | Regressed variables: {reg_vars}')
        fig.savefig(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
            'Historical_vs_Full_timeseries_' +
            f'{scen}_{reg_vars}_{min_regressed_range}_to_{max_regressed_range}.png')
        plt.close(fig)

        #######################################################################
        # Generate the projected warming in 2023 constrained by regressed years
        #######################################################################
        print('Creating constrained results for:', reg_vars)
        # Calculate how the expected final year of the timeseries changes
        # depending on the years that are regressed. Expect that the attributed
        # values in 2023 (end year of the full timeseries) will have larger
        # uncertainties, the earlier/shorter the range of regressed years is.

        # Create new empty dataframes to store the constrained results:
        constrained_year = end_yr
        df_constrained = results_dfs[scen][reg_vars][reg_years_all[0]].copy()
        df_constrained[:] = 0

        # For each iteration, add the final row of the dataframe to the new
        # df_hist. The row index it should be inserted at is the same as the
        # second year in the iteration name.
        for reg_years in reg_years_all:
            # print(iteration, iteration.split('-')[1], constrained_year)
            df_constrained.loc[int(reg_years.split('-')[1])] = \
                results_dfs[scen][reg_vars][reg_years].loc[constrained_year]

        # Remove all years that are not the end of an attribution period to
        # avoid confusion:
        df_constrained = df_constrained.loc[smallest_end_year:, :]
        
        #######################################################################
        # Plot this dataframe df_constrined in the same way as df_hist ########
        
        print('Plotting constrained results for:', reg_vars)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
        gr.gwi_timeseries(
            ax, None, None, df_constrained,
            plot_vars, var_colours, sigmas=['5', '95', '50'])

        ax.set_ylim(-1, 3)
        ax.set_xlim(smallest_end_year, largest_end_year)
        ax.set_ylabel('Warming in 2023 ⁰C')
        ax.set_xlabel('Regressed years: 1850-<year>')
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

        #######################################################################
        # Generate timeseries showing source of changes in GWI value each year.
        #######################################################################
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
        # Create a new empty dataframe copied from before:
        df_delta_additional_forcing_year = df_constrained.copy()
        df_delta_revised_previous_year = df_constrained.copy()
        df_delta_additional_forcing_year[:] = 0
        df_delta_revised_previous_year[:] = 0

        differ_years = sorted([r.split('-')[1] for r in reg_years_all])
        # switch the sorted order of the list years
        differ_years = differ_years[::-1]
        # remove the smallest year
        differ_years = differ_years[:-1]

        for y in differ_years:
            delta_new = (results_dfs[scen][reg_vars][f'1850-{y}'].loc[int(y)] -
                         results_dfs[scen][reg_vars][f'1850-{y}'].loc[int(y)-1])
            delta_rev = (results_dfs[scen][reg_vars][f'1850-{y}'].loc[int(y)-1] -
                         results_dfs[scen][reg_vars][f'1850-{int(y)-1}'].loc[int(y)-1])
            df_delta_revised_previous_year.loc[int(y)] = delta_rev
            df_delta_additional_forcing_year.loc[int(y)] = delta_new

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
        print(df_delta_revised_previous_year.head())
        print(df_hist.loc[smallest_end_year:, ('Res', '50')].head())
        line_alpha = 0.9

        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), rowspan=1, colspan=1)

        changing_var = 'Ant' if 'Ant' in plot_vars else 'Tot'
        ax1.fill_between(
            df_delta_additional_forcing_year.index,
            df_hist.loc[smallest_end_year:, ('Res', '5')].values,
            df_hist.loc[smallest_end_year:, ('Res', '95')].values,
            color='seagreen', alpha=0.1, lw=0)
        ax1.fill_between(
            df_delta_revised_previous_year.index,
            df_delta_revised_previous_year.loc[:, (changing_var, '5')].values,
            df_delta_revised_previous_year.loc[:, (changing_var, '95')].values,
            color='steelblue', alpha=0.3, lw=0)
        ax1.fill_between(
            df_delta_additional_forcing_year.index,
            df_delta_additional_forcing_year.loc[:, (changing_var, '5')].values,
            df_delta_additional_forcing_year.loc[:, (changing_var, '95')].values,
            color='indianred', alpha=0.3, lw=0)
        ax1.plot(
            df_delta_additional_forcing_year.index,
            df_hist.loc[smallest_end_year:, ('Res', '50')].values,
            label='Residual (internal variability) in year Y+1',
            color='seagreen', ls='dashed', alpha=line_alpha)
        ax1.plot(
            df_delta_revised_previous_year.index,
            df_delta_revised_previous_year.loc[:, (changing_var, '50')].values,
            label='Revised warming in year Y',
            color='steelblue', alpha=line_alpha)
        ax1.plot(
            df_delta_additional_forcing_year.index,
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

        ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=1, colspan=1)
        years = [2023, 2022, 2021, 2020]
        for year in years:
            df_new = results_dfs[scen][reg_vars][f'1850-{year}']
            ax2.plot(df_new.loc[:year, :].index,
                    df_new.loc[:year, (changing_var, '50')],
                    label=f'1850-{year}',
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
                df_old = results_dfs[scen][reg_vars][f'1850-{year-1}']

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
        ax2.set_xlim(2019.5, 2023.5)
        ax2.set_ylim(1.1, 1.5)

        fig.suptitle(f'Contributions to the change in {changing_var} warming '+
                    'each year Y → Y+1')
        gr.overall_legend(fig, 'lower center', 3)
        fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
        fig.savefig(
            f'plots/aggregated/SCENARIO--{scen}/VARIABLES--{reg_vars}/' +
            'Historical_delta_contributions_' +
            f'{reg_vars}_{min_regressed_range}_to_{max_regressed_range}.png')
        plt.close(fig)

        # Compare variation between internal variation (using Residual as a proxy
        # for this, because ideally speaking, all forced warming is accounted for,
        # so the remaining should largely be internal variability). Use RMS:
        delta_rms = np.sqrt(
            np.mean(df_delta_revised_previous_year.loc[:, (changing_var, '50')
                                                    ].values**2))
        residual_rms = np.sqrt(
            np.mean(df_hist.loc[smallest_end_year:, ('Res', '50')].values**2))

        print(f'Revision RMS for {reg_vars}: {delta_rms}')
        print(f'Residual RMS for {reg_vars}: {residual_rms}')
        print(f'Average fractional variation for {reg_vars}:',
              delta_rms / residual_rms)