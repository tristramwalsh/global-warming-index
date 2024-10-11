import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.graphing as gr
import src.definitions as defs

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

results_folder = 'results'

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


# AVERAGE THE TIMESERIES AND HEADLINES ITERATIONS #############################
def combine_repeats(result_type, regressed_years):
    dict_iterations = {}
    size_iterations = {}

    iterations = [f.split('_')[-1].split('.')[0]
                  for f in os.listdir(results_folder)
                  if result_type in f
                  and f"REGRESSED-YEARS--{regressed_years}" in f]
    # Remove previously averaged dataset in case it already exists
    iterations = sorted(list(set(iterations) - set(['AVERAGE'])))
    print('iterations: ', iterations)

    for iteration in iterations:
        fname = [f for f in os.listdir(results_folder)
                 if result_type in f
                 and f"REGRESSED-YEARS--{regressed_years}" in f
                 and iteration in f][0]
        fname = f"{results_folder}/{fname}"
        print(fname)
        ens_size = int(fname.split('ENSEMBLE-SIZE--')[-1].split('_')[0])
        df_iteration = pd.read_csv(
            fname, index_col=0,  header=[0, 1], skiprows=0)
        dict_iterations[iteration] = df_iteration
        size_iterations[iteration] = ens_size

    # Produce the averaged dataset
    df_avg = (dict_iterations[iterations[0]].copy() *
              size_iterations[iterations[0]])
    df_avg[:] = 0
    for iteration in iterations:
        df_avg += dict_iterations[iteration] * size_iterations[iteration]
    # df_avg /= len(iterations)
    df_avg /= sum(size_iterations.values())

    df_avg.to_csv(
        f'{results_folder}/GWI_results_{result_type}_' +
        f'REGRESSED-YEARS--{regressed_years}_' +
        'AVERAGE.csv')

    return df_avg, dict_iterations, size_iterations


# Loop through all regressed years that are available.
regressed_years_all = sorted(list(set([
    f.split('_')[-2].split('--')[-1] for f in os.listdir(results_folder)])))


# for regressed_years in regressed_years_all:
#     print(f'Regressed years: {regressed_years}')
#     # Timeseries averaging
#     df_avg, dict_iterations, size_iterations = combine_repeats(
#         'timeseries', regressed_years)
#     # Headlines may not have been calculated if only timeseries were needed
#     if headline_toggle:
#         combine_repeats('headlines', regressed_years)

#     # Plot each iteration of the data to check
#     fig = plt.figure(figsize=(15, 10))
#     ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)
#     ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)
#     for sigma in ['5', '95', '50']:
#         for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
#             for iteration in dict_iterations.keys():
#                 ax1.plot(df_avg.index,
#                          dict_iterations[iteration][(var, sigma)],
#                          label=f'{iteration} {var}',
#                          color=var_colours[var],
#                          alpha=1 if sigma == '50' else 0.5
#                          #  linestyle=linestyles[iteration]
#                          )
#             ax1.plot(df_avg.index, df_avg[(var, sigma)],
#                      label=f'Avg {var}',
#                      color='black',
#                      #  linestyle='--'
#                      )
#     # plt.legend()
#     ax1.set_ylabel(f'Iteration results, ⁰C')
#     ax1.set_title('Multiple iterations of samples,' +
#                   '5th, 50th, 95th percentiles\n' +
#                   str(size_iterations.values()))

#     # Plot the difference between the data in each iteration and the average
#     for iteration in dict_iterations.keys():
#         for sigma in ['5', '95', '50']:
#             for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
#                 ax2.plot(
#                     df_avg.index,
#                     dict_iterations[iteration][(var, sigma)] - df_avg[(var, sigma)],
#                     label=f'{iteration} {var}',
#                     color=var_colours[var],
#                     alpha=1 if sigma == '50' else 0.5
#                     #  linestyle=linestyles[iteration]
#                     )
#     # plt.legend()
#     ax2.set_ylabel(f'Iteration minus average, ⁰C')
#     ax2.set_title('Difference between iterations and average, ' +
#                   '5th, 50th, 95th percentiles')
#     fig.suptitle(f'Iterations for regressed years: {regressed_years}')
#     fig.savefig(f'plots/Compare_iterations_{regressed_years}.png')


# Generate the historical-only timeseries

# Get a list of all files with 'AVERAGE' in them:
results_files = {f.split('_')[-2].split('--')[-1]: f'{results_folder}/{f}'
                 for f in os.listdir(results_folder)
                 if 'AVERAGE' in f}
results_dfs = {}
for iteration in results_files.keys():
    df_iteration = pd.read_csv(
            results_files[iteration], index_col=0,  header=[0, 1], skiprows=0)
    results_dfs[iteration] = df_iteration
print(results_dfs.keys())
# Create a new empty dataframe to store the historical-only results:
df_hist = results_dfs['1850-2023'].copy()
df_hist[:] = 0

# For each iteration, add the row that corresponds to the final year of the
# regressed years to the new df_hist. The row indedx it should be inserted at
# is the same as the second year in the iteration name.
for iteration in results_dfs.keys():
    df_hist.loc[int(iteration.split('-')[1])] = \
        results_dfs[iteration].loc[int(iteration.split('-')[1])]
# Remove all years that are not the end of an attribution period to avoid
# confusion.
end_years = [int(iteration.split('-')[1]) for iteration in results_dfs.keys()]
smallest_end_year = min(end_years)
largest_end_year = max(end_years)

min_regressed_range = min(list(results_files.keys()))
max_regressed_range = max(list(results_files.keys()))

df_hist = df_hist.loc[smallest_end_year:, :]
df_hist.to_csv(
    f'{results_folder}/GWI_results_timeseries_HISTORICAL-ONLY' +
    f'__REGRESSED-YEARS--{min_regressed_range}_to_{max_regressed_range}.csv')

# Plot the dataset using gr.gwi_timeseries
start_pi = 1850
end_pi = 1900
start_yr = int(max_regressed_range.split('-')[0])
end_yr = int(max_regressed_range.split('-')[1])

df_temp_Obs = defs.load_HadCRUT(start_pi, end_pi, start_yr, end_yr)

plot_vars = ['Ant', 'GHG', 'Nat', 'OHF']
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0), rowspan=1, colspan=1)
gr.gwi_timeseries(
    ax, df_temp_Obs, None, df_hist,
    plot_vars, var_colours, sigmas=['5', '95', '50'],
    hatch='\\', linestyle='dashed')
gr.gwi_timeseries(
    ax, df_temp_Obs, None, results_dfs[max(list(results_files.keys()))],
    plot_vars, var_colours, sigmas=['5', '95', '50'],
    hatch=None, linestyle='solid')
ax.set_ylim(-1, 2)
ax.set_xlim(smallest_end_year, largest_end_year)
ax.text(1875, -0.85, f'{start_pi}\N{EN DASH}{end_pi}\nPreindustrial Baseline',
        ha='center')

xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 10))
xticks.append(largest_end_year)
print(xticks)
ax.set_xticks(xticks, xticks)

ax.set_title(f'Regressed years range: {min_regressed_range} to {max_regressed_range}')
gr.overall_legend(fig, 'lower center', 6)


fig.suptitle('Historical-only (dashed) versus Full-information (solid)')

fig.savefig(
    f'plots/Historical_vs_Full_timeseries_{min_regressed_range}_to_{max_regressed_range}.png')

