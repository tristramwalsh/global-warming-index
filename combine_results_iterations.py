import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import src.graphing as gr

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

    # Plot each iteration of the data
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)
    for sigma in ['5', '95', '50']:
        for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
            for iteration in iterations:
                ax1.plot(df_avg.index,
                         dict_iterations[iteration][(var, sigma)],
                         label=f'{iteration} {var}',
                         color=var_colours[var],
                         alpha=1 if sigma == '50' else 0.5
                         #  linestyle=linestyles[iteration]
                         )
            ax1.plot(df_avg.index, df_avg[(var, sigma)],
                     label=f'Avg {var}',
                     color='black',
                     #  linestyle='--'
                     )
    # plt.legend()
    ax1.set_ylabel(f'Iteration results, ⁰C')
    ax1.set_title('Multiple iterations of samples,' +
                  '5th, 50th, 95th percentiles\n' +
                  str(size_iterations.values()))

    # Plot the difference between the data in each iteration and the average
    for iteration in iterations:
        for sigma in ['5', '95', '50']:
            for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
                ax2.plot(
                    df_avg.index,
                    dict_iterations[iteration][(var, sigma)] - df_avg[(var, sigma)],
                    label=f'{iteration} {var}',
                    color=var_colours[var],
                    alpha=1 if sigma == '50' else 0.5
                    #  linestyle=linestyles[iteration]
                    )
    # plt.legend()
    ax2.set_ylabel(f'Iteration minus average, ⁰C')
    ax2.set_title('Difference between iterations and average, ' +
                  '5th, 50th, 95th percentiles')
    fig.suptitle(f'Iterations for regressed years: {regressed_years}')
    fig.savefig(f'plots/Compare_iterations_{regressed_years}.png')


# Loop through all regressed years that are available.
regressed_years_all = sorted(list(set([
    f.split('_')[-2].split('--')[-1] for f in os.listdir(results_folder)])))


for regressed_years in regressed_years_all:
    print(f'Regressed years: {regressed_years}')
    # Timeseries averaging
    combine_repeats('timeseries', regressed_years)
    # Headlines may not have been calculated if only timeseries were needed
    if headline_toggle:
        combine_repeats('headlines', regressed_years)

