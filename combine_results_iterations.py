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

# Specify which ensemble size to average across
if '--ensemble-size' in argv_dict:
    ensemble_size = argv_dict['--ensemble-size']
else:
    ensemble_size = input('Sample size to average across (int): ')

# Specify which regressed years to average across
if '--regressed-years' in argv_dict:
    regressed_years = argv_dict['--regressed-years']
else:
    regressed_years = input('Regressed years to average across (y1-yn): ')


# AVERAGE THE TIMESERIES AND HEADLINES ITERATIONS #############################

dict_ts = {}
dict_hl = {}

iterations = [file.split('_')[-1].split('.')[0]
              for file in os.listdir(results_folder)
              if 'timeseries' in file
              and f"ENSEMBLE-SIZE--{ensemble_size}" in file
              and f"REGRESSED-YEARS--{regressed_years}" in file]
iterations = sorted(list(set(iterations) - set(['AVERAGE'])))
print('iterations: ', iterations)

for iteration in iterations:
    # TIMESERIES
    fname = (f"{results_folder}/GWI_results_timeseries_" +
             f"ENSEMBLE-SIZE--{ensemble_size}_" +
             f"REGRESSED-YEARS--{regressed_years}_" +
             f"{iteration}.csv")
    print(fname)
    skiprows = 0
    df_method_ts = pd.read_csv(
        fname, index_col=0,  header=[0, 1], skiprows=skiprows)
    dict_ts[iteration] = df_method_ts

    # HEADLINES
    fname = (f"{results_folder}/GWI_results_headlines_" +
             f"ENSEMBLE-SIZE--{ensemble_size}_" +
             f"REGRESSED-YEARS--{regressed_years}_" +
             f"{iteration}.csv")
    skiprows = 0
    df_method_hl = pd.read_csv(
        fname, index_col=0,  header=[0, 1], skiprows=skiprows)
    dict_hl[iteration] = df_method_hl

# Produce the averaged dataset
# TIMESERIES
df_ts_avg = dict_ts[iterations[0]].copy()
df_ts_avg[:] = 0
for iteration in iterations:
    df_ts_avg += dict_ts[iteration]
df_ts_avg /= len(iterations)

df_ts_avg.to_csv(
    f'{results_folder}/GWI_results_timeseries_' +
    f'ENSEMBLE-SIZE--{ensemble_size}_' +
    f'REGRESSED-YEARS--{regressed_years}_' +
    'AVERAGE.csv')

# HEADLINES
df_hl_avg = dict_hl[iterations[0]].copy()
df_hl_avg[:] = 0
for iteration in iterations:
    df_hl_avg += dict_hl[iteration]
df_hl_avg /= len(iterations)
df_hl_avg.to_csv(
    f'{results_folder}/GWI_results_headlines_' +
    f'ENSEMBLE-SIZE--{ensemble_size}_' +
    f'REGRESSED-YEARS--{regressed_years}_' +
    'AVERAGE.csv')


# PLOTTING #####################################################################
var_colours = {'Tot': '#d7827e',
               'Ant': '#b4637a',
               'GHG': '#907aa9',
               'Nat': '#56949f',
               'OHF': '#ea9d34',
               'Res': '#9893a5',
               'Obs': '#797593',
               'PiC': '#cecacd'}

# Plot each iteration of the data
plt.figure()
for sigma in ['5', '95', '50']:
    for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
        for iteration in iterations:
            plt.plot(df_ts_avg.index,
                     dict_ts[iteration][(var, sigma)],
                     label=f'{iteration} {var}',
                     color=var_colours[var],
                     alpha=1 if sigma == '50' else 0.5
                     #  linestyle=linestyles[iteration]
                     )
        plt.plot(df_ts_avg.index, df_ts_avg[(var, sigma)],
                 label=f'Avg {var}',
                 color='black',
                 #  linestyle='--'
                 )
# plt.legend()
plt.ylabel('2023 results, ⁰C')
plt.title('Multiple iterations of 6,048,000 samples, 50th percentiles')
plt.savefig(f'plots/Compare_iterations_{ensemble_size}.png')

# Plot the difference between the data in each iteration and the average
plt.figure()
for iteration in iterations:
    for sigma in ['5', '95', '50']:
        for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
            plt.plot(
                df_ts_avg.index,
                dict_ts[iteration][(var, sigma)] - df_ts_avg[(var, sigma)],
                label=f'{iteration} {var}',
                color=var_colours[var],
                alpha=1 if sigma == '50' else 0.5
                #  linestyle=linestyles[iteration]
                )
# plt.legend()
plt.ylabel('2023 results minus average, ⁰C')
plt.title('Difference between iterations and average, 50th percentiles')
plt.savefig(f'plots/Compare_iterations_diff_{ensemble_size}_REG_YRS--{regressed_years}.png')