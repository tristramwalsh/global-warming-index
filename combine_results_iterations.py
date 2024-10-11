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

if '--re-calculate' in argv_dict:
    re_calculate = argv_dict['--re-calculate']
    re_calculate = True if re_calculate == 'y' else False
else:
    re_calculate = input('Re-calculate? (y/n): ')
    re_calculate = True if re_calculate == 'y' else False


# AVERAGE THE TIMESERIES AND HEADLINES ITERATIONS #############################
def combine_repeats(result_type, regressed_years):
    dict_iterations = {}
    size_iterations = {}

    iterations = [f.split('_')[-1].split('.')[0]
                  for f in os.listdir(results_folder)
                  if result_type in f
                  and f"REGRESSED-YEARS--{regressed_years}" in f
                  and 'AVERAGE' not in f
                  and 'HISTORICAL-ONLY' not in f]
    # Remove previously averaged dataset in case it already exists
    iterations = sorted(list(set(iterations) - set(['AVERAGE'])))
    # print('iterations: ', iterations)

    for iteration in iterations:
        fname = [f for f in os.listdir(results_folder)
                 if result_type in f
                 and f"REGRESSED-YEARS--{regressed_years}" in f
                 and iteration in f][0]
        fname = f"{results_folder}/{fname}"
        # print(fname)
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
if re_calculate:
    regressed_years_all = sorted(list(set([
        f.split('_')[-2].split('--')[-1] for f in os.listdir(results_folder)
        if 'AVERAGE' not in f
        and 'HISTORICAL-ONLY' not in f])))

    for regressed_years in regressed_years_all:
        print(f'Regressed years: {regressed_years}')
        # Timeseries averaging
        df_avg, dict_iterations, size_iterations = combine_repeats(
            'timeseries', regressed_years)
        # Headlines may not have been calculated if only timeseries were needed
        if headline_toggle:
            combine_repeats('headlines', regressed_years)

        # Plot each iteration of the data to check
        fig = plt.figure(figsize=(15, 10))
        ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)
        for sigma in ['5', '95', '50']:
            for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
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
                      str(size_iterations.values()))

        # Plot difference between the data in each iteration and the average
        for iteration in dict_iterations.keys():
            for sigma in ['5', '95', '50']:
                for var in ['Ant', 'Nat', 'GHG', 'OHF', 'Tot']:
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
        fig.suptitle(f'Iterations for regressed years: {regressed_years}')
        fig.savefig(f'plots/Compare_iterations_{regressed_years}.png')


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

xticks = list(np.arange(smallest_end_year, largest_end_year + 1, 5))
xticks.append(largest_end_year)
ax.set_xticks(xticks, xticks)

ax.set_title(
    f'Regressed years range: {min_regressed_range} to {max_regressed_range}')
gr.overall_legend(fig, 'lower center', 6)


fig.suptitle('Historical-only (dashed) versus Full-information (solid)')

fig.savefig(
    'plots/Historical_vs_Full_timeseries_' +
    f'{min_regressed_range}_to_{max_regressed_range}.png')


# Calculate how the expected final year of the timeseries changes depending on
# the years that are regressed. Expect that the attributed values in 2023 (end
# year of the full timeseries) will have larger uncertainties, the
# earlier/shorter the range of regressed years is.

# Create new empty dataframes to store the constrained results:
constrained_year = end_yr
df_constrained = results_dfs['1850-2023'].copy()
df_constrained[:] = 0

# For each iterstion, add the final row of the dataframe to the new df_hist.
# The row index it should be inserted at is the same as the second year in
# the iteration name.
for iteration in results_dfs.keys():
    # print(iteration, iteration.split('-')[1], constrained_year)
    df_constrained.loc[int(iteration.split('-')[1])] = \
        results_dfs[iteration].loc[constrained_year]
# Remove all years that are not the end of an attribution period to avoid
# confusion.

df_constrained = df_constrained.loc[smallest_end_year:, :]
# Plot this dataframe df_constrined in the same way as df_hist
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
    f'plots/Projected_warming_in_{constrained_year}' +
    '_constrained_by_regressed_years' +
    f'_{min_regressed_range}_to_{max_regressed_range}.png')


# Generate timeseries that shows how each component of the GWI changes each
# year. From year Y to year Y+1, you have contributions from:
# 1. the change in temp in year Y in the old dataset to year Y in the new
# dataset
# 2. the change in temp in year Y+1 in the old dataset to the temp in year Y in
# the new dataset.
# 3. any changes in historical forcing in the new dataset
# 4. any changes in HadCRUT temperatures in the new dataset
# Only factor 1 and 2 are considered in this calcualtion. The other factors may
# be added later, but sourcing historical T and ERF data is significantly
# more wrangling.

# Create a new empty dataframe copied from before:
df_delta_additional_forcing_year = df_constrained.copy()
df_delta_revised_previous_year = df_constrained.copy()
df_delta_additional_forcing_year[:] = 0
df_delta_revised_previous_year[:] = 0

differ_years = sorted([r.split('-')[1] for r in list(results_files.keys())])
# switch the sorted order of the list years
differ_years = differ_years[::-1]
# remove the smallest year
differ_years = differ_years[:-1]

for y in differ_years:
    delta_new = (results_dfs[f'1850-{y}'].loc[int(y)] -
                 results_dfs[f'1850-{y}'].loc[int(y)-1])
    delta_rev = (results_dfs[f'1850-{y}'].loc[int(y)-1] -
                 results_dfs[f'1850-{int(y)-1}'].loc[int(y)-1])
    df_delta_revised_previous_year.loc[int(y)] = delta_rev
    df_delta_additional_forcing_year.loc[int(y)] = delta_new

# Plot the results
# The red line is the additional warming in year Y+1 relative to year Y in the
# new dataset.
# The blue line is the revised warming in year Y calculated in the year Y+1
# dataset relative to the year Y calculated in the year Y dataset.
# The green dashed line is the residual warming in year Y relative in the
# dataset for year Y. That is so say, this line comes from the historical-only
# dataset.


line_alpha = 0.9

fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)

ax1.fill_between(df_delta_additional_forcing_year.index,
                 df_hist.loc[smallest_end_year:, ('Res', '5')].values,
                 df_hist.loc[smallest_end_year:, ('Res', '95')].values,
                 color='seagreen', alpha=0.1, lw=0)
ax1.fill_between(df_delta_revised_previous_year.index,
                 df_delta_revised_previous_year.loc[:, ('Ant', '5')].values,
                 df_delta_revised_previous_year.loc[:, ('Ant', '95')].values,
                 color='steelblue', alpha=0.3, lw=0)
ax1.fill_between(df_delta_additional_forcing_year.index,
                 df_delta_additional_forcing_year.loc[:, ('Ant', '5')].values,
                 df_delta_additional_forcing_year.loc[:, ('Ant', '95')].values,
                 color='indianred', alpha=0.3, lw=0)
ax1.plot(df_delta_additional_forcing_year.index,
         df_hist.loc[smallest_end_year:, ('Res', '50')].values,
         label='Residual (internal variability) in year Y+1',
         color='seagreen', ls='dashed', alpha=line_alpha)
ax1.plot(df_delta_revised_previous_year.index,
         df_delta_revised_previous_year.loc[:, ('Ant', '50')].values,
         label='Revised warming in year Y',
         color='steelblue', alpha=line_alpha)
ax1.plot(df_delta_additional_forcing_year.index,
         df_delta_additional_forcing_year.loc[:, ('Ant', '50')].values,
         label='Additional warming from year Y to Y+1 in new dataset',
         color='indianred', alpha=line_alpha)

# Add a zero line for reference
ax1.axhline(0, color='black', linestyle='solid')
# xticks = list(np.arange(int(min(differ_years)), int(max(differ_years)) + 1, 5))
# xticks.append(int(max(differ_years)))
xticks.append(int(min(differ_years)))
ax1.set_xticks(xticks, xticks)
ax1.set_xlim(int(min(differ_years)), int(max(differ_years))+0.5)
ax1.set_ylim(-0.3, +0.3)
ax1.set_ylabel('Interannual warming delta, ⁰C')

# Plot schematic diagram
ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)
years = [2023, 2022, 2021, 2020]
for year in years:
    df_new = results_dfs[f'1850-{year}']
    ax2.plot(df_new.loc[:year, :].index,
             df_new.loc[:year, ('Ant', '50')],
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
                 [df_new.loc[year-1, ('Ant', '50')],
                  df_new.loc[year-1, ('Ant', '50')]],
                 color='indianred', linestyle='dashed', lw=2,
                 alpha=(year-min(years)+1)/len(years))
        ax2.plot([year, year],
                 [df_new.loc[year-1, ('Ant', '50')],
                  df_new.loc[year, ('Ant', '50')]],
                 color='indianred', linestyle='solid', lw=2,
                 alpha=(year-min(years)+1)/len(years))
        # Plot the blue line for the previous year's revision
        df_old = results_dfs[f'1850-{year-1}']

        ax2.plot([year-1, year],
                 [df_old.loc[year-1, ('Ant', '50')],
                  df_old.loc[year-1, ('Ant', '50')]],
                 color='steelblue', linestyle='dashed', lw=2,
                 alpha=(year-min(years)+1)/len(years))
        ax2.plot([year, year],
                 [df_new.loc[year-1, ('Ant', '50')],
                  df_old.loc[year-1, ('Ant', '50')]],
                 color='steelblue', linestyle='solid', lw=2,
                 alpha=(year-min(years)+1)/len(years))

ax2.plot(df_hist.index,
         df_hist.loc[:, ('Ant', '50')].values,
         label='Historical-only GWI',
         color='slateblue', ls='dashed', alpha=line_alpha, lw=1.5,
         marker='o', markeredgewidth=0)

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

fig.suptitle('Contributions to the change in Ant warming each year Y → Y+1')
gr.overall_legend(fig, 'lower center', 3)
fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
fig.savefig(
    'plots/Historical_delta_contributions_' +
    f'{min_regressed_range}_to_{max_regressed_range}.png')


# Find out how much the historical residual is as a fraction of the regression
# residual in that year. This is to say, how sensitive is the GWI to end
# effects of adding an additional year? If in year Y we have a very hot year,
# (the internal variability is taken to be the regression residual, since this
# is the amount of warming in the observations that is not accounted for by
# forced warming of any variety), how much does the 'Tot' warming change?
fractional_change = (
    df_delta_revised_previous_year.loc[:, ('Ant', '50')].values /
    df_hist.loc[smallest_end_year:, ('Res', '50')].values
    )
print('Revision / Residual:', fractional_change)
print('Average:', np.mean(fractional_change))
