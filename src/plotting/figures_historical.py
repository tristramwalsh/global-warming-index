"""The historical-only figures drawn by combine_results_iterations.py:
historical-only timeseries and headlines, 20-year running means (first
drawn for Thorne et al.), constrained warming, and the contributions to each
year's change.
"""
import numpy as np
import matplotlib.pyplot as plt

import src.plotting.graphing as gr
import src.plotting.plot_style as plot_style
import src.postprocessing.results_layout as results_layout
import src.method.variables as variables
import src.postprocessing.results_reading as results_reading


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
                      if v in variables.SUB_VAR_MAPPING or v == 'Res']

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
            gr.half_degree_limits(
                obs_dfs[scen][ens]['timeseries'].values)[1])
        ax.set_xlim(params['start_year'], params['end_year'])
        xticks = list(
            np.arange(params['start_year'], params['end_year'] + 1, 5))
        xticks.append(params['end_year'])
        ax.set_xticks(xticks, xticks)
        ax.set_title(
            'Regressed years range: ' +
            f"{params['min_range']} to {params['max_range']}".replace(
                '-', '\N{EN DASH}'))
        gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)

        if legend_loc == 'center right':
            plt.subplots_adjust(right=0.8)
        fig.suptitle(f'Historical-only {headline}')

        gr.add_configuration_text(fig, scen, ens, reg_vars)

        gr.save_and_close_figure(
            fig, plot_path,
            f'Historical_only_{headline}_' +
            f'{scen}_{ens}_{reg_vars}_' +
            f"{params['min_range']}_to_{params['max_range']}.png")


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
        -1, gr.half_degree_limits(obs_dfs[scen][ens]['timeseries'].values)[1])
    ax.set_xlim(params['start_year'], params['end_year'])
    xticks = list(np.arange(params['start_year'], params['end_year'] + 1, 5))
    xticks.append(params['end_year'])
    ax.set_xticks(xticks, xticks)

    ax.set_title(
        'Regressed years range: ' +
        f"{params['min_range']} to {params['max_range']}".replace(
            '-', '\N{EN DASH}'))
    gr.overall_legend(fig, 'lower center', 6)

    fig.suptitle(
        'Historical-only (dashed) versus Full-information (solid)')

    gr.add_configuration_text(fig, scen, ens, reg_vars)

    gr.save_and_close_figure(
        fig, plot_path,
        'ANNUAL_Historical_vs_Full_timeseries_' +
        f'{scen}_{ens}_{reg_vars}_' +
        f"{params['min_range']}_to_{params['max_range']}.png")


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

    # Residual statistics for the historical-only estimates against the
    # 20-year Obs running mean (i.e. the data plotted in the ax2 subplot).
    resid_stats = []

    # for headline in headlines:
    for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
        unwanted_vars = ['GHG', 'OHF', 'Res']
        # Filtered in the results' own column order: a set difference would
        # come back in an order that changes every time Python starts.
        plot_vars_main = [
            v for v in params['plot_vars'] if v not in unwanted_vars]
        for vv in plot_vars_main:
            # Determine line style
            ls = plot_style.HEADLINE_LINE_STYLE.get(vv)
            if ls is None:
                # Try to get style from parent
                parent = params['scaling_map'].get(vv)
                ls = plot_style.HEADLINE_LINE_STYLE.get(parent, 'solid')

            # Plot the historical only timeseries
            ax1.plot(results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].index,
                     results_dfs[scen][ens][reg_vars][
                         'HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                     label=f'{headline}-{vv}',
                     linestyle=ls,
                     color=plot_style.HEADLINE_COLOURS[headline]
                     )
            if vv != 'Nat':
                df_hist_headline = results_dfs[scen][ens][reg_vars][
                    'HISTORICAL-ONLY'][headline]
                resid = (df_hist_headline.loc[:, (vv, '50')]
                         - df_temp_Obs_20yr)
                ax2.plot(
                    resid,
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=plot_style.HEADLINE_COLOURS[headline]
                )
                # Score the residual against the headline's own uncertainty,
                # with sigma inferred from the 5-95 percentile range.
                resid = resid.dropna()
                sigma = (df_hist_headline.loc[resid.index, (vv, '95')]
                         - df_hist_headline.loc[resid.index, (vv, '5')]
                         ) / (2 * 1.645)
                loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2)
                                       + (resid / sigma) ** 2)
                resid_stats.append(
                    (headline, vv, len(resid), resid.mean(), resid.std(),
                     np.sqrt((resid ** 2).mean()), loglik))

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
                     color=plot_style.HEADLINE_COLOURS[headline]
                     )
            if vv != 'Nat':
                ax4.plot(
                    (df_fullinfo_defs - df_temp_Obs_20yr),
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=plot_style.HEADLINE_COLOURS[headline]
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
    min_y, max_y = gr.half_degree_limits(
        obs_dfs[scen][ens]['timeseries'].loc[
            params['start_year']:params['end_year']].values)
    min_y = min([-0.5, min_y])
    ax1.set_ylim(min_y, max_y)
    ax3.set_ylim(min_y, max_y)
    for ax in [ax1, ax2]:
        ax.set_xlim(params['start_year'], params['end_year'])
    gr.overall_legend(fig, 'lower center', 5)

    # Report the ax2 residual statistics against the 20-year Obs running mean
    print(f'        Historical-only minus 20-year Obs running mean | '
          f'{scen} | {ens} | {reg_vars}')
    print('          {:<8} {:<4} {:>4} {:>8} {:>8} {:>8} {:>10}'.format(
        'Headline', 'Var', 'N', 'bias', 'std', 'rmse', 'loglik'))
    for row in sorted(resid_stats):
        print('          {:<8} {:<4} {:>4d} {:>8.4f} {:>8.4f} {:>8.4f} '
              '{:>10.2f}'.format(*row))

    ax2.set_ylabel(r'$\Delta$ vs 20-year obs, ⁰C')
    ax4.set_ylabel(r'$\Delta$ vs 20-year obs, ⁰C')
    ax1.set_title('Historical-only')
    ax3.set_title('Full-information')
    fig.suptitle(
        'Historical-only and Full-information vs 20-year Obs running mean')

    gr.add_configuration_text(fig, scen, ens, reg_vars)

    gr.save_and_close_figure(
        fig, plot_path,
        'Historical_and_full_headlines_' +
        f'{scen}_{ens}_{reg_vars}_' +
        f"{params['min_range']}_to_{params['max_range']}.png")


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
    unwanted_vars = ['GHG', 'OHF', 'Res']
    # Filtered in the results' own column order: a set difference would come
    # back in an order that changes every time Python starts.
    plot_vars_main = [
        v for v in params['plot_vars'] if v not in unwanted_vars]

    for headline in results_dfs[scen][ens][reg_vars]['HISTORICAL-ONLY'].keys():
        for vv in plot_vars_main:
            # Determine line style
            ls = plot_style.HEADLINE_LINE_STYLE.get(vv)
            if ls is None:
                # Try to get style from parent
                parent = params['scaling_map'].get(vv)
                ls = plot_style.HEADLINE_LINE_STYLE.get(parent, 'solid')

            # Plot the historical only timeseries
            ax.plot(results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].index,
                    results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'][headline].loc[:, (vv, '50')],
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=plot_style.HEADLINE_COLOURS[headline]
                    )
    ax.plot(df_temp_Obs_20yr.index, df_temp_Obs_20yr,
            label='Obs 20-year running mean',
            color='black'
            )

    # Slice the df_temp_Obs using the smallest and largest end years
    min_y, max_y = gr.half_degree_limits(
        obs_dfs[scen][ens]['timeseries'].loc[
            params['start_year']:params['end_year']].values)
    min_y = min([-0.5, min_y])

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

    gr.add_configuration_text(fig, scen, ens, reg_vars)

    gr.save_and_close_figure(
        fig, plot_path,
        'Historical_only_headlines_' +
        f'{scen}_{ens}_{reg_vars}_' +
        f"{params['min_range']}_to_{params['max_range']}.png")


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

    has_priors = results_reading.priors_available(priors_dfs, scen, ens)

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

    # Variables to plot. These are taken from the priors dataset when it is
    # available (its column set is the reference used elsewhere in this
    # figure), otherwise from df_constrained itself, which carries the same
    # variables.
    if has_priors:
        plot_vars_priors = priors_dfs[scen][ens][
            'timeseries'].columns.get_level_values(0).unique().to_list()
    else:
        plot_vars_priors = df_constrained.columns.get_level_values(
            0).unique().to_list()

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
    ax1.set_xlabel(f'Regressed years: {start_regress}\N{EN DASH}<year>')
    xticks = list(np.arange(params['start_year'], params['end_year'] + 1, 5))
    xticks.append(params['end_year'])
    ax1.set_xticks(xticks, xticks)
    ax1.set_title(
        f'Constrained: {constrained_year} (with Obs only up to year <year>)')

    # Create box and whisker plot for prior warming in each variable.
    # Skipped entirely when the prior dataset was not calculated.
    bar_width = 0.4

    for vv in (plot_vars_priors if has_priors else []):
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

    gr.add_configuration_text(fig, scen, ens, reg_vars)

    gr.save_and_close_figure(
        fig, plot_path,
        f'Projected_warming_in_{constrained_year}_' +
        f'regressing_{reg_vars}_'
        'constrained_by_regressed_years_' +
        f"{params['min_range']}_to_{params['max_range']}.png")

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
                 label=f'{start_regress}\N{EN DASH}{year}',
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
        label=f'Reference Temp: {results_layout.gmt_id_from_label(scen)}')

    ax2.set_ylabel('Global Warming, ⁰C')
    ax2.set_xticks(years, years)
    # ax2.set_xlim(2019.5, 2023.5)
    ax2.set_xlim(params['end_year']-4+0.5, params['end_year']+0.5)

    fig.suptitle(
        f'Contributions to the change in {changing_var} warming ' +
        'each year Y → Y+1')
    gr.add_configuration_text(fig, scen, ens, reg_vars)
    gr.overall_legend(fig, 'lower center', 3)
    fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])

    gr.save_and_close_figure(
        fig, plot_path,
        'Historical_delta_contributions_' +
        f"{reg_vars}_{params['min_range']}_to_{params['max_range']}.png")

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
