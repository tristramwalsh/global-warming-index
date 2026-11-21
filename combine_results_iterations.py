import os
import sys
import gc
import numpy as np
import matplotlib
# Draw figures straight to file: this script only saves them, never shows
# them. Without this, macOS picks its interactive backend, which is not safe
# to use in the forked worker processes that draw the figures.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from highlight_text import ax_text
from PIL import Image
import src.graphing as gr
import src.compute_resources as compute_resources
import src.results_layout as results_layout
import src.variables as variables
import src.warming_definitions as warming_definitions
import src.iteration_averaging as iteration_averaging
import src.results_reading as results_reading
import src.historical_only as historical_only
import src.figure_selection as figure_selection
import functools


# Results are filed by the inputs of the run that made them, in folders named
# ERF--<id>_GMT--<id>_PREINDUSTRIAL--<start>-<end>[_COMMITTED--<from>-<to>]
# (see gwi.py). Throughout this file, `scen` and `scenario` hold that whole
# folder name, e.g.
# 'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'.


def close_figure(fig):
    """Close a figure and free its memory straight away.

    plt.close only drops pyplot's own reference to the figure. The figure,
    its axes and their artists still refer to one another in cycles, which
    only Python's periodic cycle collection frees, so a process drawing many
    figures holds hundreds of megabytes of closed ones between collections.
    gc.collect frees them now.

    In the pipeline: used by the figure functions that run in a pool, where
    that memory is held once per worker.
    """
    plt.close(fig)
    gc.collect()


def figure_timeseries(reg_range, df_ts, scen, ens, reg_vars,
                      df_temp_Obs, params
                      ):
    """Plot the GWI timeseries of one regressed range, and save it.

    df_ts is the range's averaged timeseries, and df_temp_Obs the observed
    temperatures it is drawn against. Returns the saved figure's path.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range of a selection, and makes the selection's gif from
    the paths returned.

    Replaces a version that took the whole results_dfs and looked the
    range's timeseries up in it. A pool sends every argument to its worker,
    so each figure was sent every scenario, selection and range -- enough,
    for a member-by-member scenario, to run the job out of memory. It is now
    sent only the frame it draws.
    """
    # print('Creating single timeseries plots for:',
    #       scen, ens, reg_vars, reg_range, end='\r')

    # Get all variables present in the data
    all_data_vars = df_ts.columns.get_level_values(0).unique().to_list()

    # Define major variables (for plumes)
    major_vars = reg_vars.split('-')
    major_vars.extend(variables.extra_vars(major_vars))
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
    configuration = (f'Inputs: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars} | '
                     f'Regressed range: {reg_range}')
    if configuration:
        fig.text(0.5, 0.01, configuration, ha='center',
                 fontsize='x-small', fontfamily='monospace',
                 )

    plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    plot_name = (f'{plot_path}/' +
                 f'Timeseries_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    # plot_names.append(plot_name)
    fig.savefig(plot_name)
    close_figure(fig)
    return plot_name


def figure_rates(reg_range, df_rates, scen, ens, reg_vars,
                 df_temp_Obs, params
                 ):
    """Plot the warming rates of one regressed range, and save it.

    df_rates is the range's averaged rates, indexed by period (e.g.
    '1941-1950 (AR6 rate definition)'), and df_temp_Obs the observed
    temperatures. Re-indexes df_rates in place by each period's end year,
    which is safe because each pool worker has its own copy. Returns the
    saved figure's path.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range that has rates.

    Replaces a version that took the whole results_dfs; see
    figure_timeseries.
    """
    # Get all variables present in the data
    all_data_vars = df_rates.columns.get_level_values(0).unique().to_list()

    # Transform the index of df_rates to numeric - it is currently of the form
    # '1941-1950 (AR6 rate definition)' and we want that to be '1950' for
    # plotting.
    df_rates.index = df_rates.index.to_series().apply(
        lambda x: (int(x.split('-')[1].split()[0]) if '-' in x
                   else int(x.split()[0]))
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
    trunc_start = df_rates.index.min()
    trunc_end = df_rates.index.max()

    if not df_rates.loc[reg_end:, :].empty:
        gr.gwi_timeseries(
            ax, None, None,
            df_rates.loc[reg_end:, :],
            all_data_vars, params['colours'], hatch='x', linestyle='dashed',
            plume_vars=plume_vars, ylabel='Warming Rate')

    gr.gwi_timeseries(
        ax, None, None,
        df_rates.loc[reg_start:reg_end, :],
        all_data_vars, params['colours'], linestyle=var_linestyles,
        plume_vars=plume_vars, ylabel='Warming Rate')

    try:
        if 'Res' in df_rates.columns.get_level_values(0):
            df_for_ylim = df_rates.drop(columns='Res', level=0)
        else:
            df_for_ylim = df_rates
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
    configuration = (f'Inputs: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars} | '
                     f'Regressed range: {reg_range}')
    if configuration:
        fig.text(0.5, 0.01, configuration, ha='center',
                 fontsize='x-small', fontfamily='monospace',
                 )

    plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    plot_name = (f'{plot_path}/' +
                 f'Rates_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    fig.savefig(plot_name)
    close_figure(fig)
    return plot_name


def figure_iteration_comparison(reg_range, df_avg, scen, ens, reg_vars,
                                params
                                ):
    """Plot the spread between repeat iterations of the sampling.

    Each GWI run subsamples the full ensemble, so repeat iterations of the
    same configuration differ slightly. The left panel shows each iteration
    against the ensemble-size-weighted average of all of them; the right panel
    shows each iteration minus that average, which is the sampling variance.

    df_avg is the range's averaged timeseries; its iterations are read from
    ITERATIONS_FOLDER. Returns the saved figure's path, or None where the
    range has fewer than two iterations.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range, on the same tasks as figure_timeseries.

    Replaces a version that took the whole results_dfs; see
    figure_timeseries.
    """
    dict_iterations, _ = iteration_averaging.load_iteration_dfs(
        reg_range, 'timeseries', scen, ens, reg_vars)

    # The comparison is only meaningful where a configuration has been run
    # more than once.
    if len(dict_iterations) < 2:
        return None

    # Compare only the major variables, as including the sub-variables makes
    # the spread hard to read.
    major_vars = reg_vars.split('-')
    major_vars.extend(variables.extra_vars(major_vars))
    plot_vars = [v for v in major_vars
                 if v in df_avg.columns.get_level_values(0)]

    # Draw the median last, so that it sits on top of the outer percentiles
    # and provides the solid handle that overall_legend keeps for each label.
    percentiles = ['5', '95', '50']

    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), rowspan=1, colspan=1)

    for var in plot_vars:
        for pct in percentiles:
            # Emphasise the median over the 5th and 95th percentiles.
            alpha = 1 if pct == '50' else 0.5

            # Label every line with just its variable or 'Average', so that
            # the legend holds one entry per variable rather than one per
            # iteration; overall_legend removes the duplicates.
            for iteration in sorted(dict_iterations.keys()):
                df_iter = dict_iterations[iteration]
                ax1.plot(df_iter.index, df_iter[(var, pct)],
                         color=params['colours'][var], alpha=alpha,
                         linewidth=0.8, label=var)
                ax2.plot(df_iter.index,
                         df_iter[(var, pct)] - df_avg[(var, pct)],
                         color=params['colours'][var], alpha=alpha,
                         linewidth=0.8, label=var)

            ax1.plot(df_avg.index, df_avg[(var, pct)],
                     color='black', alpha=alpha,
                     linewidth=0.8, label='Average')

    ax2.axhline(0, color='darkslategray', linestyle='--', linewidth=0.8)

    ax1.set_ylabel('Iteration results, ⁰C')
    ax1.set_title('Individual iterations against their average\n'
                  '5th, 50th, 95th percentiles')
    ax2.set_ylabel('Iteration minus average, ⁰C')
    ax2.set_title('Difference between iterations and average\n'
                  '5th, 50th, 95th percentiles')

    for ax in (ax1, ax2):
        ax.set_xlim(df_avg.index.min(), df_avg.index.max())

    gr.overall_legend(fig, 'lower center', 7)

    # Add title
    fig.text(ax1.get_position().x0, ax1.get_position().y1+0.06,
             'Comparison of Repeat Sampling Iterations',
             ha='left',
             fontsize=plt.rcParams['axes.titlesize'],
             fontweight='bold'
             )

    # Add configuration text
    configuration = (f'Inputs: {scen} | '
                     f'Ensemble: {ens} | '
                     f'Regressed variables: {reg_vars} | '
                     f'Regressed range: {reg_range} | '
                     f'Iterations: {len(dict_iterations)}')
    fig.text(0.5, 0.01, configuration, ha='center',
             fontsize='x-small', fontfamily='monospace',
             )

    plot_path = (f'{results_layout.PLOT_FOLDER}iterations/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

    plot_name = (f'{plot_path}/' +
                 f'Compare-Iterations_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    fig.savefig(plot_name)
    close_figure(fig)
    return plot_name


def figure_spm2(
        reg_range, df_headlines, df_obs_headlines, scen, ens, reg_vars,
        params):
    """Plot the SPM2 bar plot for each headline period of one regressed range.

    df_headlines is the range's averaged headlines, one row per headline
    period. df_obs_headlines is its observed headlines, or None where none
    were written, in which case the figures are drawn without observations.
    Saves one figure per period, and returns nothing.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range that has headlines.

    Replaces a version that took the whole results_dfs and obs_dfs; see
    figure_timeseries.
    """

    periods = list(df_headlines.index)
    if not periods:
        return

    # Describe this run, so that an unexpectedly missing headline can be
    # traced back to the results that produced it.
    context = (f' (inputs {scen}, ensemble {ens}, variables {reg_vars}, '
               f'regressed years {reg_range})')

    # Determine variables for SPM2 panels 2 and 3.
    possible_vars_p2 = ['Tot', 'Ant', 'GHG', 'OHF', 'Nat', 'Res']
    for period in periods:
        # Panel 2: Aggregated. Selected per period rather than once for the
        # whole dataframe, because a variable can be available for some
        # periods and legitimately absent for others (specifically 'Res' on a
        # CGWL period; see headline_var_available).
        vars_panel2 = [
            v for v in possible_vars_p2
            if results_reading.headline_var_available(
                df_headlines, period, v, context)]

        # Panel 3: Components
        vars_panel3 = []
        if variables.SUB_VAR_MAPPING:
            for group in ['GHG', 'OHF', 'Nat']:
                if group in variables.SUB_VAR_MAPPING:
                    # Identify available variables in this group
                    group_vars = [
                        sub_var for sub_var in variables.SUB_VAR_MAPPING[group]
                        if results_reading.headline_var_available(
                            df_headlines, period, sub_var, context)]
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
                               params['colours'], variables.VAR_NAMES,
                               ylim, show_ylabel=True, show_yticklabels=True,
                               xlim=(-1.5, 1.5))

        # Panel 2: Aggregated
        gr.plot_spm2_panel(axes[1], df_headlines, period, vars_panel2,
                           params['colours'], variables.VAR_NAMES,
                           ylim, show_ylabel=False, show_yticklabels=False)

        # Panel 3: Components
        if vars_panel3 and len(axes) > 2:
            gr.plot_spm2_panel(axes[2], df_headlines, period, vars_panel3,
                               params['colours'], variables.VAR_NAMES,
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
        configuration = (f'Inputs: {scen} | '
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
        plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                     f'{scen}/' +
                     f'ENSEMBLE-MEMBER--{ens}/' +
                     f'VARIABLES--{reg_vars}/' +
                     f'REGRESSED-YEARS--{reg_range}/')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)

        period_token = str(period).replace(' ', '-')
        plot_name = (f'{plot_path}/' +
                     f'SPM2_BarPlot_{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        fig.savefig(plot_name)
        close_figure(fig)


def figure_waterfall(
        reg_range, df_headlines, df_obs_headlines, scen, ens, reg_vars,
        params):
    """Plot the waterfall bar plot for each headline period of one range.

    The waterfall is horizontal, with subtotals. df_headlines is the range's
    averaged headlines, one row per headline period. df_obs_headlines is its
    observed headlines, or None where none were written, in which case the
    figures are drawn without observations. Saves one figure per period, and
    returns nothing.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range that has headlines.

    Replaces a version that took the whole results_dfs and obs_dfs; see
    figure_timeseries.
    """

    periods = list(df_headlines.index)
    if not periods:
        return

    # Describe this run, so that an unexpectedly missing headline can be
    # traced back to the results that produced it.
    context = (f' (inputs {scen}, ensemble {ens}, variables {reg_vars}, '
               f'regressed years {reg_range})')

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
            # Filter and sort components. Availability is tested per period,
            # since 'Res' is legitimately absent on CGWL periods; any other
            # missing headline raises (see headline_var_available).
            vars_in_group = [
                v for v in source_vars
                if results_reading.headline_var_available(
                    df_headlines, period, v, context)]
            # Sort from largest to smallest warming contribution
            vars_in_group.sort(key=lambda v: get_stats(v)[0], reverse=True)

            for v in vars_in_group:
                plot_items.append({'var': v, 'type': 'component'})

        # Define the structure of the waterfall
        # GHG Group
        add_components(variables.SUB_VAR_MAPPING['GHG'])
        plot_items.append({'var': 'GHG', 'type': 'subtotal'})

        # OHF Group
        add_components(variables.SUB_VAR_MAPPING['OHF'])
        plot_items.append({'var': 'OHF', 'type': 'subtotal'})

        # Ant Total
        plot_items.append({'var': 'Ant', 'type': 'total'})

        # Nat Group
        add_components(variables.SUB_VAR_MAPPING['Nat'])
        plot_items.append({'var': 'Nat', 'type': 'subtotal'})

        # Tot Total
        plot_items.append({'var': 'Tot', 'type': 'total'})

        # Res (Components only)
        add_components(['Res'])

        # Obs Total. Only plotted where an observed headline actually exists:
        # it is absent when the observations do not cover this period (eg a
        # CGWL window reaching past the end of the record, which gwi.py
        # declines to compute), or when the run produced no observation
        # results at all. Without this check get_stats falls back to
        # (0, 0, 0), which draws a zero-height bar labelled "Observed
        # warming" rather than omitting it.
        if results_reading.headline_var_available(
                df_obs_headlines, period, 'Obs', context):
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
            label = variables.VAR_NAMES.get(var, var)
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
                ax.axhline(y=y_pos, color=params['colours'][var],
                           linewidth=1.5)

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
                        f"Sum of <{variables.VAR_NAMES['GHG']}> and "
                        f"<{variables.VAR_NAMES['OHF']}>"
                    )
                    highlight_textprops = [
                        {"color": params['colours']['GHG'],
                         "fontweight": "bold"},
                        {"color": params['colours']['OHF'],
                         "fontweight": "bold"}
                    ]
                elif var == 'Tot':
                    s = (
                        f"Sum of <{variables.VAR_NAMES['Ant']}> and "
                        f"<{variables.VAR_NAMES['Nat']}>"
                    )
                    highlight_textprops = [
                        {"color": params['colours']['Ant'],
                         "fontweight": "bold"},
                        {"color": params['colours']['Nat'],
                         "fontweight": "bold"}
                    ]
                elif var == 'Obs':
                    s = (
                        f"Sum of <{variables.VAR_NAMES['Tot']}> and "
                        f"<{variables.VAR_NAMES['Res']}>"
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
            'Change in global mean surface temperature relative to '
            '1850-1900 (°C)',
            fontsize=12)

        # Title
        fig.text(0.05, 0.95,
                 f'Attributable contributions to warming ({period})',
                 ha='left', fontsize=16, fontweight='bold')

        # Configuration text
        configuration = (f'Inputs: {scen} | '
                         f'Ensemble: {ens} | '
                         f'Regressed variables: {reg_vars} | '
                         f'Regressed range: {reg_range}')
        fig.text(0.05, 0.93, configuration, ha='left', fontsize=8,
                 fontfamily='monospace', color='#555555')

        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))

        # Save plot
        plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                     f'{scen}/' +
                     f'ENSEMBLE-MEMBER--{ens}/' +
                     f'VARIABLES--{reg_vars}/' +
                     f'REGRESSED-YEARS--{reg_range}/')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path, exist_ok=True)

        period_token = str(period).replace(' ', '-')
        plot_name = (f'{plot_path}/' +
                     f'Waterfall_BarPlot_{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        fig.savefig(plot_name)
        close_figure(fig)


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
                  if v in variables.SUB_VAR_MAPPING or v == 'Res']

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
        f'Inputs: {scen} | '
        f'Ensemble: {ens} | '
        f'Regressed variables: {reg_vars}')
    plot_path = (
        f'{results_layout.PLOT_FOLDER}priors/' +
        f'{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plot_name = (
        f'{plot_path}/' +
        f'Prior_Timeseries_{scen}_' +
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
                  if v in variables.SUB_VAR_MAPPING or v == 'Res']

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
        f'Inputs: {scen} | '
        f'Ensemble: {ens} | '
        f'Regressed variables: {reg_vars}')
    plot_path = (
        f'{results_layout.PLOT_FOLDER}erfs/' +
        f'{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    plot_name = (
        f'{plot_path}/' +
        f'ERF_Timeseries_{scen}_' +
        f'ENSEMBLE-MEMBER--{ens}_' +
        f'VARIABLES--{reg_vars}.png')
    fig.savefig(plot_name)
    plt.close(fig)


def figure_picontrol(scen, ens, obs_dfs):
    """Plot the piControl segments a run sampled internal variability from,
    against a rough estimate of the observed internal variability, and save
    the two figures: running means of the segments, and their distribution.

    scen: the inputs label, e.g.
        'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'
    ens: the member selection, e.g. 'ERF-all_GMT-all'
    obs_dfs: the loaded observations (see load_gwi_priors_erf_obs)

    In the pipeline: overarching_base_result_plotter calls it once for each
    selection it draws figures for. These figures used to be drawn by gwi.py
    (behind its plot_piControl switch).
    """
    # The filtered piControl segments, as gwi.py saved them
    df_temp_PiC = results_reading.read_picontrol_timeseries(scen)
    if df_temp_PiC is None:
        print('      Skipping piControl figures (no piControl file)')
        return

    # A rough estimate of the observed internal variability: the median of
    # the observations, minus its 30-year smoothed signal
    df_temp_Obs = obs_dfs[scen][ens]['timeseries']
    obs_median = df_temp_Obs.quantile(q=0.5, axis=1)
    temp_Obs_IV = obs_median - warming_definitions.temp_signal(
        obs_median.to_numpy(), 30, 'extrapolate')

    # Running-mean windows (years), and the likely and very likely ranges
    timeframes = [1, 3, 30]
    sigmas = [[17, 83], [5, 95]]
    sigmas_all = [5, 17, 83, 95, 50]
    gmt_id = results_layout.gmt_id_from_label(scen)

    # Where the figures go
    plot_path = (f'{results_layout.PLOT_FOLDER}picontrol/{scen}/'
                 f'ENSEMBLE-MEMBER--{ens}/')
    os.makedirs(plot_path, exist_ok=True)
    plot_label = f'{scen}_ENSEMBLE-MEMBER--{ens}'

    # 1. Running means of each segment, and their distributions
    fig = plt.figure(figsize=(15, 10))
    gr.running_mean_internal_variability(
        timeframes, df_temp_PiC, temp_Obs_IV, gmt_id)
    gr.overall_legend(fig, loc='lower center', ncol=2, nrow=False)
    fig.suptitle(
        'Selected Sample of Internal Variability from CMIP6 PiControl')
    fig.savefig(f'{plot_path}piControl_Selected_Segments_{plot_label}.png')
    close_figure(fig)

    # 2. The spread of the segments against the observed estimate
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
    ax2 = plt.subplot2grid(shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)
    gr.plot_internal_variability_sample(
        ax1, ax2, df_temp_PiC, df_temp_Obs, temp_Obs_IV,
        sigmas, sigmas_all, gmt_id)
    gr.overall_legend(fig, loc='lower center', ncol=3, nrow=False)
    fig.suptitle(
        'Selected Sample of Internal Variability from CMIP6 pi-control')
    fig.savefig(
        f'{plot_path}piControl_Internal_Variability_Distribution_'
        f'{plot_label}.png')
    close_figure(fig)


def parse_argvs():
    """Read the command line and return the settings it asks for.

    Options are given as --name=value, in any order. All of them are optional:

      --output-tag=<tag>   read results_<tag>/ and write plots_<tag>/ instead
                           of the plain results/ and plots/ folders. Use the
                           same tag that was given to gwi.py. Empty or absent
                           means the plain folders, as before.
      --re-calculate=y|n   whether to recompute the iteration averages, and
                           the historical-only datasets derived from them,
                           or reuse what is already in the aggregated folder.
                           Defaults to y.
      --plot-every=<N>     thin the figures of member-by-member selections
                           only -- the runs made with
                           --specify-ensemble-member that pull out one member
                           (e.g. ERF-all_GMT-7) and run the whole analysis
                           against it, as for Thorne et al. The standard
                           ERF-all_GMT-all run handles the members inside the
                           Monte Carlo instead and is ALWAYS plotted in full,
                           whatever N is.
                           For member-by-member selections: every one gets
                           its final regressed range; beyond that, only every
                           Nth member is plotted, and only at ranges ending
                           on every Nth year. The historical-only figures are
                           never thinned. Defaults to 10; pass 1 to draw
                           everything.

    Returns (output_tag, re_calculate, plot_every).
    """
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

    output_tag = argv_dict.get('--output-tag')
    # Default to re-calculating if not asked otherwise.
    re_calculate = argv_dict.get('--re-calculate', 'y') == 'y'
    plot_every = int(argv_dict.get('--plot-every',
                                   figure_selection.PLOT_EVERY))
    if plot_every < 1:
        sys.exit(f'--plot-every must be 1 or more, not {plot_every}')
    return output_tag, re_calculate, plot_every


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
        f'{results_layout.PLOT_FOLDER}aggregated/{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/' +
        f'Timeseries-animation_{scen}_' +
        f'Ensemble-Members--{ens}_' +
        f'Regressed--{reg_vars}_' +
        f'{min(reg_ranges_all)}_to_{max(reg_ranges_all)}.gif',
        save_all=True, append_images=images_list[1:],
        optimize=False, duration=500, loop=0)


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

        configuration = (f'Inputs: {scen} | '
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

    configuration = (f'Inputs: {scen} | '
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
                df_hist_headline = results_dfs[scen][ens][reg_vars][
                    'HISTORICAL-ONLY'][headline]
                resid = (df_hist_headline.loc[:, (vv, '50')]
                         - df_temp_Obs_20yr)
                ax2.plot(
                    resid,
                    label=f'{headline}-{vv}',
                    linestyle=ls,
                    color=gr.HEADLINE_COLOURS[headline]
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

    configuration = (f'Inputs: {scen} | '
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
    unwanted_vars = ['GHG', 'OHF', 'Res']
    # Filtered in the results' own column order: a set difference would come
    # back in an order that changes every time Python starts.
    plot_vars_main = [
        v for v in params['plot_vars'] if v not in unwanted_vars]

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

    configuration = (f'Inputs: {scen} | '
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
    ax1.set_xlabel(f'Regressed years: {start_regress}-<year>')
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

    configuration = (f'Inputs: {scen} | '
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
        label=f'Reference Temp: {results_layout.gmt_id_from_label(scen)}')

    ax2.set_ylabel('Global Warming, ⁰C')
    ax2.set_xticks(years, years)
    # ax2.set_xlim(2019.5, 2023.5)
    ax2.set_xlim(params['end_year']-4+0.5, params['end_year']+0.5)

    fig.suptitle(
        f'Contributions to the change in {changing_var} warming ' +
        'each year Y → Y+1')
    configuration = (f'Inputs: {scen} | '
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
    """Draw the base figures of every scenario, selection and variable set.

    For each, draws the GWI timeseries, rates, iteration comparison, SPM2 and
    waterfall figures for the regressed ranges ranges_to_plot chooses, and
    the gif, priors, ERF and piControl figures where there is something to
    draw. Saves
    them under PLOT_FOLDER, and returns nothing.

    In the pipeline: step 3 of the script, after the averaged results are
    loaded. Each per-range figure type runs in its own pool, in parallel
    over the regressed ranges, and each task carries only the frames its
    figure draws (tasks_ts, tasks_rates, tasks_headlines).

    Replaces a version that passed results_dfs and obs_dfs whole with every
    task. A pool sends every argument to its worker, so each figure was sent
    every scenario, selection and range -- which, for a member-by-member
    scenario, ran the job out of memory.
    """

    print('\nPlotting single-run timeseries')
    for scen in results_dfs.keys():
        print('INPUTS:', scen)
        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)

            # The piControl figures depend on the inputs and the members,
            # not on the regressed variables, so are drawn once here (for
            # the selections that get figures; see ensemble_is_plotted)
            if figure_selection.ensemble_is_plotted(
                    ens, figure_selection.PLOT_EVERY):
                print('    Plotting piControl figures')
                figure_picontrol(scen, ens, obs_dfs)

            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED_VARIABLES:', reg_vars)

                params = figure_selection.setup_plot_params(
                    scen, ens, reg_vars, results_dfs)
                reg_ranges_all = params['reg_ranges_all']

                # Check if all years are available
                steps = warming_definitions.check_steps(reg_ranges_all)
                if steps['check_bool']:
                    print('      All years available for: ', steps['range'])

                # The regressed ranges to draw the base figures below for,
                # chosen once here for all of them (see ranges_to_plot).
                # params keeps the full list, reg_ranges_all, because the
                # figures also look results up by params['max_range'] etc.
                reg_ranges_plot = figure_selection.ranges_to_plot(
                    ens, reg_ranges_all)
                if figure_selection.ensemble_uses_all_members(ens):
                    print('      Plotting all '
                          f'{len(reg_ranges_all)} regressed ranges '
                          '(full ensemble: --plot-every does not apply)')
                elif len(reg_ranges_plot) == len(reg_ranges_all):
                    print('      Plotting all '
                          f'{len(reg_ranges_all)} regressed ranges '
                          f'(--plot-every={figure_selection.PLOT_EVERY})')
                else:
                    ending = figure_selection.describe_ranges(reg_ranges_plot)
                    print(f'      Plotting {len(reg_ranges_plot)} of '
                          f'{len(reg_ranges_all)} regressed ranges '
                          f'(--plot-every={figure_selection.PLOT_EVERY}), '
                          f'ending {ending}')

                ###############################################################
                # 1. Plot GWI Timeseries
                valid_ranges_ts = [
                    r for r in reg_ranges_plot
                    if results_reading.is_dataset_present(
                        results_dfs[scen][ens][reg_vars][r], 'timeseries')
                    ]
                tasks_ts = [
                    (r, results_dfs[scen][ens][reg_vars][r]['timeseries'])
                    for r in valid_ranges_ts]
                if valid_ranges_ts:
                    ending = figure_selection.describe_ranges(valid_ranges_ts)
                    print('        Plotting figure_timeseries for GWI: '
                          '1 figure per regressed range, for '
                          f'{len(valid_ranges_ts)} regressed range/s '
                          f'ending {ending}; '
                          'in parallel on '
                          f'{compute_resources.n_workers()} worker/s')
                    plot_names = compute_resources.run_in_pool(
                        functools.partial(
                            figure_timeseries,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            df_temp_Obs=obs_dfs[scen][ens]['timeseries'],
                            params=params
                            ),
                        tasks_ts)

                valid_ranges_rates = [
                    r for r in reg_ranges_plot
                    if results_reading.is_dataset_present(
                        results_dfs[scen][ens][reg_vars][r], 'rates')
                    ]
                tasks_rates = [
                    (r, results_dfs[scen][ens][reg_vars][r]['rates'])
                    for r in valid_ranges_rates]
                if valid_ranges_rates:
                    ending = figure_selection.describe_ranges(
                        valid_ranges_rates)
                    print('        Plotting figure_rates for GWI: '
                          '1 figure per regressed range, for '
                          f'{len(valid_ranges_rates)} regressed range/s '
                          f'ending {ending}; '
                          'in parallel on '
                          f'{compute_resources.n_workers()} worker/s')
                    compute_resources.run_in_pool(
                        functools.partial(
                            figure_rates,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            df_temp_Obs=obs_dfs[scen][ens]['timeseries'],
                            params=params
                            ),
                        tasks_rates)

                ###########################################################
                # 2. Create GIF of Timeseries Plots

                # Add a toggle, because this is quite slow for the SMILE
                # ensembles (e.g. where we have an entirely different
                # set of results for a different ensemble member).
                # One frame is not an animation, which is what a member
                # selection thinned down to its final range produces.
                gif_toggle = True
                if gif_toggle and len(valid_ranges_ts) > 1:
                    figure_gif_animation(
                        plot_names, scen, ens, reg_vars, valid_ranges_ts)

                ###############################################################
                # 2b. Plot Comparison of Repeat Sampling Iterations

                # Add a toggle, because this is a diagnostic of the sampling
                # variance rather than a result. The figure is skipped
                # automatically where a configuration has only one iteration.
                # It draws each range's averaged timeseries against that
                # range's iterations, so it takes the same ranges and tasks
                # as the timeseries figures above.
                iteration_comparison_toggle = True
                if iteration_comparison_toggle and valid_ranges_ts:
                    ending = figure_selection.describe_ranges(valid_ranges_ts)
                    print('        Plotting figure_iteration_comparison: '
                          '1 figure per regressed range with more '
                          'than one iteration, for '
                          f'{len(valid_ranges_ts)} regressed range/s '
                          f'ending {ending}; '
                          'in parallel on '
                          f'{compute_resources.n_workers()} worker/s')
                    compute_resources.run_in_pool(
                        functools.partial(
                            figure_iteration_comparison,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            params=params
                            ),
                        tasks_ts)

                ###############################################################
                # 3. Plot Priors Timeseries
                if results_reading.priors_available(priors_dfs, scen, ens):
                    print('        Plotting figure_timeseries for PRIORS')
                    figure_priors_timeseries(
                        scen, ens, reg_vars, priors_dfs, obs_dfs,  params)
                else:
                    print('        Skipping PRIORS timeseries (no prior data)')

                ###############################################################
                # 3b. Plot ERF Timeseries
                print('        Plotting figure_timeseries for ERF')
                figure_erf_timeseries(
                    scen, ens, reg_vars, erf_dfs, params)

                ###############################################################
                # 4. Plot SPM2 Bar Plot
                valid_ranges_headlines = [
                    r for r in reg_ranges_plot
                    if results_reading.is_dataset_present(
                        results_dfs[scen][ens][reg_vars][r], 'headlines')
                    ]
                # Each range here has its own headlines -- that is what the
                # filter above checks -- but observed headlines are optional:
                # the loader only adds them where it found a file. So they
                # are looked up with .get, and a range without them gets
                # None and is drawn without observations.
                tasks_headlines = [
                    (r, results_dfs[scen][ens][reg_vars][r]['headlines'],
                     obs_dfs[scen][ens][r].get('headlines'))
                    for r in valid_ranges_headlines]

                # For the log: the SPM2 and waterfall figures each draw one
                # bar plot per headline period (row) of each range's
                # headlines, and these are the headline definitions they
                # belong to.
                headline_periods = [
                    period for r in valid_ranges_headlines
                    for period in results_dfs[scen][ens][reg_vars][r][
                        'headlines'].index
                    ]
                n_headline_figures = len(headline_periods)
                headline_names = list(dict.fromkeys(
                    historical_only.map_headline_to_index(
                        index_str=str(period), invert=True)
                    for period in headline_periods))
                if valid_ranges_headlines:
                    ending = figure_selection.describe_ranges(
                        valid_ranges_headlines)
                    print('        Plotting SPM2 for GWI: '
                          f'{n_headline_figures} figure/s, 1 per headline '
                          f'period ({", ".join(headline_names)}) of each '
                          'range, for '
                          f'{len(valid_ranges_headlines)} regressed range/s '
                          f'ending {ending}; in '
                          'parallel over the ranges on '
                          f'{compute_resources.n_workers()} worker/s')
                    compute_resources.run_in_pool(
                        functools.partial(
                            figure_spm2,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            params=params
                        ),
                        tasks_headlines)

                ###############################################################
                # 5. Plot Waterfall Plot
                if valid_ranges_headlines:
                    ending = figure_selection.describe_ranges(
                        valid_ranges_headlines)
                    print('        Plotting Waterfall for GWI: '
                          f'{n_headline_figures} figure/s, 1 per headline '
                          f'period ({", ".join(headline_names)}) of each '
                          'range, for '
                          f'{len(valid_ranges_headlines)} regressed range/s '
                          f'ending {ending}; in '
                          'parallel over the ranges on '
                          f'{compute_resources.n_workers()} worker/s')
                    compute_resources.run_in_pool(
                        functools.partial(
                            figure_waterfall,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            params=params
                        ),
                        tasks_headlines)


def overarching_historical_only_plotter(
    results_dfs,
    obs_dfs,
    priors_dfs
):
    """Plot figures of historical-only results."""

    print('\nGenerating historical-only timeseries')
    for scen in sorted(results_dfs.keys()):
        print('INPUTS:', scen)

        for ens in results_dfs[scen].keys():
            print('  ENSEMBLE-MEMBER:', ens)

            for reg_vars in sorted(results_dfs[scen][ens].keys()):
                print('    REGRESSED-VARIABLES:', reg_vars)

                if not results_reading.is_dataset_present(
                        results_dfs[scen][ens][reg_vars], 'HISTORICAL-ONLY'):
                    print('      No historical-only datasets available; '
                          'skipping historical-only plotting.')
                    continue

                plot_path = f'{results_layout.PLOT_FOLDER}aggregated/' + \
                    f'{scen}/' + \
                    f'ENSEMBLE-MEMBER--{ens}/' + \
                    f'VARIABLES--{reg_vars}/'
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)

                params = figure_selection.setup_plot_params(
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

    output_tag, re_calculate, plot_every = parse_argvs()
    results_layout.set_output_folders(output_tag)
    figure_selection.PLOT_EVERY = plot_every
    print(f'Re-calculate results: {re_calculate}')
    if figure_selection.PLOT_EVERY > 1:
        print(f'--plot-every={figure_selection.PLOT_EVERY}: member-by-member '
              'selections are thinned to one in '
              f'{figure_selection.PLOT_EVERY} members and ranges; '
              'ERF-all_GMT-all is always plotted in full')
    else:
        print('--plot-every=1: plotting every figure')

    # Ensure directoriesfor plots and results exist
    for folder in [results_layout.PLOT_FOLDER,
                   results_layout.AGGREGATED_FOLDER,
                   results_layout.ITERATIONS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # 1.Calculate gwi iteration averages
    if re_calculate:
        iteration_averaging.calculate_iteration_averages()

    # 2. Load results (gwi, priors, obs) into dataframes
    results_dfs, priors_dfs, erf_dfs, obs_dfs = (
        results_reading.load_gwi_priors_erf_obs())

    # 3. Plot the basic results
    overarching_base_result_plotter(
        results_dfs, obs_dfs, erf_dfs, priors_dfs)

    # 4. Generate historical-only timeseries and save to CSVs.
    if re_calculate:
        historical_only.calculate_historical_only(results_dfs)

    # 5. Load the historical-only data into dataframes
    results_dfs = results_reading.load_historical_only_dfs(results_dfs)

    # 6. Plot the historical-only results
    overarching_historical_only_plotter(
        results_dfs, obs_dfs, priors_dfs)
