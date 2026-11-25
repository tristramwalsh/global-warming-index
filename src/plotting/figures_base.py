"""The base figures drawn by combine_results_iterations.py for each set of
inputs, member selection and regressed range: timeseries, rates, iteration
comparison, SPM2 and waterfall bar plots, priors, ERFs, piControl and the
timeseries animation.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from highlight_text import ax_text

import src.plotting.graphing as gr
import src.postprocessing.results_layout as results_layout
import src.method.variables as variables
import src.method.warming_definitions as warming_definitions
import src.postprocessing.iteration_averaging as iteration_averaging
import src.postprocessing.results_reading as results_reading


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

    ax.set_ylim(*gr.half_degree_limits(
        df_ts.drop(columns='Res', level=0).values))
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
    gr.add_configuration_text(fig, scen, ens, reg_vars, reg_range)

    plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    file_name = (f'Timeseries_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    return gr.save_and_close_figure(fig, plot_path, file_name)


def figure_rates(reg_range, df_rates, scen, ens, reg_vars,
                 df_temp_Obs, params
                 ):
    """Plot the warming rates of one regressed range, and save it.

    df_rates is the range's averaged rates, indexed by each trend's last
    year, and df_temp_Obs the observed temperatures. Returns the saved
    figure's path.

    In the pipeline: overarching_base_result_plotter runs it in a pool, once
    per regressed range that has rates.

    Replaces a version that took the whole results_dfs; see
    figure_timeseries.
    """
    # Get all variables present in the data
    all_data_vars = df_rates.columns.get_level_values(0).unique().to_list()

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
    gr.add_configuration_text(fig, scen, ens, reg_vars, reg_range)

    plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    file_name = (f'Rates_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    return gr.save_and_close_figure(fig, plot_path, file_name)


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
        reg_range, 'timeseries-summary-ANNUAL', scen, ens, reg_vars)

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
    gr.add_configuration_text(
        fig, scen, ens, reg_vars, reg_range,
        extra=f'Iterations: {len(dict_iterations)}')

    plot_path = (f'{results_layout.PLOT_FOLDER}iterations/' +
                 f'{scen}/' +
                 f'ENSEMBLE-MEMBER--{ens}/' +
                 f'VARIABLES--{reg_vars}/' +
                 f'REGRESSED-YEARS--{reg_range}/')
    file_name = (f'Compare-Iterations_{scen}_' +
                 f'ENSEMBLE-MEMBER--{ens}_' +
                 f'VARIABLES--{reg_vars}_' +
                 f'REGRESSED-YEARS--{reg_range}.png')
    return gr.save_and_close_figure(fig, plot_path, file_name)


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
                 'Observed warming and contributions ('
                 + str(period).replace('-', '\N{EN DASH}') + ')',
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
        gr.add_configuration_text(fig, scen, ens, reg_vars, reg_range)

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
        period_token = str(period).replace(' ', '-')
        file_name = (f'SPM2_BarPlot_{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        gr.save_and_close_figure(fig, plot_path, file_name)


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
            '1850\N{EN DASH}1900 (°C)',
            fontsize=12)

        # Title
        fig.text(0.05, 0.95,
                 'Attributable contributions to warming ('
                 + str(period).replace('-', '\N{EN DASH}') + ')',
                 ha='left', fontsize=16, fontweight='bold')

        # Configuration text
        gr.add_configuration_text(fig, scen, ens, reg_vars, reg_range,
                                  x=0.05, y=0.93, ha='left', fontsize=8,
                                  color='#555555')

        fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))

        # Save plot
        plot_path = (f'{results_layout.PLOT_FOLDER}aggregated/' +
                     f'{scen}/' +
                     f'ENSEMBLE-MEMBER--{ens}/' +
                     f'VARIABLES--{reg_vars}/' +
                     f'REGRESSED-YEARS--{reg_range}/')
        period_token = str(period).replace(' ', '-')
        file_name = (f'Waterfall_BarPlot_{scen}_' +
                     f'ENSEMBLE-MEMBER--{ens}_' +
                     f'VARIABLES--{reg_vars}_' +
                     f'REGRESSED-YEARS--{reg_range}_' +
                     f'PERIOD--{period_token}.png')
        gr.save_and_close_figure(fig, plot_path, file_name)


def figure_plume_timeseries(df_ts, df_temp_Obs, scen, ens, reg_vars, params,
                            title, ylabel, start_year, folder, file_prefix):
    """Plot a timeseries of each variable's percentile plume, and save it:
    used for the prior warming and the ERFs.

    df_ts: the timeseries, columns (variable, percentile)
    df_temp_Obs: observations to draw too, or None
    title: e.g. 'Prior Warming Timeseries'
    ylabel: the y-axis label, or None for gr.gwi_timeseries's default
    start_year: the earliest year to show, e.g. 1850
    folder: the folder under PLOT_FOLDER, e.g. 'priors'
    file_prefix: the start of the file name, e.g. 'Prior_Timeseries'
    """
    plot_vars = df_ts.columns.get_level_values(0).unique().to_list()

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
        ax, df_temp_Obs, None, df_ts,
        plot_vars, params['colours'],
        hatch='x', linestyle=var_linestyles,
        plume_vars=plume_vars,
        ylabel=ylabel)

    if sub_vars:
        legend_loc = 'center right'
        legend_cols = 1
        reorder = gr.get_legend_reorder_indices(fig)
    else:
        legend_loc = 'lower center'
        legend_cols = 7
        reorder = None

    ax.set_ylim(*gr.half_degree_limits(df_ts.values))
    ax.set_xlim(max(start_year, df_ts.index.min()), df_ts.index.max())
    gr.overall_legend(fig, legend_loc, legend_cols, reorder=reorder)

    if legend_loc == 'center right':
        plt.subplots_adjust(right=0.8)

    # Title, above the axes (as in the GWI timeseries figure)
    fig.text(ax.get_position().x0, ax.get_position().y1+0.02,
             title,
             ha='left',
             fontsize=plt.rcParams['axes.titlesize'],
             fontweight='bold'
             )

    # Add configuration text
    gr.add_configuration_text(fig, scen, ens, reg_vars)

    plot_path = (
        f'{results_layout.PLOT_FOLDER}{folder}/' +
        f'{scen}/' +
        f'ENSEMBLE-MEMBER--{ens}/' +
        f'VARIABLES--{reg_vars}/')
    file_name = (
        f'{file_prefix}_{scen}_' +
        f'ENSEMBLE-MEMBER--{ens}_' +
        f'VARIABLES--{reg_vars}.png')
    gr.save_and_close_figure(fig, plot_path, file_name)


def figure_priors_timeseries(
        scen, ens, reg_vars,
        priors_dfs, obs_dfs,
        params
):
    """Plot timeseries for PRIOR warming."""
    figure_plume_timeseries(
        priors_dfs[scen][ens][reg_vars]['timeseries'],
        obs_dfs[scen][ens]['timeseries'],
        scen, ens, reg_vars, params,
        title='Prior Warming Timeseries', ylabel=None, start_year=1850,
        folder='priors', file_prefix='Prior_Timeseries')


def figure_erf_timeseries(
        scen, ens, reg_vars,
        erf_dfs, params
):
    """Plot timeseries for ERF."""
    figure_plume_timeseries(
        erf_dfs[scen][ens][reg_vars]['timeseries'], None,
        scen, ens, reg_vars, params,
        title='Effective Radiative Forcing Timeseries',
        ylabel='Effective Radiative Forcing (W m⁻²)', start_year=1750,
        folder='erfs', file_prefix='ERF_Timeseries')


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
    plot_label = f'{scen}_ENSEMBLE-MEMBER--{ens}'

    # 1. Running means of each segment, and their distributions
    fig = plt.figure(figsize=(15, 10))
    gr.running_mean_internal_variability(
        timeframes, df_temp_PiC, temp_Obs_IV, gmt_id)
    gr.overall_legend(fig, loc='lower center', ncol=2, nrow=False)
    fig.suptitle(
        'Selected Sample of Internal Variability from CMIP6 PiControl')
    gr.save_and_close_figure(
        fig, plot_path, f'piControl_Selected_Segments_{plot_label}.png')

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
    gr.save_and_close_figure(
        fig, plot_path,
        f'piControl_Internal_Variability_Distribution_{plot_label}.png')


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
