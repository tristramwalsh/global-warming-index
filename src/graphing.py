"""Plotting helpers for combine_results_iterations.py."""

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import src.variables as variables
import gc


def overall_legend(fig, loc, ncol, nrow=False, reorder=None):
    """Add a clean legend to a figure with multiple subplots."""
    handles, labels = [], []

    for ax in fig.axes:
        hs, ls = ax.get_legend_handles_labels()
        if hs not in handles:
            handles.extend(hs)
        if ls not in labels:
            labels.extend(ls)

    by_label = dict(zip(labels, handles))
    if reorder is None:
        order = np.arange(len(by_label))
    else:
        order = reorder

    fig.legend([list(by_label.values())[i] for i in order],
               [list(by_label.keys())[i] for i in order],
               loc=loc, ncol=ncol)

    ## rect = (left, bottom, right, top)
    # if loc == 'right':
    #     fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.94))
    # elif loc == 'lower center':
    #     # fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.94))
    #     fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))


def get_dynamic_colours(reg_vars, plot_vars, base_colours):
    """
    Generate dynamic colors for sub-variables based on their parent variables.

    Args:
        reg_vars (str): The regression variables string (e.g., 'GHG-Nat').
        plot_vars (list): List of variables to plot.
        base_colours (dict): Dictionary of base colors.

    Returns:
        tuple: (current_var_colours, scaling_map)
    """
    regress_vars_list = reg_vars.split('-')
    scaling_map = variables.map_var_to_regression_aggregate(
        plot_vars, regress_vars_list)

    current_var_colours = base_colours.copy()
    for var, parent in scaling_map.items():
        if parent in current_var_colours:
            current_var_colours[var] = current_var_colours[parent]

    return current_var_colours, scaling_map


def get_variable_linestyle(var):
    """
    Determine the linestyle for a variable based on its position in the hierarchy.
    Aggregate variables get 'solid'.
    Sub-variables get a unique linestyle based on their index among siblings.
    """
    # Aggregate variables are always solid
    if var in variables.SUB_VAR_MAPPING:
        return 'solid'

    # Define a list of linestyles to cycle through
    # Using tuples for more distinct patterns if needed, or standard strings
    linestyles = [
        '--',                          # Dashed
        ':',                           # Dotted
        '-.',                          # Dash-dot
        (0, (1, 1)),                   # Densely dotted
        (0, (5, 1)),                   # Densely dashed
        (0, (3, 1, 1, 1)),             # Densely dash-dotted
        (0, (3, 5, 1, 5)),             # Loosely dash-dotted
        (0, (5, 5)),                   # Loosely dashed
        (0, (1, 5)),                   # Loosely dotted
        (0, (3, 1, 1, 1, 1, 1))        # Dash-dot-dot
    ]

    for parent, children in variables.SUB_VAR_MAPPING.items():
        if var in children:
            idx = children.index(var)
            return linestyles[idx % len(linestyles)]

    # Fallback for unknown variables
    return 'solid'


def get_dynamic_linestyles(plot_vars):
    """
    Generate a dictionary of linestyles for the given plot variables.
    """
    linestyles = {}
    for var in plot_vars:
        linestyles[var] = get_variable_linestyle(var)
    return linestyles


def get_legend_reorder_indices(fig):
    """
    Determine the order of indices for the legend based on a predefined hierarchy.
    """
    # Collect labels in order of appearance
    all_labels = []
    for ax in fig.axes:
        _, labels = ax.get_legend_handles_labels()
        all_labels.extend(labels)

    # Get unique labels preserving order
    unique_labels = []
    seen = set()
    for lbl in all_labels:
        if lbl not in seen:
            unique_labels.append(lbl)
            seen.add(lbl)

    # Define hierarchy
    hierarchy = ['Reference Temp', 'Tot', 'Res']

    def add_subs(var):
        if var in variables.SUB_VAR_MAPPING:
            for sub in variables.SUB_VAR_MAPPING[var]:
                if sub not in hierarchy:
                    hierarchy.append(sub)
                add_subs(sub)
    add_subs('Tot')

    # Build desired order
    desired_order = []
    # 1. Items in hierarchy, in hierarchy order
    for item in hierarchy:
        if item in unique_labels:
            desired_order.append(item)

    # 2. Items in unique_labels not in hierarchy (append at end)
    for item in unique_labels:
        if item not in desired_order:
            desired_order.append(item)

    # Calculate indices
    indices = [unique_labels.index(item) for item in desired_order]

    return indices


def running_mean_internal_variability(
    timeframes, df_temp_PiC, temp_Obs_IV, gmt_id
):
    """Plot running means of internal variability ensemble.

    gmt_id: the reference temperature dataset, for the legend, e.g.
        'HadCRUT-2025'
    """
    lims = [0.6, 0.4, 0.15]
    for t in range(len(timeframes)):
        axA = plt.subplot2grid(
            shape=(len(timeframes), 4), loc=(t, 0), rowspan=1, colspan=3)
        axB = plt.subplot2grid(
            shape=(len(timeframes), 4), loc=(t, 3), rowspan=1, colspan=1)

        axA.set_ylim(-lims[t], lims[t])
        cut_beg = timeframes[t]//2
        cut_end = timeframes[t]-1-timeframes[t]//2
        _time = df_temp_PiC.index.to_numpy()

        if timeframes[t] == 1:
            _time_sliced = _time[:]
        else:
            _time_sliced = _time[cut_beg: -cut_end]

        for ens in df_temp_PiC.columns:
            _data = np.convolve(
                df_temp_PiC[ens],
                np.ones(timeframes[t]),
                'valid') / timeframes[t]
            # _data = moving_average(df_temp_PiC[ens], timeframes[t])
            axA.plot(_time_sliced, _data, label='CMIP6 PiControl',
                     color='gray', alpha=0.3)

            density = ss.gaussian_kde(_data)
            x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
            y = density(x)
            axB.plot(y, x, color='gray', alpha=0.3)

        # _data = moving_average(temp_Obs_IV, timeframes[t])
        _data = np.convolve(
            temp_Obs_IV,
            np.ones(timeframes[t]),
            'valid') / timeframes[t]
        axA.plot(_time_sliced, _data, label=f'{gmt_id} median',
                 color='xkcd:teal', alpha=1)
        density = ss.gaussian_kde(_data)
        x = np.linspace(axA.get_ylim()[0], axA.get_ylim()[1], 50)
        y = density(x)
        axB.plot(y, x, color='xkcd:teal', alpha=1)

        # axes[i].set_ylim([-0.6, +0.6])
        axA.set_xlim(1845, 2030)
        axB.set_ylim(axA.get_ylim())
        # axB.get_yaxis().set_visible(False)
        axB.get_xaxis().set_visible(False)
        axA.set_ylabel('Internal Variability (°C) \n'+
                       f'({timeframes[t]}-year moving mean)')


def plot_internal_variability_sample(
        ax1, ax2,
        df_temp_PiC, df_temp_Obs, temp_Obs_IV, sigmas, sigmas_all, gmt_id):
    """Plot internal variability: sigma timeseries and box plot.

    gmt_id: the reference temperature dataset, for the legend, e.g.
        'HadCRUT-2025'
    """
    for p in range(len(sigmas)):
        ax1.fill_between(
            df_temp_PiC.index,
            df_temp_PiC.quantile(q=sigmas_all[p]/100, axis=1),
            df_temp_PiC.quantile(q=sigmas_all[-(p+2)]/100, axis=1),
            color='gray', alpha=0.2)
        ax2.fill_between(
            x=[3, 4],
            y1=df_temp_PiC.quantile(
                q=sigmas_all[p]/100, axis=1).mean()*np.ones(2),
            y2=df_temp_PiC.quantile(
                q=sigmas_all[-(p+2)]/100, axis=1).mean()*np.ones(2),
            color='gray', alpha=0.2)

    ax1.plot(df_temp_PiC.index, df_temp_PiC.quantile(q=0.5, axis=1),
             color='gray', alpha=0.7, label='CMIP6 piControl')
    ax2.plot([3, 4], df_temp_PiC.quantile(q=0.5, axis=1).mean()*np.ones(2),
             color='gray', alpha=0.7)

    # Reference temperature median
    ax1.plot(df_temp_Obs.index,
             (np.percentile(temp_Obs_IV, sigmas_all) *
              np.ones((len(df_temp_Obs.index), len(sigmas_all)))),
             color='xkcd:teal', alpha=0.7, ls='--')
    ax1.plot(df_temp_Obs.index, temp_Obs_IV,
             color='xkcd:teal', alpha=0.7, label=f'{gmt_id} median')
    for p in sigmas:
        ax2.fill_between(x=[1, 2],
                         y1=np.percentile(temp_Obs_IV, p[0])*np.ones(2),
                         y2=np.percentile(temp_Obs_IV, p[1])*np.ones(2),
                         color='xkcd:teal', alpha=0.2)
    ax2.plot([1, 2], np.percentile(temp_Obs_IV, 50)*np.ones(2),
             color='xkcd:teal', alpha=0.7)

    ax1.set_ylabel('Internal Variability (°C)')
    ax1.set_ylim(-0.6, 0.6)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlim(0, 5)
    ax2.get_xaxis().set_visible(False)


def gwi_timeseries(ax, df_temp_Obs, df_temp_PiC, df_Results_ts,
                   plot_vars, plot_cols, sigmas='all', labels=True,
                   hatch=None, linestyle='solid', plume_vars=None,
                   ylabel=None):
    """Plot the GWI timeseries for the given variables."""

    if df_Results_ts is not None:
        all_vars = df_Results_ts.columns.get_level_values('variable').unique()
    else:
        all_vars = []
    
    if plume_vars is None:
        plume_vars = plot_vars

    if ylabel is None:
        ylabel = 'Attributable change in surface temperature since 1850\N{EN DASH}1900 (°C)'
    ax.set_ylabel(ylabel)
    fill_alpha = 0.25
    line_alpha = 0.7
    if sigmas == 'all':
        sigmas = df_Results_ts.columns.get_level_values('percentile').unique()
    # Shade the pre-industrial period
    ax.fill_between([1850, 1900], [-10, -10], [+10, +10],
                    color='#f4f2f1')
    ax.text(1875, -0.85, '1850\N{EN DASH}1900\nPreindustrial Baseline',
            ha='center')

    # Plot the observations
    if df_temp_Obs is not None:
        err_pos = (df_temp_Obs.quantile(q=0.95, axis=1) -
                   df_temp_Obs.quantile(q=0.5, axis=1))
        err_neg = (df_temp_Obs.quantile(q=0.5, axis=1) -
                   df_temp_Obs.quantile(q=0.05, axis=1))
        ax.errorbar(df_temp_Obs.index, df_temp_Obs.quantile(q=0.5, axis=1),
                    yerr=(err_neg, err_pos),
                    fmt='o', color=plot_cols['Obs'], ms=2.5, lw=1,
                    label=labels*'Reference Temp')
    # Plot the PiControl ensemble
    if df_temp_PiC is not None:
        if len(sigmas) > 1:
            for s in range(len(sigmas)//2):
                # Plot the PiControl ensemble
                ax.fill_between(
                    df_temp_PiC.index,
                    df_temp_PiC.quantile(q=float(sigmas[s])/100, axis=1),
                    df_temp_PiC.quantile(q=float(sigmas[-(s+2)])/100, axis=1),
                    color=plot_cols['PiC'], alpha=fill_alpha, linewidth=0.0)
        ax.plot(df_temp_PiC.index, df_temp_PiC.quantile(q=0.5, axis=1),
                color=plot_cols['PiC'], alpha=line_alpha,
                label=labels*'CMIP6 piControl')

    # Plot the GWI timeseries
    if df_Results_ts is not None:
        for s in range(max(len(sigmas)//2, 1)):  # max to enable 50% only
            # Plot the GWI timeseries
            for var in all_vars:

                # Determine linestyle
                if isinstance(linestyle, dict):
                    ls = linestyle.get(var, 'solid')
                else:
                    ls = linestyle

                # Because ROF (Gillett) method has different percentile results
                # available for different variables (ie Tot only has 50th), check
                # for each variable first whether to plot plume.
                var_sigmas = df_Results_ts.iloc[\
                    :, df_Results_ts.columns.get_level_values('variable') == var
                    ].columns.get_level_values('percentile').unique()
                if (len(var_sigmas) > 1) and var in plume_vars:
                    ax.fill_between(
                        df_Results_ts.index,
                        df_Results_ts.loc[:, (var, sigmas[s])].values,
                        df_Results_ts.loc[:, (var, sigmas[-(s+2)])].values,
                        color=plot_cols[var], alpha=fill_alpha, linewidth=0.0,
                        hatch=hatch, linestyle=ls)
                ax.plot(df_Results_ts.index,
                        df_Results_ts.loc[:, (var, sigmas[-1])].values,
                        color=plot_cols[var], alpha=line_alpha,
                        label=labels*var,
                        linestyle=ls)

    if df_Results_ts is not None:
        end_tick = df_Results_ts.index[-1]
    elif df_temp_Obs is not None:
        end_tick = df_temp_Obs.index[-1]

    ax.set_xticks([1850, 1900, 1950, 2000, end_tick],
                  [1850, 1900, 1950, 2000, end_tick])


def plot_spm2_panel(ax, df, period, vars_list, var_colours, var_names, ylim,
                    show_ylabel=False, show_yticklabels=False, xlim=None):
    """
    Helper to plot a single panel of bars.
    """

    for var in vars_list:
        
        med = df.loc[period, (var, '50')]
        neg = med - df.loc[period, (var, '5')]
        pos = df.loc[period, (var, '95')] - med

        ax.bar(vars_list.index(var), med, yerr=[[neg], [pos]],
               error_kw=dict(lw=0.8, capsize=2, capthick=0.8),
               color=var_colours[var], width=0.5)

    # X-axis labels
    labels = [var_names.get(var, var) for var in vars_list]
    
    ax.set_xticks(range(len(vars_list)))
    ax.set_xticklabels(labels, rotation=90)

    ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-0.5, len(vars_list) - 0.5)
        
    ax.axhline(0, color='black', linewidth=0.5)
    
    if show_ylabel:
        ax.set_ylabel('Change in global mean surface temperature'
                      '\nrelative to 1850\N{EN DASH}1900 (°C)')
    
    if not show_yticklabels:
        ax.set_yticklabels([])


def draw_grouping_arrow(ax, vars_list, target, sources, y_offsets=None, line_y_offset=0.26):
    """
    Draw arrows grouping source variables to a target variable.
    
    Args:
        ax: The axes to draw on.
        vars_list: List of variables in the plot (to find indices).
        target: The target variable name (e.g., 'Ant').
        sources: List of source variable names (e.g., ['GHG', 'OHF']).
        y_offsets: Dictionary of vertical offsets for each variable (from ax.y0).
        line_y_offset: Vertical offset for the connecting horizontal line (from ax.y0).
    """
    if y_offsets is None:
        y_offsets = {}

    # Extract variable names from vars_list (handling tuples)
    vars_names_only = [v[0] if isinstance(v, tuple) else v for v in vars_list]
    
    if target not in vars_names_only:
        return
    
    # Get bounds of ax
    bbox = ax.get_position()
    x0 = bbox.x0
    y0 = bbox.y0
    wi = bbox.width
    
    n_vars = len(vars_list)
    
    def get_x_fig(name):
        idx = vars_names_only.index(name)
        # Center of bar is at index idx.
        # Axis limits are -0.5 to n_vars - 0.5. Range is n_vars.
        # Fraction = (idx - (-0.5)) / n_vars = (idx + 0.5) / n_vars
        return x0 + wi * ((idx + 0.5) / n_vars)

    # Build lists for the loop
    xcoords = []
    ycoords = []
    arrow_styles = []
    
    # Target vertical line (Arrow pointing up to label)
    tx = get_x_fig(target)
    ty_start = y0 - y_offsets.get(target, 0.225)
    ty_line = y0 - line_y_offset
    
    xcoords.append([tx, tx])
    ycoords.append([ty_start, ty_line])
    arrow_styles.append('<-') # Arrow at start (top)
    
    # Source vertical lines
    source_xs = []
    for src in sources:
        if src in vars_names_only:
            sx = get_x_fig(src)
            source_xs.append(sx)
            sy_start = y0 - y_offsets.get(src, 0.225)
            sy_line = y0 - line_y_offset
            
            xcoords.append([sx, sx])
            ycoords.append([sy_start, sy_line])
            arrow_styles.append(']-') # Bracket at start (top)
            
    # Horizontal line connecting sources and target
    if source_xs:
        all_x = source_xs + [tx]
        min_x = min(all_x)
        max_x = max(all_x)
        
        xcoords.append([min_x, max_x])
        ycoords.append([ty_line, ty_line])
        arrow_styles.append('-')
    
    # Draw
    for i in range(len(xcoords)):
        ax.annotate('',
                    xy=(xcoords[i][1], ycoords[i][1]),
                    xytext=(xcoords[i][0], ycoords[i][0]),
                    xycoords='figure fraction',
                    textcoords='figure fraction',
                    arrowprops=dict(arrowstyle=arrow_styles[i],
                                    shrinkA=0, shrinkB=0,
                                    color='gainsboro',
                                    lw=1)
                    )


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


def add_configuration_text(fig, scen, ens, reg_vars, reg_range=None,
                           extra=None, x=0.5, y=0.01, **text_kwargs):
    """Write the line saying what a figure shows, by default centred at its
    foot in small monospace, e.g.
    'Inputs: ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900 |
    Ensemble: ERF-all_GMT-all | Regressed variables: GHG-Nat-OHF |
    Regressed range: 1850-2025'.

    reg_range: the regressed range, e.g. '1850-2025', if the figure has one
    extra: any further 'Name: value' part, e.g. 'Iterations: 3'
    x, y, text_kwargs: where and how to write it, passed to fig.text, e.g.
        x=0.05, y=0.93, ha='left', color='#555555'
    """
    # The parts every figure has
    parts = [f'Inputs: {scen}',
             f'Ensemble: {ens}',
             f'Regressed variables: {reg_vars}']

    # The parts only some figures have
    if reg_range is not None:
        parts.append(f'Regressed range: {reg_range}')
    if extra is not None:
        parts.append(extra)

    # Write it: centred at the foot unless told otherwise
    style = {'ha': 'center', 'fontsize': 'x-small', 'fontfamily': 'monospace'}
    style.update(text_kwargs)
    fig.text(x, y, ' | '.join(parts), **style)


def half_degree_limits(values):
    """Axis limits that just contain the values, rounded out to the nearest
    half: e.g. values from -0.3 to 1.32 give (-0.5, 1.5)."""
    return (np.floor(np.min(values) * 2) / 2,
            np.ceil(np.max(values) * 2) / 2)


def save_and_close_figure(fig, folder, file_name):
    """Save a figure as folder/file_name, making the folder if needed, then
    close it and free its memory (see close_figure). Returns the file path.

    folder: e.g. results_layout.PLOT_FOLDER + 'aggregated/<inputs>/...'
    file_name: e.g. 'Timeseries_<inputs>_ENSEMBLE-MEMBER--<ens>_....png'
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    fig.savefig(path)
    close_figure(fig)
    return path
