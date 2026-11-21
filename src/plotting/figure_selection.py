"""Which figures combine_results_iterations.py draws: the plotting
parameters, and the thinning of member-by-member selections
(--plot-every).
"""
import src.plotting.graphing as gr
import src.plotting.plot_style as plot_style


# How far to thin the figures of member-by-member selections (ERF-all_GMT-7
# and the like). Never applies to ERF-all_GMT-all, which is always drawn in
# full. Overridden from the command line by --plot-every; see
# ranges_to_plot() for what it means and parse_argvs() for the option
# itself. 1 draws everything.
PLOT_EVERY = 10


def setup_plot_params(scen, ens, reg_vars, results_dfs):
    """Extract common plotting parameters."""
    reg_ranges_all = sorted(
        [item for item in results_dfs[scen][ens][reg_vars].keys()
            if item != 'HISTORICAL-ONLY']
        )
    min_regressed_range = reg_ranges_all[0]
    max_regressed_range = reg_ranges_all[-1]
    smallest_end_year = int(min_regressed_range.split('-')[1])
    largest_end_year = int(max_regressed_range.split('-')[1])

    first_range = reg_ranges_all[0]
    plot_vars = results_dfs[scen][ens][reg_vars][first_range][
        'timeseries'].columns.get_level_values(0).unique().to_list()
    all_var_colours, scaling_map = gr.get_dynamic_colours(
        reg_vars, plot_vars, plot_style.VAR_COLOURS)

    return {
        'min_range': min_regressed_range,
        'max_range': max_regressed_range,
        'start_year': smallest_end_year,
        'end_year': largest_end_year,
        'plot_vars': plot_vars,
        'colours': all_var_colours,
        'scaling_map': scaling_map,
        'reg_ranges_all': reg_ranges_all
    }


def ensemble_uses_all_members(ens):
    """Whether a selection uses every member of every source.

    True for ERF-all_GMT-all. False for any selection that names a member,
    such as ERF-all_GMT-7 or ERF-7_GMT-all.

    In the pipeline: ranges_to_plot and ensemble_is_plotted use it to exempt
    the full-ensemble selection from --plot-every, and
    overarching_base_result_plotter to say so in the log.
    """
    return set(
        combo.split('-')[1] for combo in ens.split('_')
        ) == {'all'}


def ensemble_is_plotted(ens, plot_every):
    """Whether a selection gets base figures beyond its final regressed range.

    True for the full-ensemble selection, and for a member-by-member
    selection whose member number is a multiple of plot_every -- GMT-10,
    GMT-20, ... for the default of 10. False for every other member.

    In the pipeline: ranges_to_plot calls it for each member-by-member
    selection.

    With ranges_to_plot, replaces toggle_single_timeseries, which picked the
    same one-in-ten members but applied the choice to the timeseries, rates
    and gif only. Every member still got the iteration comparison, SPM2 and
    waterfall figures for every regressed range.
    """
    if ensemble_uses_all_members(ens):
        return True
    ens_nums = [
        s for s in (combo.split('-')[1] for combo in ens.split('_'))
        if s.isdigit()
        ]
    return any(int(s) % plot_every == 0 for s in ens_nums)


def ranges_to_plot(ens, reg_ranges, plot_every=None):
    """Choose the regressed ranges a selection gets base figures for.

    Returns a sorted subset of reg_ranges:
      * all of them, for the full-ensemble selection (ERF-all_GMT-all), and
        for every selection when plot_every is 1;
      * the final range only, for a member-by-member selection whose member
        is not a multiple of plot_every (GMT-7, with the default of 10);
      * the final range and every range ending on a multiple of plot_every
        (1850-1950, 1850-1960, ...), for one whose member is (GMT-10).
    plot_every defaults to PLOT_EVERY, which --plot-every sets.

    Member-by-member selections are the runs that pull one member out of an
    ensemble and run the whole analysis against it, as for the Thorne et al.
    comparison against each GMST realisation. The standard run,
    ERF-all_GMT-all, samples the members inside the Monte Carlo instead. So
    with the default of 10, all hundred members of a member-by-member
    scenario get figures for their final range, and ten of them for
    1850-1950, 1850-1960, ... as well.

    In the pipeline: overarching_base_result_plotter calls it once for each
    selection and set of regressed variables, and draws every base figure --
    timeseries, rates, iteration comparison, SPM2 and waterfall -- for the
    ranges it returns only. The historical-only figures do not use it, and
    are drawn for every selection.

    Replaces toggle_single_timeseries (see ensemble_is_plotted). This is
    better because every base figure is thinned the same way, which takes a
    hundred-member scenario from about 61,600 figures to about 1,600, and
    because the final range -- the headline result, and the reason for
    running every member -- is kept for every member.
    """
    if plot_every is None:
        plot_every = PLOT_EVERY

    reg_ranges = sorted(reg_ranges)
    if not reg_ranges:
        return reg_ranges
    final_range = reg_ranges[-1]

    if plot_every <= 1 or ensemble_uses_all_members(ens):
        return reg_ranges

    if not ensemble_is_plotted(ens, plot_every):
        return [final_range]

    return [
        r for r in reg_ranges
        if r == final_range or int(r.split('-')[1]) % plot_every == 0
        ]


def describe_ranges(reg_ranges):
    """Summarise a list of regressed ranges by their end years, for the log.

    Returns the end years joined by commas -- '1950, 1960, 2025' -- or, for
    more than ten ranges, the first two and last two: '1950, 1951, ... 2024,
    2025'. Returns 'none' for an empty list. The ranges of one selection
    share a start year, so their end years are what tell them apart.

    In the pipeline: overarching_base_result_plotter uses it in the log line
    for each selection, which says which ranges it is drawing.
    """
    if not reg_ranges:
        return 'none'

    ends = [r.split('-')[1] for r in reg_ranges]
    if len(ends) <= 10:
        return ', '.join(ends)
    return f'{ends[0]}, {ends[1]}, ... {ends[-2]}, {ends[-1]}'
