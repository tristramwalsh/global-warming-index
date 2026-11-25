import os
import subprocess
import sys
import matplotlib
# Draw figures straight to file: this script only saves them, never shows
# them. Without this, macOS picks its interactive backend, which is not safe
# to use in the forked worker processes that draw the figures. It must come
# before the figure modules below, which import matplotlib.pyplot.
matplotlib.use('Agg')
import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.method.warming_definitions as warming_definitions
import src.postprocessing.iteration_averaging as iteration_averaging
import src.postprocessing.results_reading as results_reading
import src.postprocessing.historical_only as historical_only
import src.plotting.figure_selection as figure_selection
import src.plotting.figures_base as figures_base
import src.plotting.figures_historical as figures_historical
import src.plotting.plot_style as plot_style
import functools


# Results are filed by the inputs of the run that made them, in folders named
# ERF--<id>_GMT--<id>_PREINDUSTRIAL--<start>-<end>[_COMMITTED--<from>-<to>]
# (see gwi.py). Throughout this file, `scen` and `scenario` hold that whole
# folder name, e.g.
# 'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'.


def read_recipe_output_tag(recipe):
    """Return the RECIPE_OUTPUT_TAG that a recipe file sets.

    The recipe is sourced by bash, as run-gwi.sh does, so the value is read
    exactly as the runs saw it, quoting and all. An empty tag returns None,
    meaning results/untagged/ and plots/untagged/.
    """
    if not os.path.isfile(recipe):
        sys.exit(f'Recipe not found: {recipe}')
    result = subprocess.run(
        ['bash', '-c', 'source "$1" && printf %s "${RECIPE_OUTPUT_TAG-}"',
         '_', recipe],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'Could not read the recipe {recipe}:\n{result.stderr}')
    return result.stdout or None


def parse_argvs():
    """Read the command line and return the settings it asks for.

    Options are given as --name=value, in any order. All of them are optional:

      --output-tag=<tag>   read results/<tag>/ and write plots/<tag>/. Use
                           the same tag that was given to gwi.py. Empty or
                           absent means results/untagged/ and
                           plots/untagged/.
      --recipe=<file>      an alternative to --output-tag: take the tag
                           from a recipe instead, e.g.
                           --recipe=recipes/TEMPLATE.sh reads its
                           RECIPE_OUTPUT_TAG. Recipes in one experiment
                           share a tag, so any of them will do. Give this
                           or --output-tag, not both.
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
    if '--recipe' in argv_dict:
        if output_tag is not None:
            sys.exit('Give --recipe or --output-tag, not both.')
        output_tag = read_recipe_output_tag(argv_dict['--recipe'])
    # Default to re-calculating if not asked otherwise.
    re_calculate = argv_dict.get('--re-calculate', 'y') == 'y'
    plot_every = int(argv_dict.get('--plot-every',
                                   figure_selection.PLOT_EVERY))
    if plot_every < 1:
        sys.exit(f'--plot-every must be 1 or more, not {plot_every}')
    return output_tag, re_calculate, plot_every


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
                figures_base.figure_picontrol(scen, ens, obs_dfs)

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
                            figures_base.figure_timeseries,
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
                            figures_base.figure_rates,
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
                    figures_base.figure_gif_animation(
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
                            figures_base.figure_iteration_comparison,
                            scen=scen, ens=ens, reg_vars=reg_vars,
                            params=params
                            ),
                        tasks_ts)

                ###############################################################
                # 3. Plot Priors Timeseries
                if results_reading.priors_available(priors_dfs, scen, ens,
                                                    reg_vars):
                    print('        Plotting figure_timeseries for PRIORS')
                    figures_base.figure_priors_timeseries(
                        scen, ens, reg_vars, priors_dfs, obs_dfs,  params)
                else:
                    print('        Skipping PRIORS timeseries (no prior data)')

                ###############################################################
                # 3b. Plot ERF Timeseries
                print('        Plotting figure_timeseries for ERF')
                figures_base.figure_erf_timeseries(
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
                            figures_base.figure_spm2,
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
                            figures_base.figure_waterfall,
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
                figures_historical.figure_historical_only_timeseries(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 2. Plot historical-only vs full dataset timeseries
                figures_historical.figure_historical_vs_full_timeseries(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs, priors_dfs,
                    plot_path, params
                )

                # 3. Plot comparison of all headlines datasets
                figures_historical.figure_headlines_comparison(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 4. Create one-off figure for Thorne et al paper
                figures_historical.figure_thorne_et_al(
                    scen, ens, reg_vars,
                    results_dfs, obs_dfs,
                    plot_path, params
                )

                # 5. Generate the projected warming for final constrained year
                df_constrained = figures_historical.figure_constrained_warming(
                    scen, ens, reg_vars,
                    results_dfs, priors_dfs,
                    plot_path, params
                )

                # 6. Generate timeseries showing source of changes in GWI value
                # each year
                figures_historical.figure_delta_contributions(
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
    # Apply the figure style (fonts, axes, grid) before drawing anything
    plot_style.use()
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
