"""Averaging the repeat iterations of each GWI run, for
combine_results_iterations.py.
"""
import os
import pandas as pd

import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.method.warming_definitions as warming_definitions


# The GWI result types gwi.py writes, as in their file names,
# GWI_results_<type>_<...>.csv: the timeseries of each warming definition,
# the headlines and the rates
RESULT_TYPES = ('timeseries-ANNUAL', 'timeseries-SR15', 'timeseries-AR6',
                'timeseries-CGWL', 'timeseries-RATES', 'headlines')


def is_result_type(file_name, result_type):
    """Whether a results file is of the given type: its name starts with
    GWI_results_<type>_."""
    return file_name.startswith(f'GWI_results_{result_type}_')


def range_result_types(base_path):
    """List the result types that have iteration files in one directory.

    Returns the subset of RESULT_TYPES for which base_path -- one
    regressed-years directory under ITERATIONS_FOLDER -- holds at least one
    file, or [] if the directory does not exist. The directory is listed
    once, and that one listing answers for every type. A file is of a type
    when its name starts with GWI_results_<type>_, so that a type that is
    the start of another's name doesn't also match the other's files.

    In the pipeline: calculate_iteration_averages runs it in a pool, once for
    every regressed-years directory of every scenario, ensemble selection and
    set of regressed variables, and queues an average only for the result
    types it returns.

    Replaces check_headlines_files and check_rates_files. Those ran in the
    parent, one configuration at a time; each listed the configuration's
    directories until it found a file of its type, and if it did, switched
    that type on for every range of the configuration. This is better
    because:
      - the listings run in parallel rather than one after another. Rates
        are rarely written, so check_rates_files listed every directory --
        all 7676 in results_Thorne_2026, which took from 2 s to 35 s on the
        network filesystem depending on its load;
      - each directory is judged on its own files, so nothing is queued for
        a range without iterations of that type. The old checks queued
        those, and each then printed 'No iterations found'.
    """
    try:
        files = os.listdir(base_path)
    except FileNotFoundError:
        return []

    # any() consumes the generator lazily, so for each result type it stops
    # at the first file name of that type rather than testing every name.
    return [
        result_type for result_type in RESULT_TYPES
        if any(is_result_type(f, result_type) for f in files)
        ]


def write_repeat_average(regressed_years, result_type, scenario,
                         ensemble_selection, regressed_vars):
    """Average one dataset's iterations and write the average to disk.

    Calls combine_repeats for one regressed range and result type, which
    writes the ensemble-size-weighted average of that dataset's iterations
    to AGGREGATED_FOLDER. Returns nothing.

    In the pipeline: calculate_iteration_averages runs it in a pool, once for
    every (regressed range, result type) that range_result_types found
    iteration files for.

    Replaces running combine_repeats in the pool directly. combine_repeats
    also returns the averaged frame and every iteration behind it, and a
    pool pickles every return value back to the parent -- which discarded
    them. Returning nothing avoids that transfer, and leaves combine_repeats,
    which does the averaging itself, unchanged.
    """
    combine_repeats(regressed_years, result_type, scenario,
                    ensemble_selection, regressed_vars)


def calculate_iteration_averages():
    """Average the iterations of every result type (RESULT_TYPES) of every
    run.

    Walks ITERATIONS_FOLDER through every scenario, ensemble selection, set
    of regressed variables and regressed range; finds which result types
    each range has iteration files for; and writes the average of each of
    those datasets to AGGREGATED_FOLDER. Returns nothing: the averages are
    read back from disk by load_gwi_priors_erf_obs.

    In the pipeline: the first step of the script, run when --re-calculate=y.
    It uses two pools, each over one flat list of all the work -- first
    range_result_types over every regressed-years directory, then
    write_repeat_average over every (range, result type) to average.

    Replaces a loop over the configurations (scenario, ensemble selection,
    regressed variables) that built a new pool for each one and averaged
    only that configuration's ranges in it. This is better because one pool
    is started instead of one per configuration -- a hundred, for a
    member-by-member scenario -- and the workers share one list of all the
    work instead of one configuration's ranges at a time.
    """

    scenarios_all = results_layout.get_input_labels(
        results_layout.ITERATIONS_FOLDER)
    print(scenarios_all)

    # Walk the tree first, so that the log still reads as a hierarchy even
    # though the averaging that follows is one flat pass.
    range_dirs = []

    for scenario in scenarios_all:
        print('Collecting INPUTS:', scenario)

        ensemble_selections = results_layout.get_subdirs(
            f'{results_layout.ITERATIONS_FOLDER}/{scenario}/',
            'ENSEMBLE-MEMBER--')

        for ensemble_selection in ensemble_selections:

            regressed_variables_all = results_layout.get_subdirs(
                f'{results_layout.ITERATIONS_FOLDER}/{scenario}/'
                f'ENSEMBLE-MEMBER--{ensemble_selection}/',
                'VARIABLES--')

            for regressed_vars in regressed_variables_all:
                _path = (f'{results_layout.ITERATIONS_FOLDER}/' +
                         f'{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')

                regressed_years_vars = results_layout.get_subdirs(
                    _path, 'REGRESSED-YEARS--')

                steps = warming_definitions.check_steps(regressed_years_vars)
                if steps['check_bool']:
                    print(f'  {ensemble_selection} {regressed_vars}: '
                          "all regressed years "
                          f"{steps['range']}")

                for regressed_years in regressed_years_vars:
                    range_dirs.append(
                        (regressed_years, scenario,
                         ensemble_selection, regressed_vars))

    # The directory of each regressed range, whose files say which result
    # types it has iterations for.
    range_paths = [
        iteration_base_path(regressed_years, scenario,
                            ensemble_selection, regressed_vars)
        for regressed_years, scenario, ensemble_selection, regressed_vars
        in range_dirs]

    # Which result types each directory holds. These are thousands of
    # network listings, none depending on another, so they run in the pool.
    print(f'\nChecking {len(range_paths)} regressed-year directories')
    types_per_dir = compute_resources.run_in_pool(
        range_result_types, [(path,) for path in range_paths])

    # One averaging task for each result type found in each directory.
    tasks = []
    for range_dir, result_types in zip(range_dirs, types_per_dir):
        (regressed_years, scenario,
         ensemble_selection, regressed_vars) = range_dir
        for result_type in result_types:
            tasks.append((regressed_years, result_type, scenario,
                          ensemble_selection, regressed_vars))

    print(f'Averaging {len(tasks)} datasets from '
          f'{len(range_dirs)} regressed-year directories')
    compute_resources.run_in_pool(write_repeat_average, tasks)


def iteration_base_path(regressed_years, scenario, ensemble_selection,
                        regressed_vars):
    """Return the iterations directory for a specific configuration."""
    return (
        f'{results_layout.ITERATIONS_FOLDER}/'
        f'{scenario}/'
        f'ENSEMBLE-MEMBER--{ensemble_selection}/'
        f'VARIABLES--{regressed_vars}/'
        f'REGRESSED-YEARS--{regressed_years}/'
    )


def load_iteration_dfs(regressed_years, result_type, scenario,
                       ensemble_selection, regressed_vars):
    """
    Load each repeat iteration's results for a specific configuration.

    Args:
        regressed_years: The range of years used for regression.
        result_type: The type of result, one of RESULT_TYPES (e.g.,
            'timeseries-ANNUAL', 'headlines').
        scenario: The scenario name.
        ensemble_selection: The ensemble selection name.
        regressed_vars: The regressed variables.

    Returns:
        A tuple containing a dictionary of the DataFrame for each iteration,
        and a dictionary of the ensemble size of each iteration, both keyed by
        iteration filename. Both are empty if no files are found.
    """
    dict_iterations = {}
    size_iterations = {}

    base_path = iteration_base_path(
        regressed_years, scenario, ensemble_selection, regressed_vars)

    if not os.path.exists(base_path):
        print(f'Path not found: {base_path}')
        return dict_iterations, size_iterations

    iteration_files = [
        f for f in os.listdir(base_path)
        if is_result_type(f, result_type)
    ]

    if len(iteration_files) == 0:
        print('No iterations found for:',
              result_type, scenario, ensemble_selection,
              regressed_years, regressed_vars)
        return dict_iterations, size_iterations

    for iteration in iteration_files:
        fname = os.path.join(base_path, iteration)
        ens_size = results_layout.ensemble_size_from_name(iteration)
        df_iteration = pd.read_csv(
            fname, index_col=0,  header=[0, 1], skiprows=0)
        dict_iterations[iteration] = df_iteration
        size_iterations[iteration] = ens_size

    return dict_iterations, size_iterations


def combine_repeats(regressed_years, result_type, scenario, ensemble_selection,
                    regressed_vars):
    """
    Average results across iterations for a specific configuration.

    Args:
        regressed_years: The range of years used for regression.
        result_type: The type of result, one of RESULT_TYPES (e.g.,
            'timeseries-ANNUAL', 'headlines').
        scenario: The scenario name.
        ensemble_selection: The ensemble selection name.
        regressed_vars: The regressed variables.

    Returns:
        A tuple containing the averaged DataFrame, a dictionary of all
        iterations, and a dictionary of ensemble sizes, or (None, None, None)
        if no files found.
    """
    dict_iterations, size_iterations = load_iteration_dfs(
        regressed_years, result_type, scenario, ensemble_selection,
        regressed_vars)

    if len(dict_iterations) == 0:
        return None, None, None

    iteration_files = list(dict_iterations.keys())

    # Produce the averaged dataset
    df_avg = (dict_iterations[iteration_files[0]].copy() *
              size_iterations[iteration_files[0]])
    df_avg[:] = 0
    for iteration in iteration_files:
        df_avg += dict_iterations[iteration] * size_iterations[iteration]
    # df_avg /= len(iterations)
    df_avg /= sum(size_iterations.values())

    # Create the specific directory for these regressed years
    out_path = iteration_base_path(
        regressed_years, scenario, ensemble_selection, regressed_vars
        ).replace(results_layout.ITERATIONS_FOLDER,
                  results_layout.AGGREGATED_FOLDER)
    os.makedirs(out_path, exist_ok=True)

    df_avg.to_csv(
        f'{out_path}' +
        f'GWI_results_{result_type}_' +
        f'{scenario}_'
        f'ENSEMBLE-MEMBER--{ensemble_selection}_' +
        f'VARIABLES--{regressed_vars}_' +
        f'REGRESSED-YEARS--{regressed_years}_' +
        'AVERAGE.csv')

    return df_avg, dict_iterations, size_iterations
