"""Reading the averaged results, observations, priors, ERFs and piControl
segments back in, for combine_results_iterations.py.
"""
import os
import numpy as np
import pandas as pd

import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout


def read_observations_timeseries(scen, ens):
    """The reference temperatures a run used, as gwi.py saved them: the
    selected GMT members, with the run's pre-industrial offset removed. One
    column per member, indexed by year.

    scen: the inputs label, e.g.
        'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'
    ens: the member selection, e.g. 'ERF-all_GMT-all'
    """
    # The one Obs_timeseries file for these inputs and members
    folder = (f'{results_layout.OBSERVATIONS_FOLDER}/{scen}/'
              f'ENSEMBLE-MEMBER--{ens}/')
    files = sorted(f for f in os.listdir(folder)
                   if f.startswith('Obs_timeseries_'))

    # Stop if it isn't there (results made before gwi.py saved it)
    if len(files) != 1:
        raise FileNotFoundError(
            f'Expected one Obs_timeseries_*.csv file in {folder}, but found '
            f'{len(files)}. Rerun gwi.py for these inputs to save it.')

    return pd.read_csv(os.path.join(folder, files[0]), index_col=0)


def read_picontrol_timeseries(scen):
    """The filtered piControl segments a run sampled internal variability
    from, as gwi.py saved them: one column per segment, indexed by year. None
    if there is no such file (results made before gwi.py saved it).

    scen: the inputs label, e.g.
        'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'
    """
    # The one PiControl_timeseries file for these inputs
    folder = f'{results_layout.PICONTROL_FOLDER}/{scen}/'
    if not os.path.exists(folder):
        return None
    files = sorted(f for f in os.listdir(folder)
                   if f.startswith('PiControl_timeseries_'))
    if len(files) != 1:
        return None

    return pd.read_csv(os.path.join(folder, files[0]), index_col=0)


def read_results_csv(path):
    """Read one results CSV into a DataFrame, or None if it does not exist.

    The file has two header rows -- variable and percentile -- read as a
    column MultiIndex, with the year or period as its index. A missing file
    gives None, which is how the rest of the script tells a result that was
    never written from one that was.

    In the pipeline: load_nested_dfs runs it in a pool, once for every path
    in a nested dictionary of results files (or directly, for a small one).

    Replaces the reading step inside the old recursive load_nested_dfs,
    which read each file as it came to it. It is a function of its own now
    so that the pool can call it.
    """
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, header=[0, 1], skiprows=0)
    return None


def collect_nested_paths(d, paths):
    """Add every file path in a nested dictionary to the set `paths`.

    Walks dictionaries nested to any depth and takes every string value to
    be a path. Returns `paths`, so it can be called as
    collect_nested_paths(d, set()).

    In the pipeline: load_nested_dfs calls it first, for the flat list of
    files it then reads in the pool.
    """
    if isinstance(d, dict):
        for v in d.values():
            collect_nested_paths(v, paths)
    elif isinstance(d, str):
        paths.add(d)
    return paths


def substitute_nested_dfs(d, frames):
    """Copy a nested dictionary, replacing each path with its frame.

    Keeps the keys, their nesting and their order. Every string value is
    looked up in `frames` (path -> DataFrame, or None for a missing file);
    anything else is kept as it is. A path that is not in `frames` raises
    KeyError, rather than coming back as None and being taken for a file
    that was never written.

    In the pipeline: load_nested_dfs calls it last, to put the frames read
    in the pool back where their paths were.
    """
    if isinstance(d, dict):
        return {k: substitute_nested_dfs(v, frames) for k, v in d.items()}
    elif isinstance(d, str):
        return frames[d]
    return d


def load_nested_dfs(d):
    """Return a nested dictionary of file paths with the files read in.

    Takes a dictionary, nested to any depth, whose string values are paths
    to results CSVs, and returns the same structure with each path replaced
    by its DataFrame -- or by None where the file does not exist.

    In the pipeline: load_gwi_priors_erf_obs calls it four times, for the GWI
    results, priors, ERFs and observations. It collects every path into one
    flat list (collect_nested_paths), reads them in a pool
    (read_results_csv), and puts the frames back (substitute_nested_dfs). A
    dictionary of fewer than 64 files is read directly instead.

    Replaces a version that walked the dictionary recursively and read each
    file in the parent as it came to it. This is better because the reading
    is now parallel. pandas spends several milliseconds on each file building
    its two-row column MultiIndex, and a scenario run member by member --
    e.g. GMT={1..100} with ERF=all, as in the Thorne et al. analysis -- is
    fifteen thousand files: for results_Thorne_2026, 87 s one after another
    against about 5 s in the pool on 28 workers. The frames are small, so
    sending them back to the parent costs far less than parsing them there.
    """
    paths = sorted(collect_nested_paths(d, set()))

    # Starting a pool costs about 0.2 s, the same as reading 60-70 files one
    # after another (measured with all of results_Thorne_2026 loaded, 28
    # workers). So smaller dictionaries -- the priors and ERFs, a handful of
    # files each -- are read directly.
    if len(paths) < 64:
        frames = {path: read_results_csv(path) for path in paths}
    else:
        frames = dict(zip(paths, compute_resources.run_in_pool(
            read_results_csv, [(path,) for path in paths])))

    return substitute_nested_dfs(d, frames)


def load_gwi_priors_erf_obs():
    """Load all averaged datasets."""
    results_files = {}
    priors_files = {}
    erf_files = {}
    obs_files = {}

    scenarios_all = results_layout.get_input_labels(
        results_layout.AGGREGATED_FOLDER)

    for scenario in scenarios_all:
        results_files.update({scenario: {}})
        priors_files.update({scenario: {}})
        erf_files.update({scenario: {}})
        obs_files.update({scenario: {}})

        _path = f'{results_layout.AGGREGATED_FOLDER}/{scenario}/'

        ensembles_seletions_all = results_layout.get_subdirs(
            _path, 'ENSEMBLE-MEMBER--')

        for ensemble_selection in ensembles_seletions_all:
            results_files[scenario].update({ensemble_selection: {}})
            priors_files[scenario].update({ensemble_selection: {}})
            erf_files[scenario].update({ensemble_selection: {}})
            obs_files[scenario].update({ensemble_selection: {}})

            _path = (f'{results_layout.AGGREGATED_FOLDER}/' +
                     f'{scenario}/' +
                     f'ENSEMBLE-MEMBER--{ensemble_selection}/')

            regressed_variables_all = results_layout.get_subdirs(
                _path, 'VARIABLES--')

            for regressed_vars in regressed_variables_all:
                results_files[scenario][ensemble_selection].update(
                    {regressed_vars: {}})
                priors_files[scenario][ensemble_selection].update(
                    {regressed_vars: {}})
                erf_files[scenario][ensemble_selection].update(
                    {regressed_vars: {}})

                # The priors and ERF files, under the same regressed
                # variables (their variables are the forcing variables those
                # give)
                _path_prior_dir = (f'{results_layout.PRIORS_FOLDER}/' +
                                   f'{scenario}/' +
                                   f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                                   f'VARIABLES--{regressed_vars}/')
                _path_erf_dir = (f'{results_layout.ERFS_FOLDER}/' +
                                 f'{scenario}/' +
                                 f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                                 f'VARIABLES--{regressed_vars}/')
                if os.path.exists(_path_prior_dir):
                    for f in os.listdir(_path_prior_dir):
                        if f.startswith('PRIOR_results_timeseries_'):
                            priors_files[scenario][ensemble_selection][
                                regressed_vars]['timeseries'] = (
                                    os.path.join(_path_prior_dir, f))
                if os.path.exists(_path_erf_dir):
                    for f in os.listdir(_path_erf_dir):
                        if f.startswith('ERF_results_timeseries_'):
                            erf_files[scenario][ensemble_selection][
                                regressed_vars]['timeseries'] = (
                                    os.path.join(_path_erf_dir, f))

                _path = (f'{results_layout.AGGREGATED_FOLDER}/' +
                         f'{scenario}/' +
                         f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                         f'VARIABLES--{regressed_vars}/')

                regressed_years_vars = results_layout.get_subdirs(
                    _path, 'REGRESSED-YEARS--')

                for regressed_years in regressed_years_vars:
                    # Load GWI results files
                    # The plots' names for the results they read, and the
                    # result types of their files
                    res_type_dict = {
                        res_name: (
                                f'{results_layout.AGGREGATED_FOLDER}/' +
                                f'{scenario}/' +
                                f'ENSEMBLE-MEMBER--{ensemble_selection}/' +
                                f'VARIABLES--{regressed_vars}/' +
                                f'REGRESSED-YEARS--{regressed_years}/' +
                                f'GWI_results_{res_type}_' +
                                f'{scenario}_'
                                f'ENSEMBLE-MEMBER--{ensemble_selection}_' +
                                f'VARIABLES--{regressed_vars}_' +
                                f'REGRESSED-YEARS--{regressed_years}_' +
                                'AVERAGE.csv'
                            )
                        for res_name, res_type in [
                            ('timeseries', 'timeseries-ANNUAL'),
                            ('headlines', 'headlines'),
                            ('rates', 'timeseries-RATES')]
                    }
                    results_files[scenario
                                  ][ensemble_selection
                                    ][regressed_vars
                                      ].update({
                                          regressed_years: res_type_dict
                                          })

                    # Load observations files
                    obs_files[scenario
                              ][ensemble_selection
                                ].update({regressed_years: {}})

                    _path_obs_dir = (f'{results_layout.OBSERVATIONS_FOLDER}/' +
                                     f'{scenario}/' +
                                     f'ENSEMBLE-MEMBER--{ensemble_selection}/'
                                     f'REGRESSED-YEARS--{regressed_years}/')
                    # The observation headlines don't change for different
                    # regression variables, which means that the file only
                    # needs to be loaded once. It is faster to check this than
                    # access os.listdir multiple times and keep overwriting the
                    # file path in the dictionary.
                    if 'headlines' not in obs_files[scenario
                                                    ][ensemble_selection
                                                      ][regressed_years]:
                        if os.path.exists(_path_obs_dir):
                            for f in os.listdir(_path_obs_dir):
                                if f.startswith('Obs_results_headlines_'):
                                    obs_files[scenario
                                              ][ensemble_selection
                                                ][regressed_years
                                                  ]['headlines'
                                                    ] = \
                                        os.path.join(_path_obs_dir, f)

    print('\nLoading all averaged datasets')
    # Load results timeseries and headlines
    results_dfs = load_nested_dfs(results_files)
    # Load priors timeseries
    priors_dfs = load_nested_dfs(priors_files)
    # Load ERF timeseries
    erf_dfs = load_nested_dfs(erf_files)
    # Load observation headlines
    obs_dfs = load_nested_dfs(obs_files)

    # Load the temperature observations timeseries each run used
    print('Loading temperature observations timeseries...')
    for scen in results_dfs.keys():
        for ens in results_dfs[scen].keys():
            obs_dfs[scen][ens]['timeseries'] = read_observations_timeseries(
                scen, ens)

    return results_dfs, priors_dfs, erf_dfs, obs_dfs


def load_historical_only_dfs(results_dfs):
    """Load historical-only datasets; add to the results_files dictionary."""
    print('\nLoading historical-only datasets into results_dfs structure')
    # Load historical-only data from csv files if they are available
    # Iterate over the structure identified in results_files
    # This is reasonable because historical-only datasets can only exist in
    # the same directories/configurations as the main results datasets.
    for scen in results_dfs.keys():
        for ens in results_dfs[scen].keys():
            for reg_vars in results_dfs[scen][ens].keys():
                # Find all historical dataset in the results/aggregated folder
                # and add them all to the dictionary.
                _path = (f'{results_layout.AGGREGATED_FOLDER}/' +
                         f'{scen}/' +
                         f'ENSEMBLE-MEMBER--{ens}/' +
                         f'VARIABLES--{reg_vars}/')
                # Sorted, because os.listdir order is whatever the
                # filesystem gives, and the headlines are drawn in this order
                # -- unsorted, two runs on the same numbers could stack their
                # lines and legends differently.
                hist_files = sorted([
                    f for f in os.listdir(_path)
                    if f.startswith('GWI_results_')
                    and 'HISTORICAL-ONLY' in f])

                for hist_file in hist_files:
                    _df = pd.read_csv(_path + hist_file,
                                      index_col=0, header=[0, 1], skiprows=0)
                    headline_name = hist_file.split('GWI_results_')[1].split(
                        '_HISTORICAL-ONLY')[0]

                    if 'HISTORICAL-ONLY' not in results_dfs[scen
                                                            ][ens
                                                              ][reg_vars]:
                        results_dfs[scen][ens][reg_vars][
                            'HISTORICAL-ONLY'] = {}

                    results_dfs[scen][ens][reg_vars][
                        'HISTORICAL-ONLY'].update({headline_name: _df})

    return results_dfs


def is_dataset_present(data_dict, required_keys):
    """
    Check if the required datasets are present in the provided dictionary
    and are not None.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing datasets, typically
        results_dfs[scen][ens][reg_vars][reg_range]
    required_keys : list or str
        List of keys (or single key) that must be present and not None.
        
    Returns:
    --------
    bool
        True if all required datasets are present and not None, False
        otherwise.
    """
    if isinstance(required_keys, str):
        required_keys = [required_keys]
        
    for key in required_keys:
        if key not in data_dict or data_dict[key] is None:
            return False
    return True


def headline_var_available(df, period, var, context=''):
    """
    Check whether a variable can be plotted for a given headline period.

    The bar plotters previously tested only whether a variable's column
    existed, which is not the same as it having a value: a column present but
    NaN for this period was passed to matplotlib as a NaN bar with NaN error
    bars, which produces an "All-NaN axis encountered" RuntimeWarning from the
    autoscaler and a phantom labelled slot with no bar in the figure.

    There is exactly one situation in which a missing value is legitimate:
    'Res' (the residual, Obs - Tot) on a CGWL period. The CGWL definition is a
    20-year window centred on the present (year-9 to year+10), so whenever it
    extends beyond the end of the observations there is no observed
    temperature from which to form a residual, and gwi.py deliberately writes
    NaN there. Note that gwi.py already applies the same logic to the
    *observed* CGWL headline, which it declines to compute at all when the
    window runs past the data.

    Any other non-finite headline means the results do not match the
    configuration that produced them (eg a truncation range that runs past the
    end of the observations for that dataset vintage), so it is raised rather
    than quietly skipped.

    Parameters:
    -----------
    df : pd.DataFrame
        Headlines dataframe, indexed by period with (variable, percentile)
        columns.
    period : str
        The headline period (row label), eg '2016-2035 (CGWL definition)'.
    var : str
        The variable to check, eg 'GHG' or 'Res'.
    context : str
        Description of the run being plotted, included in the error message to
        make an unexpected NaN traceable back to its scenario/range.

    Returns:
    --------
    bool
        True if the variable has a finite median for this period, False if it
        is legitimately absent.
    """
    # No such dataset for this run at all (eg a scenario whose observation
    # results were never produced), or the period is absent from it. The
    # latter is how gwi.py signals an uncomputable observed headline: it
    # declines to write a CGWL row when the window runs past the observations,
    # rather than writing NaN.
    if df is None or period not in df.index:
        return False

    # Not calculated for this run at all (eg sub-variables that were not
    # requested); nothing anomalous about this, so nothing to plot.
    if (var, '50') not in df.columns:
        return False

    if np.isfinite(df.loc[period, (var, '50')]):
        return True

    # The one expected absence: a residual for a window reaching into the
    # future, where no observations exist yet.
    if var == 'Res' and 'CGWL' in str(period):
        return False

    raise ValueError(
        f"Headline '{var}' is NaN for period '{period}'{context}. Only 'Res' "
        "on a CGWL period may legitimately be missing; this indicates the "
        "results are inconsistent with the configuration that produced them "
        "(eg a truncation range extending beyond the observations).")


def priors_available(priors_dfs, scen, ens, reg_vars):
    """Whether prior warming results exist for this scenario, ensemble
    selection and set of regressed variables.

    gwi.py always writes the prior dataset now, but results written before
    (when it had a --calculate-priors-output option) may lack it. The priors
    are plotting-only, so their absence is not an error: the figures that
    use them are skipped instead.

    Checks the value rather than just the key, because load_nested_dfs maps a
    path that no longer exists to None.
    """
    return (priors_dfs.get(scen, {}).get(ens, {}).get(reg_vars, {})
            .get('timeseries') is not None)
