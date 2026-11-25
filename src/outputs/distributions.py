"""Distributions: the GWI's netCDF outputs, and the rule for which outputs
are CSV and which are netCDF.

THE OUTPUT RULE

    summary        CSV     the median and the likely and very likely ranges
                           (the 50th, 17th to 83rd and 5th to 95th
                           percentiles) of the ensemble: a small table,
                           for people (src/outputs/summaries.py; written by
                           results_layout.write_csv)
    distributions  netCDF  the ensemble's distribution in full: its
                           quantiles (the percentiles at many levels, e.g.
                           every 0.1), or members themselves (written here,
                           by write_netcdf)

In plain words:

1. A summary is a small table: a row per year, a column per (variable,
   percentile). People open it in a spreadsheet, and the combine script
   averages it across iterations. Its configuration is in its folder and
   file names (results_layout), so a CSV needs nothing more.

2. A distribution is a block of numbers with several dimensions, e.g.
   year × percentile × variable for each definition, or year × member ×
   variable: too many for a table a person reads (1001 levels of 6
   variables in 176 years is a million numbers per definition). It is
   stored as float32 and compressed, it keeps its dimensions' names and
   labels, and it carries its configuration inside it, as attributes,
   because a distribution file gets sent on by itself, away from the
   folder whose names describe it.

3. So every netCDF file says, in its attributes (run_attributes):
   - which datasets it was made from, and which of their members;
   - the pre-industrial era, the regressed variables and years, and the
     output years;
   - the sampling design, the ensemble size N and the seed, so the members
     can be drawn again;
   - the git commit and the command that made it;
   and every data variable says what it is and its units (long_name,
   units). write_netcdf refuses a file that doesn't.

For example, the attributes of a file made by a latin run of a million
members, as ncdump -h prints them:

    :Conventions = "CF-1.11" ;
    :title = "..." ;    // the file's own, from the caller
    :source = "The Global Warming Index (gwi.py), git commit 69d157f" ;
    :history = "2026-11-24T12:00:00Z: python gwi.py --sampling-design=..." ;
    :erf_dataset = "IGCC-2025" ;
    :gmt_dataset = "HadCRUT-2025" ;
    :sampling_design = "latin" ;
    :ensemble_size = 1000000 ;
    :seed = 42 ;
    ...

UNITS. Warming is a temperature difference, so its unit is K (a warming of
1 K is a warming of 1 °C, the unit the CSVs use). Not degC: in UDUNITS,
which CF and most netCDF software follow, degC is a temperature on a scale
offset from K, so software converting a warming of 1.3 degC would give
274.45 K. CF-1.11's units_metadata = "temperature: difference" says it
outright (WARMING_ATTRIBUTES, RATE_ATTRIBUTES).
"""

import datetime as dt
import sys

import numpy as np

import src.infrastructure.run_log as run_log
import src.postprocessing.results_layout as results_layout


# The CF conventions the files follow; 1.11 is the first with
# units_metadata, which marks a temperature as a difference
CF_CONVENTIONS = 'CF-1.11'

# The units of each kind of data variable, with CF's marker that the
# temperature is a difference, not a temperature on a scale (see UNITS,
# above); the caller adds each variable's long_name
WARMING_ATTRIBUTES = {'units': 'K',
                      'units_metadata': 'temperature: difference'}
RATE_ATTRIBUTES = {'units': 'K year-1',
                   'units_metadata': 'temperature: difference'}

# What every data variable must say about itself
REQUIRED_VARIABLE_ATTRIBUTES = ('long_name', 'units')

# What every file must say about itself: CF's, then the run's (see
# run_attributes)
REQUIRED_FILE_ATTRIBUTES = (
    'Conventions', 'title', 'source', 'history', 'references',
    'erf_dataset', 'gmt_dataset', 'picontrol_dataset', 'erf_members',
    'gmt_members', 'preindustrial_era', 'committed_warming',
    'regressed_variables', 'regressed_years', 'output_years',
    'include_regression_constant', 'include_sub_variables',
    'sampling_design', 'ensemble_size', 'seed', 'seed_was', 'git_commit',
)

# How to cite the GWI (as in the README's "How to cite")
REFERENCES = (
    'Walsh, T. Global Warming Index. Zenodo. '
    'https://doi.org/10.5281/zenodo.24020859; '
    'Haustein, K., Allen, M.R., Forster, P.M. et al. A real-time Global '
    'Warming Index. Sci Rep 7, 15417 (2017). '
    'https://doi.org/10.1038/s41598-017-14828-5')

# How the floating-point data variables are stored: float32, as the
# members are held in the engine (its precision, ~7 significant figures,
# is far finer than any warming's uncertainty), compressed with zlib at
# netCDF4's default level (higher levels save little on noisy floats, and
# take longer)
STORED_FLOAT_TYPE = 'float32'
ZLIB_COMPRESSION_LEVEL = 4


def run_attributes(cfg, n_members):
    """The run's configuration, as a netCDF file's attributes.

    cfg:
        the configuration (see gwi.read_argv_config), after the inputs are
        loaded (which can clip the output years to the data's)
    n_members:
        the sample's actual size, N
    Returns a dict of every attribute in REQUIRED_FILE_ATTRIBUTES but the
    title, which the caller adds: each a string or a whole number (netCDF
    attributes have no None or True/False, so options that are off are
    'none' or 'no').
    """
    # The time the file is made, in UTC, to the second
    now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    # The commit of the code, as the run log records it
    git_commit = run_log.describe_git_commit()
    # 'yes' or 'no', for the switches
    yes_no = {True: 'yes', False: 'no'}
    return {
        # CF's own attributes
        'Conventions': CF_CONVENTIONS,
        'source': f'The Global Warming Index (gwi.py), git commit '
                  f'{git_commit}',
        'history': f'{now}: python {" ".join(sys.argv)}',
        'references': REFERENCES,
        # The datasets, and which of their members
        'erf_dataset': cfg.erf_id,
        'gmt_dataset': cfg.gmt_id,
        'picontrol_dataset': cfg.picontrol_id,
        'erf_members': cfg.ensemble_members['ERF'],
        'gmt_members': cfg.ensemble_members['GMT'],
        # The years: the pre-industrial era (or 'none'), the committed
        # warming's (or 'none'), the regression's and the outputs'
        'preindustrial_era': cfg.preindustrial_str,
        'committed_warming': (
            f'{cfg.year_committed_from}-{cfg.year_committed_to}'
            if cfg.year_committed_to else 'none'),
        'regressed_variables': ','.join(cfg.regress_vars),
        'regressed_years': f'{cfg.start_regress}-{cfg.end_regress}',
        'output_years': f'{cfg.start_trunc}-{cfg.end_trunc}',
        'include_regression_constant': yes_no[bool(cfg.inc_reg_const)],
        'include_sub_variables': yes_no[bool(cfg.include_sub_vars)],
        # The sample: its design, its actual size and its seed (given, or
        # drawn by the run and logged)
        'sampling_design': cfg.sampling_design,
        'ensemble_size': int(n_members),
        'seed': int(cfg.seed),
        'seed_was': 'given' if cfg.seed_given else 'drawn',
        # The code
        'git_commit': git_commit,
    }


def check_attributes(dataset):
    """Stop if the dataset doesn't describe itself as the output rule asks
    (see the module docstring).

    dataset:
        an xarray Dataset, about to be written
    Raises ValueError, naming every attribute missing from the file and
    from each data variable.
    """
    # The file's attributes it lacks
    missing = [name for name in REQUIRED_FILE_ATTRIBUTES
               if name not in dataset.attrs]
    # Each data variable's
    for name, variable in dataset.data_vars.items():
        missing += [f'{name}: {attribute}'
                    for attribute in REQUIRED_VARIABLE_ATTRIBUTES
                    if attribute not in variable.attrs]
    if missing:
        raise ValueError(
            'A distribution file must carry its configuration and units '
            '(see src/outputs/distributions.py); missing: '
            f'{", ".join(missing)}.')


def write_netcdf(dataset, path):
    """Write a distribution to a netCDF file, in one step.

    dataset:
        an xarray Dataset, whose attributes are the title plus
        run_attributes(...) (and any more of its own), and each of whose
        data variables has a long_name and units
    path:
        the netCDF file to write (.nc); replaced if it exists

    The floating-point data variables are stored as float32 and compressed
    (STORED_FLOAT_TYPE, ZLIB_COMPRESSION_LEVEL); the rest as they are. The
    file is written under a temporary name, then renamed onto path
    (results_layout.write_atomically), so no one ever reads half of it.
    """
    # Stop if anything the output rule asks for is missing
    check_attributes(dataset)
    # Each floating-point data variable stored as float32, compressed
    encoding = {
        name: {'dtype': STORED_FLOAT_TYPE, 'zlib': True,
               'complevel': ZLIB_COMPRESSION_LEVEL}
        for name, variable in dataset.data_vars.items()
        if np.issubdtype(variable.dtype, np.floating)}
    # The whole file, written in one step
    results_layout.write_atomically(
        path, lambda temp_path: dataset.to_netcdf(
            temp_path, engine='netcdf4', encoding=encoding))
