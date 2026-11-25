"""The Global Warming Index (GWI): how much of the observed warming each
cause (greenhouse gases, other human forcings, natural forcing) is
responsible for, with its uncertainty.

One run is a short pipeline (main, at the bottom):

    read_argv_config            the command line (run-gwi.sh passes every
                                option)
    output_folders              where this run's outputs go
    load_inputs                 the forcing, the observed temperatures, the
                                piControl segments and the FaIR parameter
                                sets
    draw_members                which members of the full ensemble to
                                rebuild
    modelled_warming            stage A: FaIR's warming from each forcing
                                variable
    constrain                   stage B: the scaling factors that fit it to
                                the observations
    members.Sample              stage C: the drawn members, rebuilt as each
                                output needs them
    attributed_percentiles, observed_percentiles,
    prior_and_erf_percentiles   the percentiles: the summaries (CSV), and
                                the quantiles (netCDF)
    write_inputs_and_interims   the inputs as used, the priors and the ERF
    make_releases               the release: a few representative members,
                                rebuilt over every year and definition
    write_outputs               each definition's timeseries and the
                                headlines (CSV), the quantiles and the
                                releases (netCDF)

Each warming definition (annual, SR1.5, AR6, CGWL, rates) is described once,
in WARMING_DEFINITIONS, and applied in the same way to the attributed
warming and to the observations.

NAMING. The inputs (see load_inputs), each a labelled xarray DataArray:

    erf     the forcing                (year, erf_member, forcing_variable)
    gmt     the observed temperatures  (year, gmt_member)
    pic     the piControl segments     (year, pic_member)
    params  the FaIR parameter sets    (param_member, parameter, box)

- A member is one instance of a source of uncertainty (one ERF member, one
  observed member, one piControl segment, one parameter set), so each
  source's dimension is <source>_member: the axes e, o, c and p.
- Every input is year first, then its members, then its variables, so its
  values (.to_numpy()) are already in the order the engine uses: e.g. the
  forcing as (t, e, v).
- DataArrays are named for their source, as above; numpy arrays by their
  symbols in the method's equations (T_mod, X, Y_obs, beta), with their
  shapes in a comment; DataFrames appear only as the CSV tables.

OUTPUTS. Summaries (the median and the likely and very likely ranges)
are CSV; distributions (the quantiles, and members) are netCDF, carrying
the run's configuration as attributes: the output rule, in
src/outputs/distributions.py.
"""

import fractions
import os
import sys
import types

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

import src.method.emulators as emulators
import src.method.members as members
import src.method.regression as regression
import src.method.release as release
import src.method.sampling as sampling
import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.infrastructure.run_log as run_log
import src.method.warming_definitions as warming_definitions
import src.outputs.distributions as distributions
import src.outputs.summaries as summaries
import src.inputs.input_datasets as datasets


###############################################################################
# SETTINGS ####################################################################
###############################################################################

# Percentiles to calculate and use throughout analysis.
# SIGMA_RANGES = [[32, 68], [5, 95], [0.3, 99.7]]
# These are the percentile ranges used in the IPCC likelihood statements:
# Typically, 17-83 is likely range, and 5-95 is very likely range.
SIGMA_RANGES = [[17, 83], [5, 95]]
# Every percentile the outputs give: the ends of the ranges, sorted, then
# the median; this order is the order of the CSVs' percentile columns
SIGMAS = list(
    np.concatenate((np.sort(np.ravel(SIGMA_RANGES)), [50]), axis=0)
    )
# The quantiles (the distribution finely, e.g. every 0.1 percentile) are
# set per run, by --quantile-step (see parse_quantile_step), and taken in
# the same call as SIGMAS

# The largest release file to write (see make_releases): a guard against
# asking by mistake for a release too big to hand anyone. 500 MB is about
# 25,000 members with every year and definition (1000 members take ~21
# MB); a run asking for more stops before rebuilding anything.
MAX_RELEASE_BYTES = 500e6

# How to use a release, in each release file's attributes
RELEASE_USAGE = (
    'Every member has equal weight: use all of them, or the first of them, '
    'any number (they are in an order in which every first k is spread '
    'across the distribution), and do not weight or resample them. Each '
    'member is one consistent draw of the uncertain inputs (a FaIR '
    'parameter set, an ERF member, an observed member and a piControl '
    'segment) across all its years, variables and definitions.')

# SR1.5 warming is defined as the 30-year mean human-induced warming, relative
# to the pre-industrial period, centred using trend extrapolation where
# necessary. Where that trend is linear, the 30-year mean is equivalent to the
# end point of the trend line through the most recent 15 years.
# NOTE: here we centre warming on the mid-point of a year (not the end), so a
# 29 year mean of annual means is used instead of 30 years.
# How many annual means the SR1.5 definition's mean takes: an odd number,
# centred on the headline year:
#   29: the mean over 2011-2039 for 2025, the trend through 2011-2025
#   31: the mean over 2010-2040 for 2025, the trend through 2010-2025
# (SR1.5, Ch.1, Sect. 1.2.1)
SR15_MEAN_YEARS = 29
if SR15_MEAN_YEARS % 2 == 0:
    raise ValueError(
        f'SR15_MEAN_YEARS must be odd, not {SR15_MEAN_YEARS}.')
# How many annual means the trend is fitted through, which that mean
# implies: the headline year and the (SR15_MEAN_YEARS - 1) / 2 years before
# it (the line's value at the middle of the headline year equals its mean
# over the SR15_MEAN_YEARS years centred there)
SR15_TREND_YEARS = (SR15_MEAN_YEARS + 1) // 2

# AR6 warming is defined as the lagged 10-year mean.
# NOTE: here we index decade average warming as the mid-point of the end-year
# How many annual means the AR6 definition averages: the headline year
# and the AR6_MEAN_YEARS - 1 years before it, e.g. 2016-2025 for 2025
# (AR6's decadal means, e.g. 2011-2020)
# (AR6 WGI, Ch. 3, Table 3.1)
AR6_MEAN_YEARS = 10

# A new proposal, the CGWL (Betts et al., 2023), is defined as the 20 year
# mean centred using warming projections after the current year.
# NOTE: here we centre warming on the mid-point of a year (not the end), so a
# 19 year mean of annual means is used instead of 20 years.
# How many annual means the CGWL hybrid definition averages:
#   19: 2016-2034 for 2025, centred on mid-2025
#   20: 2016-2035 for 2025, centred on the end of 2025
CGWL_MEAN_YEARS = 19
if CGWL_MEAN_YEARS % 2 == 0:
    raise ValueError(
        f'CGWL_MEAN_YEARS must be odd, not {CGWL_MEAN_YEARS}.')
# The CGWL's half-window: the years either side of the headline year
HALF_CGWL = (CGWL_MEAN_YEARS - 1) // 2

# AR6 defined rate as the 10 year trend.
# How many annual means the AR6 rate definition's trend is fitted
# through: the year and the AR6_RATE_YEARS - 1 years before it, e.g.
# 2016-2025 for 2025; the rate is the line's slope, in °C per year
# (AR6 WGI, Ch. 3, Table 3.1)
AR6_RATE_YEARS = 10

# Each warming definition, by its name in --definitions, as a dictionary:
#   file_name     its name in the output files
#   function      applies it to a window of years (see
#                 src/method/warming_definitions.py)
#   years_before  how many years before each year its window takes
#   years_after   how many years after
#   label         its label for a year, in the headlines CSVs
#   description   what it is, for the quantiles netCDF's long_name
#   units         its units there (see src/outputs/distributions.py)
#   ipcc_years    the headline years the IPCC reports assessed with it
#                 (SR1.5's 2017, AR6's 2010-2019); --headline-years=IGCC
#                 repeats them in each update, to validate it and show how
#                 the assessment has changed since
# The attributed warming and the observations both use it.
WARMING_DEFINITIONS = {
    'annual': {
        'file_name': 'ANNUAL',
        'function': warming_definitions.annual_value,
        'years_before': 0,
        'years_after': 0,
        'label': lambda year: year,
        'description': 'attributed warming in each year',
        'units': distributions.WARMING_ATTRIBUTES,
        'ipcc_years': [2017],
    },
    'SR1.5': {
        'file_name': 'SR15',
        'function': warming_definitions.final_value_of_trend_vectorised,
        'years_before': SR15_TREND_YEARS - 1,
        'years_after': 0,
        'label': lambda year: f'{year} (SR15 definition)',
        'description': ('attributed warming, SR1.5 definition: the mean of '
                        f'the {SR15_MEAN_YEARS} years centred on each year, '
                        'using extrapolated warming for any future years; '
                        'for linear extrapolation, this is equivalent to the '
                        'end year of a line fit through the '
                        f'{SR15_TREND_YEARS} years up to each year'),
        'units': distributions.WARMING_ATTRIBUTES,
        'ipcc_years': [2017],
    },
    'AR6': {
        'file_name': 'AR6',
        'function': warming_definitions.mean_over_years,
        'years_before': AR6_MEAN_YEARS - 1,
        'years_after': 0,
        'label': lambda year: f'{year - (AR6_MEAN_YEARS - 1)}-{year}',
        'description': ('attributed warming, AR6 definition: the mean of '
                        f'the lagged {AR6_MEAN_YEARS} years up to each year'),
        'units': distributions.WARMING_ATTRIBUTES,
        'ipcc_years': [2019],
    },
    'CGWL': {
        'file_name': 'CGWL',
        'function': warming_definitions.mean_over_years,
        'years_before': HALF_CGWL,
        'years_after': HALF_CGWL,
        'label': lambda year: (f'{year - HALF_CGWL}-{year + HALF_CGWL} '
                               '(CGWL definition)'),
        'description': ('attributed warming, CGWL definition: the mean of '
                        f'the {CGWL_MEAN_YEARS} years centred on each year, '
                        'using projected warming for any future years'),
        'units': distributions.WARMING_ATTRIBUTES,
        'ipcc_years': [],
    },
    'rates': {
        'file_name': 'RATES',
        'function': warming_definitions.rate_func_vectorised,
        'years_before': AR6_RATE_YEARS - 1,
        'years_after': 0,
        'label': lambda year: (f'{year - (AR6_RATE_YEARS - 1)}-{year} '
                               '(AR6 rate definition)'),
        'description': ('attributed warming rate, AR6 rate definition: the '
                        'slope of the line fitted through the lagged '
                        f'{AR6_RATE_YEARS} years up to each year'),
        'units': distributions.RATE_ATTRIBUTES,
        'ipcc_years': [2019],
    },
}

# Every option gwi.py reads. Anything else on the command line stops the
# run: an unknown option is most likely a typo (e.g. --ensemble=1e6),
# which would otherwise be ignored silently, leaving the default in its
# place.
KNOWN_OPTIONS = (
    '--sampling-design', '--ensemble-size', '--seed', '--regress-range',
    '--truncate',
    '--definitions', '--headline-years',
    '--regress-variables', '--erf-dataset', '--gmt-dataset',
    '--picontrol-dataset', '--committed', '--preindustrial-era',
    '--include-reg-const', '--erf-member', '--gmt-member',
    '--include-sub-vars', '--output-tag', '--quantile-step',
    '--release-members', '--release-method', '--release-target',
)


###############################################################################
# STAGES A AND B ##############################################################
###############################################################################
def modelled_warming(erf, params, inc_pi_offset, start_pi, end_pi):
    """Stage A: the modelled warming from each forcing variable alone,

        T_mod[p, e, t, v]

    for every FaIR parameter set p, every ERF member e, every forcing year t
    and every forcing variable v. Both the attribution (stages B and C) and
    the priors are built from it.

    erf:
        the forcing, a DataArray (year, erf_member, forcing_variable), as
        from input_datasets.load_ERF: its ERF members in the order of the e
        axis, its variables sorted
    params:
        the FaIR thermal parameter sets, a DataArray (param_member,
        parameter, box), as from emulators.fair_parameter_sets: its sets in
        the order of the p axis
    inc_pi_offset:
        whether to express the modelled warming relative to its mean over the
        pre-industrial years, start_pi to end_pi (both ends included); without
        it, it is relative to the first forcing year
    Returns (T_mod, forcing_vars):
        T_mod: an array (p, e, t, v) over the forcing years (erf's years)
        forcing_vars: the names along v, sorted.
    """
    # The forcing variables, sorted: the v axis
    forcing_vars = erf['forcing_variable'].to_numpy().tolist()
    # The forcing years: the t axis
    forc_Yrs = erf.year.to_numpy()

    # The modelled warming, for every parameter set and every ERF member:
    # (p, e, t, v)
    T_mod = emulators.run_fair_v2_alpha(erf, params)

    # Express it relative to its pre-industrial mean, if asked to: each
    # (parameter set, ERF member, variable) relative to its own mean over
    # the pre-industrial years
    if inc_pi_offset:
        # Which forcing years are pre-industrial
        in_pi = (forc_Yrs >= start_pi) & (forc_Yrs <= end_pi)
        # The modelled warming over those years: (p, e, t_pi, v)
        T_mod_pi = T_mod[:, :, in_pi, :]
        # Its mean over the years: (p, e, 1, v)
        T_mod_pi_mean = T_mod_pi.mean(axis=2, keepdims=True)
        # Subtract it, broadcast over the years: (p, e, t, v)
        T_mod = T_mod - T_mod_pi_mean
    return T_mod, forcing_vars


def constrain(T_mod, forcing_vars, forc_Yrs, gmt, pic, regress_vars,
              inc_reg_const, start_regress, end_regress):
    """Stage B: constrain the modelled warming by the observations. The
    scaling factors that fit each (parameter set, ERF member)'s modelled
    warming to each observed member, and to each piControl segment:

        β_obs[p, e, o, k]   fit to observed member o
        β_pic[p, e, c, k]   fit to piControl segment c

    A member (p, e, o, c)'s scaling factors are β_obs[p, e, o] +
    β_pic[p, e, c] (see src/method/members.py). Every member of every
    source is fitted, whichever members are later chosen.

    The axes:

        p  FaIR parameter set
        e  ERF ensemble member
        o  observed GMT member
        c  piControl segment
        t  year
        v  forcing variable (e.g. GHG, Nat, OHF, and any sub-variables)
        k  regressor: one per regressed variable, plus one for the constant
           term if it is used

    T_mod:
        array (p, e, t, v), stage A's modelled warming (see modelled_warming)
    forcing_vars:
        the names along its v axis
    forc_Yrs:
        the years along its t axis
    gmt:
        every observed GMT member, a DataArray (year, gmt_member), as from
        input_datasets.load_GMT: its members in the order of the o axis
    pic:
        every piControl segment, a DataArray (year, pic_member), as from
        input_datasets.filter_PiControl: its segments in the order of the c
        axis
    regress_vars:
        the regressed variables, e.g. ['GHG', 'Nat', 'OHF']
    inc_reg_const:
        whether the regression has a constant term
    start_regress, end_regress:
        the regression years (both included)
    Returns regression.Coefficients:
        obs (p, e, o, k) and pic (p, e, c, k).
    """
    # Every observed GMT member, and every piControl segment, is fitted onto
    # each (parameter set, ERF member)'s modelled warming, all in one batched
    # solve. The regressors X are the regressed variables (plus a column of
    # ones, if a constant is used) over the regression years: one matrix
    # (t, k) per (p, e).

    # Where each regressed variable sits on the variable axis
    reg_indices = [forcing_vars.index(var) for var in regress_vars]
    # Which forcing years are in the regression
    in_regress_forc = (forc_Yrs >= start_regress) & (forc_Yrs <= end_regress)
    # The modelled warming over the regression years: (p, e, t, v)
    T_mod_regress = T_mod[:, :, in_regress_forc, :]
    # Its regressed variables: (p, e, t, k)
    X = T_mod_regress[..., reg_indices]
    if inc_reg_const:
        # Add a column of ones to every matrix, for the constant: k + 1
        X = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    # The observed GMT members over the regression years: (t, o)
    Y_obs = gmt.sel(year=slice(start_regress, end_regress)).to_numpy()
    # The piControl segments over the regression years: (t, c)
    Y_pic = pic.sel(year=slice(start_regress, end_regress)).to_numpy()
    # Every fit at once: coefs.obs (p, e, o, k), coefs.pic (p, e, c, k)
    coefs = regression.ols(X, Y_obs, Y_pic)
    return coefs


###############################################################################
# THE PIPELINE ################################################################
###############################################################################

def parse_ensemble_size(text):
    """The ensemble size from --ensemble-size: a whole number of members,
    at least 1.

    text:
        the option's value, a whole number (e.g. 1000000) or in e notation
        (e.g. 1e6)
    Returns N, an int.
    """
    # The value as a number, if it is one
    try:
        value = float(text)
    except ValueError:
        value = None
    # ... which must be a whole number of members, at least one
    if value is None or not value.is_integer() or value < 1:
        raise ValueError(
            f'Invalid --ensemble-size {text!r}: use a whole number of '
            'members, at least 1, e.g. 1000000 or 1e6.')
    return int(value)


def parse_quantile_step(text):
    """The quantiles' levels from --quantile-step: every step percent from
    the 0th to the 100th percentile.

    text:
        the option's value: a step in percent that divides 100 (e.g. 0.1
        for 1001 levels, 1 for 101, 5 for 21), or 'none' (or empty) for no
        quantiles
    Returns the levels, an array, or None for no quantiles.

    Each level is k × step, worked out exactly and then rounded once, so it
    prints as it reads (0.3, where steps of 0.1 added up give
    0.30000000000000004).
    """
    # No quantiles
    if text in ('', 'none'):
        return None
    # The step, exactly (0.1 as 1/10), if it is a number
    try:
        step = fractions.Fraction(text)
    except ValueError:
        step = None
    # ... more than zero, and dividing 100 into whole steps
    if step is None or step <= 0 or (100 / step).denominator != 1:
        raise ValueError(
            f'Invalid --quantile-step {text!r}: use a step in percent that '
            'divides 100 (e.g. 0.1 for every 0.1 percentile, 1 for every '
            'percentile), or none.')
    # Every step from 0 to 100, each rounded once
    return np.array([float(k * step) for k in range(int(100 / step) + 1)])


def parse_release_sizes(text):
    """The releases' sizes from --release-members: one release of M members
    for each.

    text:
        the option's value: sizes separated by commas (e.g. 1000, or
        100,1000,10000; e notation allowed, e.g. 1e3), or 0 (or empty) for
        no release
    Returns the sizes, a list of ints (empty for none).
    """
    # No release
    if text in ('', '0'):
        return []
    # Each size, a whole number of members, at least 1
    sizes = [parse_ensemble_size(size) for size in text.split(',')]
    # Stop if any is given twice (its files would overwrite each other)
    if len(set(sizes)) != len(sizes):
        raise ValueError(f'--release-members={text} gives a size twice.')
    return sizes


def parse_release_target(text):
    """What a release is made for, from --release-target: the variable and
    year whose distribution it matches best (a stratified release is
    stratified on it).

    text:
        the option's value: a variable and a year, e.g. Ant:2025 (the year
        stated, never implied); empty (or just ':') for none
    Returns (variable, year): e.g. ('Ant', 2025), or (None, None).
    """
    # None given
    if text in ('', ':'):
        return None, None
    # The variable and the year, which must be a whole number
    variable, colon, year = text.partition(':')
    if not (variable and colon and year.isdigit()):
        raise ValueError(
            f'Invalid --release-target {text!r}: give a variable and a '
            'year, e.g. Ant:2025.')
    return variable, int(year)


def read_argv_config(argv):
    """Read the run's configuration from the command line.

    argv:
        the options, each --name=value (see KNOWN_OPTIONS; run-gwi.sh passes
        them all); some missing options are asked for interactively
    Returns cfg, a SimpleNamespace of the configuration:
    - pre-industrial period:
        inc_pi_offset, start_pi, end_pi, preindustrial_str
    - regression:
        inc_reg_const, start_regress, end_regress, regress_vars
    - output years:
        start_trunc, end_trunc
    - committed warming:
        year_committed_from, year_committed_to, committed_str
    - sampling:
        sampling_design, ensemble_size (None for the all design), seed
        (drawn here if none was given), seed_given
    - outputs:
        definitions, headline_years, include_sub_vars, output_tag,
        quantile_step (as given), quantile_levels (None for no quantiles)
    - releases:
        release_sizes (empty for none), release_method,
        release_target_variable and release_target_year (None if not
        given)
    - inputs:
        erf_id, gmt_id, picontrol_id, ensemble_members, ensemble_members_str
    """
    # Each option's value, by name: options left out are absent (most then
    # fall back to a default, or to asking interactively)
    argv_dict = {}
    for argument in argv:
        name, equals, value = argument.partition('=')
        if not equals:
            raise ValueError(
                f'Invalid argument {argument!r}: give options as '
                '--name=value (e.g. --ensemble-size=1e6).')
        if name not in KNOWN_OPTIONS:
            raise ValueError(
                f'Unknown option {name!r}. The options are: '
                f'{", ".join(KNOWN_OPTIONS)}.')
        if name in argv_dict:
            raise ValueError(f'Option {name!r} is given more than once.')
        argv_dict[name] = value
    cfg = types.SimpleNamespace()

    # The pre-industrial period, whose mean is the zero of the modelled
    # warming, the observations, and the piControl segments (both years
    # included:
    # 1850-1900 is 51 years, IPCC SR1.5, Ch. 1);
    # 'n' for none
    if '--preindustrial-era' in argv_dict:
        if argv_dict['--preindustrial-era'].lower() in ['n', 'none', 'false']:
            cfg.inc_pi_offset = False
            cfg.start_pi, cfg.end_pi = None, None
        else:
            cfg.inc_pi_offset = True
            cfg.start_pi = int(argv_dict['--preindustrial-era'].split('-')[0])
            cfg.end_pi = int(argv_dict['--preindustrial-era'].split('-')[1])
    else:
        # Defaults to the standard in IPCC AR6 Ch.3
        cfg.inc_pi_offset = True
        cfg.start_pi, cfg.end_pi = 1850, 1900
    # ... as named in the outputs ('none' with no offset)
    cfg.preindustrial_str = (f'{cfg.start_pi}-{cfg.end_pi}'
                             if cfg.inc_pi_offset else 'none')

    # Whether the regression has a constant term
    if '--include-reg-const' in argv_dict:
        cfg.inc_reg_const = (argv_dict['--include-reg-const'].lower()
                             in ['y', 'yes', 'true'])
    else:
        cfg.inc_reg_const = True
    if not cfg.inc_pi_offset and not cfg.inc_reg_const:
        run_log.note('Both the preindustrial offset and the regression '
                     'constant are disabled. Without these centring factors '
                     'the regression attribution may behave unexpectedly.')

    # The regression years: for the full GWI, the whole observed record;
    # for a historical-only GWI, each end year in turn
    if '--regress-range' in argv_dict:
        regress_range = argv_dict['--regress-range']
    else:
        regress_range = input('Regression range (start-end): ')
    cfg.start_regress = int(regress_range.split('-')[0])
    cfg.end_regress = int(regress_range.split('-')[1])

    # The output years ("truncation"): may reach past the regression years,
    # to give constrained projections where the forcing goes further;
    # defaults to the regression years
    if '--truncate' in argv_dict:
        cfg.start_trunc = int(argv_dict['--truncate'].split('-')[0])
        cfg.end_trunc = int(argv_dict['--truncate'].split('-')[1])
    else:
        cfg.start_trunc, cfg.end_trunc = cfg.start_regress, cfg.end_regress

    # Committed warming: forcing held constant from one year to another,
    # 'n' for none (e.g. end_regress-2300)
    if '--committed' in argv_dict:
        committed_arg = argv_dict['--committed']
    else:
        committed_arg = input(
            'Calculate committed warming? (n or start-end): ')
    if committed_arg == 'n':
        cfg.year_committed_from, cfg.year_committed_to = False, False
    else:
        committed_arg = committed_arg.replace(
            'end_regress', str(cfg.end_regress))
        if '-' not in committed_arg:
            raise ValueError(
                f'Invalid committed warming argument provided: '
                f'{committed_arg}')
        cfg.year_committed_from = int(committed_arg.split('-')[0])
        cfg.year_committed_to = int(committed_arg.split('-')[1])
    # ... as named in the outputs (empty with none)
    cfg.committed_str = (
        f'_COMMITTED--{cfg.year_committed_from}-{cfg.year_committed_to}'
        if cfg.year_committed_to else '')

    # The sampling design: which members of the complete ensemble to
    # rebuild (see sampling.DESIGNS)
    if '--sampling-design' in argv_dict:
        cfg.sampling_design = argv_dict['--sampling-design']
    else:
        cfg.sampling_design = input(
            f'Sampling design ({", ".join(sampling.DESIGNS)}): ')
    if cfg.sampling_design not in sampling.DESIGNS:
        raise ValueError(
            f'Unknown --sampling-design {cfg.sampling_design!r}: use one of '
            f'{", ".join(sampling.DESIGNS)}.')
    # How many members to rebuild, N, as a whole number (e.g. 1000000) or
    # in e notation (e.g. 1e6); none for the all design, which rebuilds
    # every member
    ensemble_size = argv_dict.get('--ensemble-size', '')
    if cfg.sampling_design == 'all':
        if ensemble_size != '':
            raise ValueError(
                f'--ensemble-size={ensemble_size} was given with '
                '--sampling-design=all, which rebuilds every member: leave '
                '--ensemble-size out.')
        cfg.ensemble_size = None
    else:
        if ensemble_size == '':
            ensemble_size = input('Ensemble size (e.g. 1e6): ')
        cfg.ensemble_size = parse_ensemble_size(ensemble_size)

    # The label keeping this piece of work's outputs apart (results_<tag>/);
    # none for the plain results/ folder
    cfg.output_tag = argv_dict.get('--output-tag')

    # The seed of the draws: with one, repeat runs of a configuration draw
    # the same members and give identical results; without (none, or
    # empty), a seed is drawn here from the operating system's entropy, so
    # every run draws afresh, and is logged and named in the outputs, so
    # any run can be reproduced
    seed = argv_dict.get('--seed', '')
    cfg.seed_given = seed != ''
    if not cfg.seed_given:
        cfg.seed = int(np.random.SeedSequence().generate_state(1)[0])
    elif seed.isdigit() and int(seed) < 2**32:
        cfg.seed = int(seed)
    else:
        raise ValueError(
            f'Invalid --seed {seed!r}: use a whole number from 0 to '
            '2**32-1, or leave it empty for a seed drawn afresh.')

    # The warming definitions to give (see WARMING_DEFINITIONS): each gets
    # its own timeseries CSV, and its values at the headline years go into
    # the headlines CSV (the annual timeseries is always written, as the
    # plots need it)
    if '--definitions' in argv_dict:
        definitions_argv = argv_dict['--definitions']
    else:
        definitions_argv = input(
            'Warming definitions? (annual, SR1.5, AR6, CGWL, rates): ')
    cfg.definitions = [d for d in definitions_argv.split(',') if d]
    # Stop if there are none, or any isn't known
    known_definitions = list(WARMING_DEFINITIONS)
    if not cfg.definitions:
        raise ValueError(
            'No warming definitions chosen: choose at least one of '
            f'{known_definitions} (in a recipe, RECIPE_DEFINITIONS), e.g. '
            'annual for the basic annual timeseries.')
    unknown = [d for d in cfg.definitions if d not in known_definitions]
    if unknown:
        raise ValueError(
            f'Unknown warming definition(s) {unknown}: use some of '
            f'{known_definitions}.')

    # Whether FaIR runs the regressed variables' sub-variables too
    cfg.include_sub_vars = argv_dict.get('--include-sub-vars') == 'y'

    # The quantiles' levels (e.g. every 0.1 percentile), or none
    cfg.quantile_step = argv_dict.get('--quantile-step', 'none') or 'none'
    cfg.quantile_levels = parse_quantile_step(cfg.quantile_step)

    # The releases (see make_releases): their sizes, how their members are
    # chosen, and their target (what a stratified release is stratified on)
    cfg.release_sizes = parse_release_sizes(
        argv_dict.get('--release-members', ''))
    cfg.release_method = argv_dict.get('--release-method', '') or 'stratified'
    if cfg.release_method not in release.METHODS:
        raise ValueError(
            f'Unknown --release-method {cfg.release_method!r}: use one of '
            f'{release.METHODS}.')
    cfg.release_target_variable, cfg.release_target_year = (
        parse_release_target(argv_dict.get('--release-target', '')))
    # A stratified release must state its target: no default
    if (cfg.release_sizes and cfg.release_method == 'stratified'
            and cfg.release_target_variable is None):
        raise ValueError(
            'A stratified release needs a target, the variable and year to '
            'stratify on, e.g. --release-target=Ant:2025 (in a recipe, '
            'RECIPE_RELEASE_TARGET_VARIABLE and RECIPE_RELEASE_TARGET_YEAR).')

    # The headline years (e.g. '2024', '2022,2023,2024', 'end_regress',
    # 'IGCC')
    if '--headline-years' in argv_dict:
        headline_years = argv_dict['--headline-years']
    else:
        headline_years = input('Years for headlines: ')
    cfg.headline_years = warming_definitions.check_headlines(
        headline_years, cfg.end_regress, cfg.end_trunc)

    # The regressed variables, sorted
    cfg.regress_vars = sorted(
        argv_dict.get('--regress-variables', 'GHG,Nat,OHF').split(','))

    # The input datasets, by ID (see INPUT DATASETS in
    # src/inputs/input_datasets.py): e.g. --erf-dataset=IGCC-2025
    # --gmt-dataset=HadCRUT-2025 --picontrol-dataset=CMIP6. Stop if any is
    # missing.
    for flag in ['--erf-dataset', '--gmt-dataset', '--picontrol-dataset']:
        if not argv_dict.get(flag):
            raise ValueError(
                f'{flag} is required, e.g. --erf-dataset=IGCC-2025 '
                '--gmt-dataset=HadCRUT-2025 --picontrol-dataset=CMIP6.')
    cfg.erf_id = argv_dict['--erf-dataset']
    cfg.gmt_id = argv_dict['--gmt-dataset']
    cfg.picontrol_id = argv_dict['--picontrol-dataset']

    # The members of each dataset to use: 'all' (the default, also for an
    # empty value), or one, by its label (run-gwi.sh loops over lists)
    cfg.ensemble_members = {
        'ERF': argv_dict.get('--erf-member', 'all') or 'all',
        'GMT': argv_dict.get('--gmt-member', 'all') or 'all',
    }
    # Stop now, before loading anything, if the datasets or members can't
    # be attributed against each other (e.g. a model's ERF with
    # observations)
    datasets.check_dataset_pairing(
        cfg.erf_id, cfg.gmt_id, cfg.ensemble_members['ERF'],
        cfg.ensemble_members['GMT'])
    # ... as named in the outputs, e.g. ERF-all_GMT-7
    cfg.ensemble_members_str = (f'ERF-{cfg.ensemble_members["ERF"]}_'
                                f'GMT-{cfg.ensemble_members["GMT"]}')
    return cfg


def output_folders(cfg):
    """Make the folders this run writes to, and return them.

    cfg:
        the configuration (see read_argv_config)
    Returns a SimpleNamespace of folders, each ending in '/':
    - root:
        the results folder (results/ or results_<tag>/)
    - gwi:
        the attributed warming, by inputs, members, regressed variables and
        regression years
    - priors, erfs:
        by inputs, members and regressed variables (whose forcing variables
        they hold)
    - obs:
        the observations and their timeseries, by inputs and members
    - obs_headlines:
        the observed headlines, by inputs, members and regression years
    - picontrol:
        the piControl segments, by inputs only
    """
    # The results folder, tagged
    root = results_layout.tagged_folder('results', cfg.output_tag)
    # The folder names: the inputs, the members, the regressed variables
    # and the regression years
    inputs_folder = (f'ERF--{cfg.erf_id}_GMT--{cfg.gmt_id}_'
                     f'PREINDUSTRIAL--{cfg.preindustrial_str}'
                     f'{cfg.committed_str}/')
    members_folder = f'ENSEMBLE-MEMBER--{cfg.ensemble_members_str}/'
    variables_folder = f'VARIABLES--{"-".join(cfg.regress_vars)}/'
    years_folder = (f'REGRESSED-YEARS--{cfg.start_regress}-'
                    f'{cfg.end_regress}/')
    folders = types.SimpleNamespace(
        root=f'{root}/',
        gwi=(f'{root}/iterations/{inputs_folder}{members_folder}'
             f'{variables_folder}{years_folder}'),
        priors=(f'{root}/priors/{inputs_folder}{members_folder}'
                f'{variables_folder}'),
        erfs=(f'{root}/erfs/{inputs_folder}{members_folder}'
              f'{variables_folder}'),
        obs=f'{root}/observations/{inputs_folder}{members_folder}',
        obs_headlines=(f'{root}/observations/{inputs_folder}'
                       f'{members_folder}{years_folder}'),
        picontrol=f'{root}/picontrol/{inputs_folder}',
    )
    # Make them all
    for folder in vars(folders).values():
        os.makedirs(folder, exist_ok=True)
    return folders


def load_inputs(cfg):
    """Load the run's inputs, and check they fit the configuration.

    cfg:
        the configuration (see read_argv_config); its output years are
        moved into the forcing years if they reach beyond them
    Returns a SimpleNamespace:
    - erf:
        the forcing, a DataArray (year, erf_member, forcing_variable)
    - gmt:
        the observed members, a DataArray (year, gmt_member)
    - pic:
        the piControl segments, filtered, a DataArray (year, pic_member),
        over the observed years
    - params:
        the FaIR parameter sets, sorted by model (the p axis), a DataArray
        (param_member, parameter, box)
    Each is labelled along every dimension, and its members are in the
    order of their axis (see input_datasets and emulators).
    """
    inputs = types.SimpleNamespace()

    # The forcing, extended to the committed warming's end if asked
    inputs.erf = datasets.load_ERF(
        cfg.erf_id, cfg.regress_vars, cfg.ensemble_members['ERF'],
        include_sub_vars=cfg.include_sub_vars)
    if cfg.year_committed_to and cfg.year_committed_from:
        inputs.erf = datasets.extend_ERF_to_committed_year(
            inputs.erf, cfg.year_committed_to, cfg.year_committed_from)
    # Its years
    forc_Yrs = inputs.erf.year.to_numpy()
    forc_Yrs_min, forc_Yrs_max = forc_Yrs.min(), forc_Yrs.max()

    # The output years, moved into the forcing years if they reach beyond
    if cfg.start_trunc < forc_Yrs_min:
        cfg.start_trunc = forc_Yrs_min
        run_log.note('Truncation start year was before the ERF data range, '
                     f'so it was moved to the ERF minimum: {cfg.start_trunc}.')
    if cfg.end_trunc > forc_Yrs_max:
        cfg.end_trunc = forc_Yrs_max
        run_log.note('Truncation end year was after the ERF data range, so '
                     f'it was moved to the ERF maximum: {cfg.end_trunc}.')
    if cfg.year_committed_to and (cfg.year_committed_to != forc_Yrs_max):
        run_log.note('The ERF data end year does not match the committed '
                     'warming year.')
    if cfg.year_committed_to and (cfg.end_trunc < cfg.year_committed_to):
        run_log.note(f'Truncation end year {cfg.end_trunc} is before the '
                     f'committed warming end year {cfg.year_committed_to}.')

    # The observed temperatures, relative to the pre-industrial period
    inputs.gmt = datasets.load_GMT(
        cfg.gmt_id, cfg.ensemble_members['GMT'], cfg.start_pi, cfg.end_pi)
    temp_Yrs = inputs.gmt.year.to_numpy()

    # Stop if the regression years aren't within both the observations and
    # the forcing (rather than shorten them, which would run a different
    # regression from the one asked for)
    for name, yrs_min, yrs_max in [
            (f'GMT dataset {cfg.gmt_id}', temp_Yrs.min(), temp_Yrs.max()),
            (f'ERF dataset {cfg.erf_id}', forc_Yrs_min, forc_Yrs_max)]:
        if cfg.start_regress < yrs_min or cfg.end_regress > yrs_max:
            raise ValueError(
                f'Regression range {cfg.start_regress}-{cfg.end_regress} is '
                f'not within the {name}, which covers {yrs_min}-{yrs_max}.')
    # Stop if the regression ends after the committed warming starts: from
    # then on the forcing no longer matches the observed temperatures
    if cfg.year_committed_to and (cfg.end_regress > cfg.year_committed_from):
        raise ValueError(
            f'Regression end year ({cfg.end_regress}) cannot be after the '
            f'start of the committed warming period '
            f'({cfg.year_committed_from}). The regression must be performed '
            'on data consistent between the reference/observed temperatures '
            'and forcing data; once the constant ERF assumption begins, the '
            'scenario for forcing and reference warming are no longer '
            'aligned.')

    # The piControl segments: as long as the observed record, indexed by
    # its years, relative to the pre-industrial period, filtered for drift
    # and variability
    inputs.pic = datasets.filter_PiControl(
        datasets.load_PiC(cfg.picontrol_id, temp_Yrs, cfg.start_pi,
                          cfg.end_pi),
        [1, 3, 30])

    # The FaIR parameter sets, sorted by model name: the p axis
    inputs.params = emulators.fair_parameter_sets()
    return inputs


def log_configuration(cfg, inputs, folders):
    """Log the Slurm resources and the run's configuration."""
    compute_resources.log_slurm_configuration()
    run_log.log_section('Run configuration')
    forc_Yrs = inputs.erf.year.to_numpy()
    temp_Yrs = inputs.gmt.year.to_numpy()
    print(f'Git commit: {run_log.describe_git_commit()}')
    print(f'Regressed variables: {cfg.regress_vars}')
    print(f'ERF dataset: {cfg.erf_id}')
    print(f'GMT dataset: {cfg.gmt_id}')
    print(f'piControl dataset: {cfg.picontrol_id}')
    if cfg.year_committed_to:
        print(f'Committed warming: forcing constant from '
              f'{cfg.year_committed_from} to {cfg.year_committed_to}')
    print(f'Forcing range: {forc_Yrs.min()}-{forc_Yrs.max()}')
    print(f'Reference temperature range: {temp_Yrs.min()}-{temp_Yrs.max()}')
    print('Reference temperature realisation number: '
          f'{cfg.ensemble_members["GMT"]}')
    print(f'ERF realisation number: {cfg.ensemble_members["ERF"]}')
    print(f'Pre-industrial era: {cfg.start_pi}-{cfg.end_pi}')
    print(f'Truncation range: {cfg.start_trunc}-{cfg.end_trunc}')
    print(f'Regression range: {cfg.start_regress}-{cfg.end_regress}')
    print(f'Sampling design: {cfg.sampling_design}')
    size_says = (f'{cfg.ensemble_size:,} asked for'
                 if cfg.ensemble_size is not None else 'every member')
    print(f'Ensemble size: {size_says}')
    seed_says = 'given' if cfg.seed_given else 'drawn, as none was given'
    print(f'Random seed: {cfg.seed} ({seed_says})')
    print(f'Warming definitions: {", ".join(cfg.definitions)}')
    quantiles_say = (
        'none' if cfg.quantile_levels is None
        else f'every {cfg.quantile_step} percentile '
             f'({len(cfg.quantile_levels)} levels)')
    print(f'Quantiles: {quantiles_say}')
    print(f'Release: {describe_releases(cfg)}')
    print(f'Include pre-industrial offset: {cfg.inc_pi_offset}')
    print(f'Include constant term in regression: {cfg.inc_reg_const}')
    print('Using model: FaIR_V2')
    print(f'Output folder: {folders.root}'
          f'{"" if cfg.output_tag else "  (no tag set)"}')


def describe_releases(cfg):
    """The releases asked for, in words, e.g. '1000 members, stratified on
    Ant in 2025', or 'none'.

    cfg:
        the configuration (see read_argv_config)
    """
    if not cfg.release_sizes:
        return 'none'
    sizes = ', '.join(f'{size:,}' for size in cfg.release_sizes)
    return f'{sizes} members, {describe_release_method(cfg)}'


def describe_release_method(cfg):
    """How the releases' members are chosen, in words, e.g. 'stratified on
    Ant in 2025', or 'random'.

    cfg:
        the configuration (see read_argv_config)
    """
    if cfg.release_method == 'stratified':
        return (f'stratified on {cfg.release_target_variable} in '
                f'{cfg.release_target_year}')
    return cfg.release_method


def share(part, whole):
    """part as a percentage of whole, for the log: e.g. '0.127%', '100%',
    or '<0.001%' for a share too small to show (but not none).

    part, whole:
        two counts, whole at least 1 (a share of nothing is 0%)
    """
    # The percentage, to three significant figures, unless it is too small
    # to show that way (0.001% and below would print as e.g. 8.11e-05%)
    percent = 100 * part / max(whole, 1)
    if 0 < percent < 0.001:
        return '<0.001%'
    return f'{percent:.3g}%'


def draw_members(cfg, inputs):
    """Draw the members to rebuild, with the configured sampling design.

    cfg:
        the configuration (see read_argv_config)
    inputs:
        the inputs (see load_inputs)
    Returns the member index, (p, e, o, c):
        four arrays of positions along the whole axes, each (N,) (see
        src/method/sampling.py)

    The draws are the run's only random numbers, all from one
    np.random.default_rng(cfg.seed). The design treats the four sources
    alike: e.g. at small N the cube and the Monte Carlo and Latin designs
    may leave out some parameter sets, and the log says how many of each
    source the members use.
    """
    run_log.log_notes()
    run_log.log_section('Ensemble sampling')
    # The sources' sizes, (P, E, O, C): the complete ensemble's axes
    axis_lengths = (
        inputs.params.sizes['param_member'],
        inputs.erf.sizes['erf_member'],
        inputs.gmt.sizes['gmt_member'],
        inputs.pic.sizes['pic_member'])
    n_complete = int(np.prod(axis_lengths, dtype=np.int64))
    print(f'Complete ensemble: {n_complete:,} members (FaIR parameter sets '
          f'x ERF members x observed members x piControl segments = '
          f'{" x ".join(str(n) for n in axis_lengths)})')

    # The members, from the run's one random number generator; the design
    # logs how it drew them (e.g. the cube's k, or latin's swaps)
    rng = np.random.default_rng(cfg.seed)
    member_index = sampling.DESIGNS[cfg.sampling_design](
        axis_lengths, cfg.ensemble_size, rng, report=print)

    # The sample's actual size, against the size asked for and the
    # complete ensemble's
    n_members = len(member_index[0])
    asked_says = (f' ({cfg.ensemble_size:,} asked for)'
                  if cfg.ensemble_size is not None else '')
    print(f'Sampled ensemble: {n_members:,} members{asked_says}, '
          f'{share(n_members, n_complete)} of the size of the complete '
          f'ensemble')
    # How much of the complete ensemble the sample covers, and how much of
    # it is drawn more than once
    n_different, n_repeated = sampling.describe_coverage(member_index,
                                                         axis_lengths)
    print(f'Different members drawn: {n_different:,}, covering '
          f'{share(n_different, n_complete)} of the complete ensemble')
    print(f'Members drawn more than once: {n_repeated:,} '
          f'({share(n_repeated, n_complete)} of the complete ensemble), so '
          f'{n_members - n_different:,} draws '
          f'({share(n_members - n_different, n_members)} of the sample) '
          f'repeat a member already drawn')
    # How many of each source's members the sample uses: a position is
    # used if it appears at all along its axis
    for name, positions, n_axis in zip(
            ('FaIR parameter sets', 'ERF members', 'Observed members',
             'piControl segments'), member_index, axis_lengths):
        n_used = np.count_nonzero(np.bincount(positions, minlength=n_axis))
        print(f'{name} used: {n_used} of {n_axis}')
    return member_index


def get_headline_results_rows(name, timeseries, hl_years, headline_years):
    """A definition's rows for a headlines CSV: its timeseries at the
    headline years, labelled as WARMING_DEFINITIONS labels them.

    name:
        the definition, a key of WARMING_DEFINITIONS
    timeseries:
        its timeseries, indexed by year
    hl_years:
        the headline years (see warming_definitions.generate_headline_years)
    headline_years:
        the --headline-years setting: 'IGCC' adds the definition's IPCC
        years
    Returns a DataFrame of those rows:
        or None if the timeseries has none of the years. Years whose window
        reaches beyond the timeseries' years (e.g. a CGWL year without
        forcing 9 years after it) are left out, with a note.
    """
    definition = WARMING_DEFINITIONS[name]
    # The headline years, and the definition's IPCC years if asked
    years = list(hl_years)
    if headline_years == 'IGCC':
        years += [y for y in definition['ipcc_years'] if y not in years]
    # ... those the timeseries has a value for
    missing = [y for y in years if y not in timeseries.index]
    if missing:
        print(f'({name}: no value for headline year(s) {missing}, whose '
              'window reaches beyond the years available)', end=' ')
    years = [y for y in years if y in timeseries.index]
    if not years:
        return None
    # Its rows at those years, labelled
    return summaries.headline_rows(
        timeseries, years, [definition['label'](year) for year in years])


def definition_attributes(definition):
    """What a warming definition is, as the attributes of its data variable
    in a netCDF file: its long_name, units and window.

    definition:
        an entry of WARMING_DEFINITIONS
    """
    return {'long_name': definition['description'],
            **definition['units'],
            'window': f'{definition["years_before"]} years before each year '
                      f'to {definition["years_after"]} after'}


def attributed_percentiles(sample, cfg):
    """The attributed warming's percentiles: each warming definition's
    timeseries, summarised (SIGMAS) and as quantiles (cfg.quantile_levels),
    and the summary's rows at the headline years.

    sample:
        members.Sample
    cfg:
        the configuration (see read_argv_config)
    Returns (timeseries, headlines, quantiles):
        timeseries: maps each definition's file name (e.g. 'SR15') to its
            summary timeseries, a DataFrame indexed by year: the annual
            definition's always, the others' if asked for
        headlines: the headlines CSV's DataFrame (every asked-for
            definition's rows at the headline years), or None if empty
        quantiles: maps the same file names to the same timeseries at the
            quantiles' levels, each a DataArray (year, percentile,
            output_variable) (see summaries.quantiles_dataarray), or None
            if no quantiles were asked for

    The summary and the quantiles come from one percentile call, at both
    sets of levels together (each percentile is the same, to the last bit,
    as on its own). At 1001 levels that call takes about twice as long as
    at the summary's five, e.g. a block of 24 (year, variable) pairs of a
    million members each takes 0.12 s against 0.05 s on 28 threads.
    """
    # The levels to take the percentiles at (see summaries.py): the
    # summary's, and the quantiles', if asked for
    if cfg.quantile_levels is None:
        levels_taken = SIGMAS
    else:
        levels_taken = np.union1d(SIGMAS, cfg.quantile_levels)
    # The headline years
    hl_years = warming_definitions.generate_headline_years(
        cfg.headline_years, cfg.end_regress, cfg.end_trunc)
    run_log.log_section(f'Definitions ({cfg.headline_years})')
    timeseries, rows, quantiles = {}, [], {}
    for name, definition in WARMING_DEFINITIONS.items():
        # The annual timeseries always; the others if asked for
        if name != 'annual' and name not in cfg.definitions:
            continue
        time_start = dt.datetime.now()
        print(f'Calculating the {name} timeseries', end=' ')
        # Its name in the outputs, e.g. 'SR15'
        file_name = definition['file_name']

        # Every output year whose window fits: each member's value of the
        # definition, rebuilt from T_mod and T_obs with it applied first,
        # and the percentiles across the members at every level taken:
        # (levels_taken, t, variables)
        years, percentiles = summaries.definition_timeseries(
            sample, levels_taken, definition['function'], cfg.start_trunc,
            cfg.end_trunc, definition['years_before'],
            definition['years_after'])

        # The summary: the percentiles at its levels, (SIGMAS, t,
        # variables) ...
        summary = summaries.select_levels(percentiles, levels_taken, SIGMAS)
        # ... as its CSV lays it out
        timeseries[file_name] = summaries.percentiles_as_csv_table(
            summary, SIGMAS, sample.output_vars, years)

        # The quantiles, if asked for: the percentiles at their levels,
        # (quantile levels, t, variables) ...
        if cfg.quantile_levels is not None:
            quantile_values = summaries.select_levels(
                percentiles, levels_taken, cfg.quantile_levels)
            # ... labelled, with what the definition is
            quantiles[file_name] = summaries.quantiles_dataarray(
                quantile_values, cfg.quantile_levels, years,
                sample.output_vars, file_name,
                definition_attributes(definition))

        # The summary's rows at the headline years, if asked for
        if name in cfg.definitions:
            rows.append(get_headline_results_rows(
                name, timeseries[file_name], hl_years, cfg.headline_years))
        print(f'... took {dt.datetime.now() - time_start}')

    # The headlines: every definition's rows in one table (None if none)
    rows = [r for r in rows if r is not None]
    headlines = pd.concat(rows, axis=0) if rows else None
    # The quantiles (None if not asked for)
    if cfg.quantile_levels is None:
        quantiles = None
    return timeseries, headlines, quantiles


def observed_percentiles(gmt, cfg):
    """The observations' percentile summaries: each warming definition
    applied to every observed member, with percentiles across them (no
    rebuild), and their rows at the headline years.

    gmt:
        the observed members, a DataArray (year, gmt_member)
    cfg:
        the configuration (see read_argv_config)
    Returns (timeseries, headlines):
        timeseries: maps the file name of each definition asked for to its
            timeseries
        headlines: the rows of all five definitions at the headline years,
            whatever was asked for (runs with different definitions share
            the observed headlines file, which doesn't depend on the
            regressed variables, so each must write the same contents), or
            None if empty
    """
    # The headline years
    hl_years = warming_definitions.generate_headline_years(
        cfg.headline_years, cfg.end_regress, cfg.end_trunc)
    run_log.log_section(
        f'Definitions for observations ({cfg.headline_years})')
    timeseries, rows = {}, []
    for name, definition in WARMING_DEFINITIONS.items():
        print(f'Calculating the {name} timeseries (Obs)', end=' ')
        # Every year whose window lies within the observed years
        observed = summaries.observed_timeseries_table(
            gmt, SIGMAS, definition['function'],
            definition['years_before'], definition['years_after'])
        # ... kept for its CSV if asked for
        if name in cfg.definitions:
            timeseries[definition['file_name']] = observed
        # Its rows at the headline years
        rows.append(get_headline_results_rows(
            name, observed, hl_years, cfg.headline_years))
        print('... done')
    rows = [r for r in rows if r is not None]
    return timeseries, (pd.concat(rows, axis=0) if rows else None)


def prior_and_erf_percentiles(T_mod, forcing_vars, member_index,
                                        cfg, inputs):
    """The interim results' percentile summaries: the prior warming and the
    forcing.

    T_mod:
        stage A's modelled warming, (p, e, t, v)
    forcing_vars:
        the names along its v axis
    member_index:
        the drawn members (see draw_members), for the log
    cfg:
        the configuration (see read_argv_config)
    inputs:
        the inputs (see load_inputs)
    Returns (priors, erf):
        priors: the prior warming, stage A's modelled warming itself (every
            scaling factor 1), summarised over every (parameter set, ERF
            member) pair, each counted once
        erf: the forcing, summarised over the ERF members
    """
    run_log.log_section('Priors and ERF')
    # How many (parameter set, ERF member) pairs the priors summarise, and
    # how many the sample's members use: each pair as one number, p · E +
    # e, in int64, since the member index is stored in small integer types
    n_params, n_erf = T_mod.shape[:2]
    p, e = member_index[:2]
    pairs_used = np.unique(p.astype(np.int64) * n_erf + e)
    print(f'The priors are every (parameter set, ERF member) pair, '
          f'{n_params} x {n_erf} = {n_params * n_erf:,}, each counted once, '
          f'which may differ from the pairs the attribution samples: its '
          f'members use {len(pairs_used):,} of them.')
    # The two summaries
    priors = summaries.prior_timeseries_table(
        T_mod, forcing_vars, cfg.regress_vars,
        inputs.erf.year.to_numpy(), SIGMAS)
    erf = summaries.erf_timeseries_table(
        inputs.erf, forcing_vars, cfg.regress_vars, SIGMAS)
    return priors, erf


def check_releases(sample, cfg):
    """Stop now, before anything is rebuilt, if the releases asked for
    can't be made: the stratifying variable or year isn't in the outputs,
    a release is bigger than the run, or a file would be too big.

    sample:
        members.Sample
    cfg:
        the configuration (see read_argv_config)
    Logs each release file's size.
    """
    if not cfg.release_sizes:
        return
    run_log.log_section('Release check')
    # What a stratified release is stratified on: one of the outputs'
    # variables, in one of the output years
    if cfg.release_method == 'stratified':
        if cfg.release_target_variable not in sample.output_vars:
            raise ValueError(
                f'The release is stratified on '
                f'{cfg.release_target_variable!r}, which this run '
                f'doesn\'t give: use one of {sample.output_vars}.')
        if not (cfg.start_trunc <= cfg.release_target_year
                <= cfg.end_trunc):
            raise ValueError(
                f'The release is stratified on '
                f'{cfg.release_target_variable} in '
                f'{cfg.release_target_year}, outside the output years '
                f'{cfg.start_trunc}-{cfg.end_trunc}.')
    # Each member's values: every year of every definition given, for
    # every variable
    values_per_member = 0
    for name, definition in WARMING_DEFINITIONS.items():
        if name == 'annual' or name in cfg.definitions:
            years, _ = summaries.definition_years(
                cfg.start_trunc, cfg.end_trunc, definition['years_before'],
                definition['years_after'])
            values_per_member += len(years) * len(sample.output_vars)
    # Each release file: its values as float32, and its members' four
    # labels (about 8 bytes for each member), uncompressed
    for size in cfg.release_sizes:
        if size > sample.n_members:
            raise ValueError(
                f'A release of {size:,} members from a run of '
                f'{sample.n_members:,}: a release can be at most as big as '
                'its run.')
        file_bytes = size * (values_per_member * 4 + 8)
        print(f'Release of {size:,} members: about {file_bytes / 1e6:,.1f} '
              f'MB uncompressed ({values_per_member:,} values per member)')
        if file_bytes > MAX_RELEASE_BYTES:
            raise ValueError(
                f'A release of {size:,} members would be about '
                f'{file_bytes / 1e6:,.0f} MB, more than '
                f'{MAX_RELEASE_BYTES / 1e6:,.0f} MB (MAX_RELEASE_BYTES): '
                'ask for fewer members.')


def make_releases(sample, cfg, inputs):
    """The releases: for each size asked for, M of the run's members chosen
    to be representative (see src/method/release.py), rebuilt over every
    year of every definition given.

    sample:
        members.Sample
    cfg:
        the configuration (see read_argv_config)
    inputs:
        the inputs (see load_inputs), for the members' labels
    Returns a dict mapping each size M to its release, an xarray Dataset:
    one data variable per definition (as the quantiles netCDF), each
    (year, member, output_variable), and along member each member's FaIR
    parameter set, ERF member, observed member and piControl segment, so
    every member can be traced and rebuilt.

    Its memory is within the blocks' (summaries.block_bytes): the
    stratifying values of every member (4 bytes each), their positions in
    rank order (8 bytes each), and one year of the ensemble, from which
    the values are taken.
    """
    run_log.log_section('Release')
    # The stratifying value of every member, for a stratified release: the
    # annual value of the variable in the year: (N,)
    stratify_values = None
    if cfg.release_method == 'stratified':
        year = cfg.release_target_year
        # Every member's annual values in that year: (1, variables, N)
        ensemble = sample.ensemble_definition(
            warming_definitions.annual_value, [(year, year)])
        # ... of the stratifying variable: (N,)
        variable_position = sample.output_vars.index(
            cfg.release_target_variable)
        stratify_values = ensemble[0, variable_position].copy()
        del ensemble
        # Stop if any member has none (e.g. Res after the observations end)
        if np.isnan(stratify_values).any():
            raise ValueError(
                f'{cfg.release_target_variable} has no value in {year} '
                'for some members (e.g. Res after the observations end): '
                'stratify on another variable or year.')

    # The labels of every input's members, to label the released members
    labels = {'param_member': inputs.params.param_member.to_numpy(),
              'erf_member': inputs.erf.erf_member.to_numpy(),
              'gmt_member': inputs.gmt.gmt_member.to_numpy(),
              'pic_member': inputs.pic.pic_member.to_numpy()}

    releases = {}
    for size in cfg.release_sizes:
        time_start = dt.datetime.now()
        # Which members to keep: their positions in the member index, (M,)
        positions = release.select(
            cfg.release_method, sample.n_members, size,
            release.release_rng(cfg.seed, size), stratify_values)
        # Just those members, ready to be rebuilt
        released = sample.subset(positions)

        # Each definition's values for those members, in every year it has
        # a value for, labelled
        data = {}
        for name, definition in WARMING_DEFINITIONS.items():
            if name != 'annual' and name not in cfg.definitions:
                continue
            # The years, and each one's window
            years, windows = summaries.definition_years(
                cfg.start_trunc, cfg.end_trunc, definition['years_before'],
                definition['years_after'])
            # The released members' values: (t, variables, M)
            values = released.ensemble_definition(definition['function'],
                                                  windows)
            # ... members before variables, as the quantiles' percentiles:
            # (t, M, variables)
            values = values.transpose(0, 2, 1)
            data[definition['file_name']] = xr.DataArray(
                values, dims=('year', 'member', 'output_variable'),
                coords={'year': years,
                        'output_variable': list(sample.output_vars)},
                attrs=definition_attributes(definition))

        # Each released member's inputs, by label: (M,) each
        member_coords = {
            'member': np.arange(size),
            **{name: ('member', labels[name][member_positions])
               for name, member_positions in zip(
                   labels, released.member_index)}}
        releases[size] = xr.Dataset(data, coords=member_coords)
        print(f'Release of {size:,} members, {describe_release_method(cfg)}, '
              f'from {sample.n_members:,} ... took '
              f'{dt.datetime.now() - time_start}')
    return releases


def write_inputs_and_interims(cfg, inputs, priors, erf, folders):
    """Write the inputs as used and the interim results: the observed
    temperatures, the piControl segments, the prior warming and the
    forcing.

    cfg:
        the configuration (see read_argv_config)
    inputs:
        the inputs (see load_inputs)
    priors, erf:
        the prior warming's and the forcing's summaries (see
        prior_and_erf_percentiles)
    folders:
        the output folders (see output_folders)

    Each is the same for every iteration of a configuration, so its name
    has no date, and each run overwrites it in one step (see
    results_layout.write_csv). The forcing's file holds its summary, not
    every member (the members are in data/, unchanged).
    """
    # The inputs, as named in the outputs
    inputs_name = (f'ERF--{cfg.erf_id}_GMT--{cfg.gmt_id}_'
                   f'PREINDUSTRIAL--{cfg.preindustrial_str}'
                   f'{cfg.committed_str}')

    variables_name = f'VARIABLES--{"-".join(cfg.regress_vars)}'

    # The sizes: the observed members, piControl segments, (parameter set,
    # ERF member) pairs and ERF members
    n_obs = inputs.gmt.sizes['gmt_member']
    n_pic = inputs.pic.sizes['pic_member']
    n_erf = inputs.erf.sizes['erf_member']
    n_pairs = inputs.params.sizes['param_member'] * n_erf

    # The observed members used, relative to the pre-industrial period: one
    # CSV row per year, one column per member
    results_layout.write_csv(
        inputs.gmt,
        f'{folders.obs}Obs_timeseries-members_{inputs_name}_'
        f'ENSEMBLE-SIZE--{n_obs}.csv', year_heading='Year')

    # The piControl segments, after filtering: one CSV row per year, one
    # column per segment
    results_layout.write_csv(
        inputs.pic,
        f'{folders.picontrol}PiControl_timeseries-members_'
        f'{inputs_name}_ENSEMBLE-SIZE--{n_pic}.csv')

    # The prior warming's summary
    results_layout.write_csv(
        priors,
        f'{folders.priors}PRIOR_timeseries-summary_{inputs_name}_'
        f'{variables_name}_ENSEMBLE-SIZE--{n_pairs}.csv')

    # The forcing's summary
    results_layout.write_csv(
        erf,
        f'{folders.erfs}ERF_timeseries-summary_{inputs_name}_'
        f'{variables_name}_ENSEMBLE-SIZE--{n_erf}.csv')


def write_outputs(cfg, n_members, n_observed_members, attributed, observed,
                  releases, folders):
    """Write the outputs: each warming definition's timeseries and the
    headlines, for the attributed warming and for the observations.

    cfg:
        the configuration (see read_argv_config)
    n_members:
        the sample's size, N, for the attributed outputs' names
    n_observed_members:
        the observed members' number, for the observed outputs' names
    attributed:
        (timeseries, headlines, quantiles) (see
        attributed_percentiles)
    observed:
        (timeseries, headlines) (see observed_percentiles)
    releases:
        each release, by its size (see make_releases)
    folders:
        the output folders (see output_folders)

    Each name says which years the file covers and in what form (see
    OUTPUT NAMES in src/postprocessing/results_layout.py): e.g.
    GWI_timeseries-summary-AR6_, GWI_headlines-summary_,
    GWI_timeseries-quantiles_.

    The attributed outputs' names end in the sample (its design, size and
    seed), which tells the iterations of a configuration apart: a run
    repeated with the same seed draws the same members, and replaces its
    files with the same results. The observed ones are the same for every
    iteration.
    """
    # The names: the inputs, the regression and the sample, kept short
    # since the names are close to the 255-byte limit (see
    # results_layout.sample_name)
    inputs_name = (f'ERF--{cfg.erf_id}_GMT--{cfg.gmt_id}_'
                   f'PREINDUSTRIAL--{cfg.preindustrial_str}'
                   f'{cfg.committed_str}')
    sample_name = results_layout.sample_name(
        cfg.sampling_design, n_members, cfg.seed)
    gwi_name = (f'{inputs_name}_VARIABLES--{"-".join(cfg.regress_vars)}_'
                f'{sample_name}_'
                f'REGRESSED-YEARS--{cfg.start_regress}-{cfg.end_regress}')
    obs_name = f'{inputs_name}_ENSEMBLE-SIZE--{n_observed_members}'
    # The attributed warming: each definition's timeseries, the headlines,
    # the quantiles
    gwi_timeseries, gwi_headlines, gwi_quantiles = attributed
    for file_name, table in gwi_timeseries.items():
        results_layout.write_csv(
            table, f'{folders.gwi}GWI_timeseries-summary-'
            f'{file_name}_{gwi_name}.csv')
    if gwi_headlines is not None:
        results_layout.write_csv(
            gwi_headlines, f'{folders.gwi}GWI_headlines-summary_'
            f'{gwi_name}.csv')
    if gwi_quantiles is not None:
        # What the file is, and the run's configuration
        title = ('Global Warming Index: the attributed warming of each '
                 'warming definition, as percentiles across the '
                 f'{n_members:,} members, every {cfg.quantile_step} percent '
                 'from the 0th to the 100th')
        attributes = {'title': title, 'quantile_step': cfg.quantile_step,
                      **distributions.run_attributes(cfg, n_members)}
        # Every definition's quantiles in one file, one data variable each,
        # on the years of them all (NaN where a definition's window doesn't
        # fit)
        quantiles = xr.Dataset(gwi_quantiles, attrs=attributes)
        distributions.write_netcdf(
            quantiles, f'{folders.gwi}GWI_timeseries-quantiles_'
            f'{gwi_name}.nc')
    # The releases, each with what it is, how to use it, and the run's
    # configuration
    for size, members in releases.items():
        target = (
            f'{cfg.release_target_variable} in {cfg.release_target_year}'
            if cfg.release_method == 'stratified' else 'none')
        title = ('Global Warming Index: the attributed warming of '
                 f'{size:,} representative members ({cfg.release_method}) '
                 f'of the {n_members:,} of this run, every year of each '
                 'warming definition')
        members.attrs = {
            'title': title, 'release_members': size,
            'release_method': cfg.release_method,
            'release_target': target,
            'release_usage': RELEASE_USAGE,
            **distributions.run_attributes(cfg, n_members)}
        name = results_layout.release_name(
            cfg.release_method, size, cfg.release_target_variable,
            cfg.release_target_year)
        distributions.write_netcdf(
            members, f'{folders.gwi}GWI_timeseries-members_{inputs_name}_'
            f'VARIABLES--{"-".join(cfg.regress_vars)}_{sample_name}_{name}_'
            f'REGRESSED-YEARS--{cfg.start_regress}-{cfg.end_regress}.nc')
    # The observations: each definition's timeseries, the headlines
    obs_timeseries, obs_headlines = observed
    for file_name, table in obs_timeseries.items():
        results_layout.write_csv(
            table, f'{folders.obs}Obs_timeseries-summary-'
            f'{file_name}_{obs_name}.csv')
    if obs_headlines is not None:
        results_layout.write_csv(
            obs_headlines, f'{folders.obs_headlines}'
            f'Obs_headlines-summary_{obs_name}.csv')


def main(argv):
    """Run the GWI for the options in argv (see read_argv_config)."""
    time_alpha = dt.datetime.now()

    # Read the configuration, make the output folders, load the inputs
    cfg = read_argv_config(argv)
    folders = output_folders(cfg)
    inputs = load_inputs(cfg)
    log_configuration(cfg, inputs, folders)

    # Draw the members to rebuild
    member_index = draw_members(cfg, inputs)

    run_log.log_section('Stages A, B and C')
    time_start = dt.datetime.now()

    # Stage A: the modelled warming for every parameter set and ERF member,
    # (p, e, t, v)
    T_mod, forcing_vars = modelled_warming(
        erf=inputs.erf, params=inputs.params,
        inc_pi_offset=cfg.inc_pi_offset, start_pi=cfg.start_pi,
        end_pi=cfg.end_pi)

    # Stage B: the scaling factors of every (parameter set, ERF member)
    # against every observed member and piControl segment
    coefs = constrain(
        T_mod=T_mod, forcing_vars=forcing_vars,
        forc_Yrs=inputs.erf.year.to_numpy(),
        gmt=inputs.gmt, pic=inputs.pic,
        regress_vars=cfg.regress_vars, inc_reg_const=cfg.inc_reg_const,
        start_regress=cfg.start_regress, end_regress=cfg.end_regress)

    # Stage C: the drawn members, ready to be rebuilt as each output needs
    # them; the observed members lined up with the forcing years (stage
    # A's t axis), NaN after the observations end: (t, o)
    sample = members.Sample(
        T_mod=T_mod, coefs=coefs,
        T_obs=inputs.gmt.reindex(year=inputs.erf.year).to_numpy(),
        member_index=member_index, forcing_vars=forcing_vars,
        regress_vars=cfg.regress_vars, inc_reg_const=cfg.inc_reg_const,
        years=inputs.erf.year.to_numpy(),
        n_threads=compute_resources.n_workers())
    print(f'Modelling, constraining and sampling ... took '
          f'{dt.datetime.now() - time_start}')

    # Stage D: prepare computing resources to rebuild the ensemble members
    # Stop now if the outputs can't fit, and log what they should need
    expected_memory = compute_resources.check_memory(
        held_bytes=sample.nbytes + coefs.obs.nbytes + coefs.pic.nbytes,
        block_bytes=summaries.block_bytes(sample))
    # How many years of the ensemble each block holds, and what limits it
    by_threads, by_memory = summaries.years_per_block_limits(sample)
    memory_says = ('cannot be told' if by_memory is None
                   else f'allows {by_memory}')
    print(f'Years per block: {summaries.years_per_block(sample)} (the '
          f'threads allow {by_threads}; the memory {memory_says})')
    # Stop now if the releases asked for can't be made
    check_releases(sample, cfg)

    # Stage E: rebuild the ensemble: the percentile summaries of every
    # definition, for the attributed warming and the observations, and of
    # the interim results (the prior warming and the forcing)
    attributed = attributed_percentiles(sample, cfg)
    observed = observed_percentiles(inputs.gmt, cfg)
    priors, erf = prior_and_erf_percentiles(
        T_mod, forcing_vars, member_index, cfg, inputs)
    # The releases: a few representative members, rebuilt in full
    releases = make_releases(sample, cfg, inputs) if cfg.release_sizes else {}

    # Stage F: Write
    # Write the inputs as used and the interim results
    write_inputs_and_interims(cfg, inputs, priors, erf, folders)
    # Write the attribution outputs, and post-processed observation outputs
    write_outputs(cfg, sample.n_members, inputs.gmt.sizes['gmt_member'],
                  attributed, observed, releases, folders)

    # What the run cost, beside the 'Memory usage anticipated' block
    run_log.log_section('Memory usage recorded')
    compute_resources.report_memory_used(expected_memory)
    run_log.log_section('Total')
    print(f'Total time taken: {dt.datetime.now() - time_alpha}')


if __name__ == "__main__":
    main(sys.argv[1:])
