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
    calculate_attributed_percentiles, calculate_observed_percentiles,
    calculate_prior_and_erf_percentiles
                                the percentile summaries
    write_inputs_and_interims   the inputs as used, the priors and the ERF
    write_outputs               each definition's timeseries and the
                                headlines

Each warming definition (annual, SR1.5, AR6, CGWL, rates) is described once,
in WARMING_DEFINITIONS, and applied in the same way to the attributed
warming and to the observations.
"""

import os
import sys
import types

import datetime as dt

import numpy as np
import pandas as pd

import src.method.emulators as emulators
import src.method.members as members
import src.method.regression as regression
import src.method.sampling as sampling
import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.infrastructure.run_log as run_log
import src.method.warming_definitions as warming_definitions
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
        'ipcc_years': [2017],
    },
    'SR1.5': {
        'file_name': 'SR15',
        'function': warming_definitions.final_value_of_trend_vectorised,
        'years_before': SR15_TREND_YEARS - 1,
        'years_after': 0,
        'label': lambda year: f'{year} (SR15 definition)',
        'ipcc_years': [2017],
    },
    'AR6': {
        'file_name': 'AR6',
        'function': warming_definitions.mean_over_years,
        'years_before': AR6_MEAN_YEARS - 1,
        'years_after': 0,
        'label': lambda year: f'{year - (AR6_MEAN_YEARS - 1)}-{year}',
        'ipcc_years': [2019],
    },
    'CGWL': {
        'file_name': 'CGWL',
        'function': warming_definitions.mean_over_years,
        'years_before': HALF_CGWL,
        'years_after': HALF_CGWL,
        'label': lambda year: (f'{year - HALF_CGWL}-{year + HALF_CGWL} '
                               '(CGWL definition)'),
        'ipcc_years': [],
    },
    'rates': {
        'file_name': 'RATES',
        'function': warming_definitions.rate_func_vectorised,
        'years_before': AR6_RATE_YEARS - 1,
        'years_after': 0,
        'label': lambda year: (f'{year - (AR6_RATE_YEARS - 1)}-{year} '
                               '(AR6 rate definition)'),
        'ipcc_years': [2019],
    },
}

# Every option gwi.py reads. Anything else on the command line stops the
# run: an unknown option is most likely a typo (e.g. --sample=10000),
# which would otherwise be ignored silently, leaving the default in its
# place.
KNOWN_OPTIONS = (
    '--samples', '--seed', '--regress-range', '--truncate',
    '--definitions', '--headline-years',
    '--regress-variables', '--erf-dataset', '--gmt-dataset',
    '--picontrol-dataset', '--committed', '--preindustrial-era',
    '--include-reg-const', '--erf-member', '--gmt-member',
    '--include-sub-vars', '--output-tag',
)


###############################################################################
# STAGES A AND B ##############################################################
###############################################################################
def modelled_warming(df_forc, df_params, inc_pi_offset, start_pi, end_pi):
    """Stage A: the modelled warming from each forcing variable alone,

        T_mod[p, e, t, v]

    for every FaIR parameter set p, every ERF member e, every forcing year t
    and every forcing variable v. Both the attribution (stages B and C) and
    the priors are built from it.

    df_forc:
        forcing, columns (variable, ensemble), for every ERF member, in the
        order of the e axis
    df_params:
        FaIR thermal parameter sets, columns (set name, parameter), in the
        order of the p axis
    inc_pi_offset:
        whether to express the modelled warming relative to its mean over the
        pre-industrial years, start_pi to end_pi (both ends included); without
        it, it is relative to the first forcing year
    Returns (T_mod, forcing_vars):
        T_mod: an array (p, e, t, v) over the forcing years (df_forc's index)
        forcing_vars: the names along v, sorted.
    """
    # The forcing variables, sorted: the v axis
    forcing_vars = sorted(df_forc.columns.get_level_values(
        "variable").unique().to_list())
    # Every ERF member, in order: the e axis
    erf_members = df_forc.columns.get_level_values(
        "ensemble").unique().to_list()
    # The forcing years: the t axis
    forc_Yrs = df_forc.index.to_numpy()

    # The modelled warming, for every parameter set and every ERF member:
    # (p, e, t, v)
    T_mod = emulators.run_fair_v2_alpha(
        emulators.forcing_array(df_forc, forcing_vars, erf_members),
        forc_Yrs, df_params)

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


def constrain(T_mod, forcing_vars, forc_Yrs, df_temp_Obs, df_temp_PiC,
              regress_vars, inc_reg_const, start_regress, end_regress):
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
    df_temp_Obs:
        every observed GMT member (columns, in the order of the o axis),
        indexed by year
    df_temp_PiC:
        every piControl segment (columns, in the order of the c axis),
        indexed by the same years
    regress_vars:
        the regressed variables, e.g. ['GHG', 'Nat', 'OHF']
    inc_reg_const:
        whether the regression has a constant term
    start_regress, end_regress:
        the regression years (both included)
    Returns regression.Coefficients:
        obs (p, e, o, k) and pic (p, e, c, k).
    """
    # The years of the observations
    temp_Yrs = df_temp_Obs.index.to_numpy()  # observed GMT
    PiC_Yrs = df_temp_PiC.index.to_numpy()  # piControl segments

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
    Y_obs = df_temp_Obs.to_numpy()[
        (temp_Yrs >= start_regress) & (temp_Yrs <= end_regress)]
    # The piControl segments over the regression years: (t, c)
    Y_pic = df_temp_PiC.to_numpy()[
        (PiC_Yrs >= start_regress) & (PiC_Yrs <= end_regress)]
    # Every fit at once: coefs.obs (p, e, o, k), coefs.pic (p, e, c, k)
    coefs = regression.ols(X, Y_obs, Y_pic)
    return coefs


###############################################################################
# THE PIPELINE ################################################################
###############################################################################

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
        samples, seed
    - outputs:
        definitions, headline_years, include_sub_vars, output_tag
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
                '--name=value (e.g. --samples=10).')
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

    # How many members to draw from each source (the whole source if it has
    # fewer)
    if '--samples' in argv_dict:
        cfg.samples = int(argv_dict['--samples'])
    else:
        cfg.samples = int(input(
            'Max number of samples for each source (int): '))

    # The label keeping this piece of work's outputs apart (results_<tag>/);
    # none for the plain results/ folder
    cfg.output_tag = argv_dict.get('--output-tag')

    # The seed of the draws: with one, repeat runs of a configuration draw
    # the same members and give identical results; without (none, or
    # empty), every run draws afresh
    seed = argv_dict.get('--seed', '')
    if seed == '':
        cfg.seed = None
    elif seed.isdigit() and int(seed) < 2**32:
        cfg.seed = int(seed)
    else:
        raise ValueError(
            f'Invalid --seed {seed!r}: use a whole number from 0 to '
            '2**32-1, or leave it empty for unseeded sampling.')

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
    - df_forc:
        the forcing: columns (variable, ensemble), indexed by year
    - df_temp_Obs:
        the observed members, indexed by year
    - df_temp_PiC:
        the piControl segments, filtered, indexed by the same years
    - df_params:
        the FaIR parameter sets, sorted by model: the p axis
    - models:
        their names
    """
    inputs = types.SimpleNamespace()

    # The forcing, extended to the committed warming's end if asked
    inputs.df_forc = datasets.load_ERF(
        cfg.erf_id, cfg.regress_vars, cfg.ensemble_members['ERF'],
        include_sub_vars=cfg.include_sub_vars)
    if cfg.year_committed_to and cfg.year_committed_from:
        inputs.df_forc = datasets.extend_ERF_to_committed_year(
            inputs.df_forc, cfg.year_committed_to, cfg.year_committed_from)
    # Its years
    forc_Yrs = inputs.df_forc.index.to_numpy()
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
    inputs.df_temp_Obs = datasets.load_GMT(
        cfg.gmt_id, cfg.ensemble_members['GMT'], cfg.start_pi, cfg.end_pi)
    temp_Yrs = inputs.df_temp_Obs.index.to_numpy()

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
    inputs.df_temp_PiC = datasets.filter_PiControl(
        datasets.load_PiC(cfg.picontrol_id, temp_Yrs, cfg.start_pi,
                          cfg.end_pi),
        [1, 3, 30])

    # The FaIR parameter sets, sorted by model name: the p axis
    inputs.df_params, inputs.models = emulators.fair_parameter_sets()
    return inputs


def log_configuration(cfg, inputs, folders):
    """Log the Slurm resources and the run's configuration."""
    compute_resources.log_slurm_configuration()
    run_log.log_section('Run configuration')
    forc_Yrs = inputs.df_forc.index
    temp_Yrs = inputs.df_temp_Obs.index
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
    print(f'Number of samples: {cfg.samples}')
    seed_says = (cfg.seed if cfg.seed is not None
                 else 'none (unseeded sampling)')
    print(f'Random seed: {seed_says}')
    print(f'Warming definitions: {", ".join(cfg.definitions)}')
    print(f'Include pre-industrial offset: {cfg.inc_pi_offset}')
    print(f'Include constant term in regression: {cfg.inc_reg_const}')
    print('Using model: FaIR_V2')
    print(f'Output folder: {folders.root}'
          f'{"" if cfg.output_tag else "  (no tag set)"}')


def draw_members(cfg, inputs):
    """Draw the members to rebuild: every FaIR parameter set, and a random
    subset of up to cfg.samples of each other source (the cube design).

    cfg:
        the configuration (see read_argv_config)
    inputs:
        the inputs (see load_inputs)
    Returns the member index, (p, e, o, c):
        four arrays of positions along the whole axes, each (N,) (see
        sampling.cube)

    Every parameter set is used, since there are few and their uncertainty
    should be spanned fully; the ERF members, observed members and
    piControl segments are drawn in equal numbers, to spread the
    uncertainty evenly across the sources. The draws are
    np.random.choice(n, k, replace=False) on numpy's global random state,
    seeded here if asked: the draws the legacy code made, so the same seed
    draws the same members, in the same order.
    """
    # Seed the draws, if asked: they are the run's only random draws
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    run_log.log_notes()
    run_log.log_section('Ensemble sampling')
    # The sizes of the sources: E, P, O, C
    n_erf = inputs.df_forc.columns.get_level_values('ensemble').nunique()
    n_params = len(inputs.models)
    n_obs = inputs.df_temp_Obs.shape[1]
    n_pic = inputs.df_temp_PiC.shape[1]

    # 1. A random subset of the ERF members: positions along e
    print(f'Forcing ensemble all: {n_erf}')
    erf_drawn = np.random.choice(n_erf, min(cfg.samples, n_erf),
                                 replace=False)
    print(f'Forcing ensemble pruned: {len(erf_drawn)}')
    # 2. Every parameter set: positions along p
    print(f'FaIR Parameters: {n_params}')
    params_drawn = np.arange(n_params)
    # 3. A random subset of the observed members: positions along o
    print(f'Temperature ensembles all: {n_obs}')
    obs_drawn = np.random.choice(n_obs, min(cfg.samples, n_obs),
                                 replace=False)
    print(f'Temperature ensembles pruned: {len(obs_drawn)}')
    # 4. A random subset of the piControl segments: positions along c
    print(f'Internal variability ensembles all: {n_pic}')
    pic_drawn = np.random.choice(n_pic, min(cfg.samples, n_pic),
                                 replace=False)
    print(f'Internal variability ensembles pruned: {len(pic_drawn)}')

    # Every combination of the drawn members (the cube design)
    member_index = sampling.cube(params_drawn, erf_drawn, obs_drawn,
                                 pic_drawn)
    # The full ensemble's size, and the sample's
    n_all = n_erf * n_params * n_obs * n_pic
    n_sub = len(member_index[0])
    print(f'Max available ensemble: {n_all:,}')
    print(f'Sub-sampled ensemble size: {n_sub:,} '
          f'({n_sub / n_all:.2%} of all possible combinations)')
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


def calculate_attributed_percentiles(sample, cfg):
    """The attributed warming's percentile summaries: each warming
    definition's timeseries, and their rows at the headline years.

    sample:
        members.Sample
    cfg:
        the configuration (see read_argv_config)
    Returns (timeseries, headlines):
        timeseries: maps each definition's file name (e.g. 'SR15') to its
            timeseries, a DataFrame indexed by year: the annual definition's
            always, the others' if asked for
        headlines: the headlines CSV's DataFrame (every asked-for
            definition's rows at the headline years), or None if empty
    """
    # The headline years
    hl_years = warming_definitions.generate_headline_years(
        cfg.headline_years, cfg.end_regress, cfg.end_trunc)
    run_log.log_section(f'Definitions ({cfg.headline_years})')
    timeseries, rows = {}, []
    for name, definition in WARMING_DEFINITIONS.items():
        # The annual timeseries always; the others if asked for
        if name != 'annual' and name not in cfg.definitions:
            continue
        time_start = dt.datetime.now()
        print(f'Calculating the {name} timeseries', end=' ')
        # Every output year whose window fits: each member's value of the
        # definition, rebuilt from T_mod and T_obs with it applied first,
        # and the percentiles across the members
        timeseries[definition['file_name']] = (
            summaries.definition_timeseries_table(
                sample, SIGMAS, definition['function'], cfg.start_trunc,
                cfg.end_trunc, definition['years_before'],
                definition['years_after']))
        # Its rows at the headline years, if asked for
        if name in cfg.definitions:
            rows.append(get_headline_results_rows(
                name, timeseries[definition['file_name']], hl_years,
                cfg.headline_years))
        print(f'... took {dt.datetime.now() - time_start}')
    rows = [r for r in rows if r is not None]
    return timeseries, (pd.concat(rows, axis=0) if rows else None)


def calculate_observed_percentiles(df_temp_Obs, cfg):
    """The observations' percentile summaries: each warming definition
    applied to every observed member, with percentiles across them (no
    rebuild), and their rows at the headline years.

    df_temp_Obs:
        the observed members, indexed by year
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
            df_temp_Obs, SIGMAS, definition['function'],
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


def calculate_prior_and_erf_percentiles(T_mod, forcing_vars, member_index,
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
        inputs.df_forc.index.to_numpy(), SIGMAS)
    erf = summaries.erf_timeseries_table(
        inputs.df_forc, forcing_vars, cfg.regress_vars, SIGMAS)
    return priors, erf


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
        calculate_prior_and_erf_percentiles)
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
    n_obs = inputs.df_temp_Obs.shape[1]
    n_pic = inputs.df_temp_PiC.shape[1]
    n_erf = inputs.df_forc.columns.get_level_values('ensemble').nunique()
    n_pairs = len(inputs.models) * n_erf

    # The observed members used, relative to the pre-industrial period
    results_layout.write_csv(
        inputs.df_temp_Obs,
        f'{folders.obs}Obs_timeseries_{inputs_name}_'
        f'ENSEMBLE-SIZE--{n_obs}.csv')

    # The piControl segments, after filtering
    results_layout.write_csv(
        inputs.df_temp_PiC,
        f'{folders.picontrol}PiControl_timeseries_{inputs_name}_'
        f'ENSEMBLE-SIZE--{n_pic}.csv')

    # The prior warming's summary
    results_layout.write_csv(
        priors,
        f'{folders.priors}PRIOR_results_timeseries_{inputs_name}_'
        f'{variables_name}_ENSEMBLE-SIZE--{n_pairs}_.csv')

    # The forcing's summary
    results_layout.write_csv(
        erf,
        f'{folders.erfs}ERF_results_timeseries_{inputs_name}_'
        f'{variables_name}_ENSEMBLE-SIZE--{n_erf}_.csv')


def write_outputs(cfg, n_members, n_observed_members, attributed, observed,
                  folders):
    """Write the outputs: each warming definition's timeseries and the
    headlines, for the attributed warming and for the observations.

    cfg:
        the configuration (see read_argv_config)
    n_members:
        the sample's size, N, for the attributed outputs' names
    n_observed_members:
        the observed members' number, for the observed outputs' names
    attributed, observed:
        each (timeseries, headlines) (see calculate_attributed_percentiles
        and calculate_observed_percentiles)
    folders:
        the output folders (see output_folders)

    The attributed outputs' names end in the time of the run, since
    iterations of a configuration differ only in it; the observed ones are
    the same for every iteration, so have no date.
    """
    # The names: the inputs, the regression, the sample's size and the
    # time of the run
    inputs_name = (f'ERF--{cfg.erf_id}_GMT--{cfg.gmt_id}_'
                   f'PREINDUSTRIAL--{cfg.preindustrial_str}'
                   f'{cfg.committed_str}')
    run_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    gwi_name = (f'{inputs_name}_VARIABLES--{"-".join(cfg.regress_vars)}_'
                f'ENSEMBLE-SIZE--{n_members}_'
                f'REGRESSED-YEARS--{cfg.start_regress}-{cfg.end_regress}_'
                f'DATE-CALCULATED--{run_time}')
    obs_name = f'{inputs_name}_ENSEMBLE-SIZE--{n_observed_members}'
    # The attributed warming: each definition's timeseries, the headlines
    gwi_timeseries, gwi_headlines = attributed
    for file_name, table in gwi_timeseries.items():
        results_layout.write_csv(
            table, f'{folders.gwi}GWI_results_timeseries-{file_name}_'
            f'{gwi_name}.csv')
    if gwi_headlines is not None:
        results_layout.write_csv(
            gwi_headlines, f'{folders.gwi}GWI_results_headlines_'
            f'{gwi_name}.csv')
    # The observations: each definition's timeseries, the headlines
    obs_timeseries, obs_headlines = observed
    for file_name, table in obs_timeseries.items():
        results_layout.write_csv(
            table, f'{folders.obs}Obs_results_timeseries-{file_name}_'
            f'{obs_name}.csv')
    if obs_headlines is not None:
        results_layout.write_csv(
            obs_headlines, f'{folders.obs_headlines}Obs_results_headlines_'
            f'{obs_name}.csv')


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
        df_forc=inputs.df_forc, df_params=inputs.df_params,
        inc_pi_offset=cfg.inc_pi_offset, start_pi=cfg.start_pi,
        end_pi=cfg.end_pi)

    # Stage B: the scaling factors of every (parameter set, ERF member)
    # against every observed member and piControl segment
    coefs = constrain(
        T_mod=T_mod, forcing_vars=forcing_vars,
        forc_Yrs=inputs.df_forc.index.to_numpy(),
        df_temp_Obs=inputs.df_temp_Obs, df_temp_PiC=inputs.df_temp_PiC,
        regress_vars=cfg.regress_vars, inc_reg_const=cfg.inc_reg_const,
        start_regress=cfg.start_regress, end_regress=cfg.end_regress)

    # Stage C: the drawn members, ready to be rebuilt as each output needs
    # them; the observed members lined up with the forcing years (stage
    # A's t axis), NaN after the observations end: (t, o)
    sample = members.Sample(
        T_mod=T_mod, coefs=coefs,
        T_obs=inputs.df_temp_Obs.reindex(inputs.df_forc.index).to_numpy(),
        member_index=member_index, forcing_vars=forcing_vars,
        regress_vars=cfg.regress_vars, inc_reg_const=cfg.inc_reg_const,
        years=inputs.df_forc.index.to_numpy(),
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

    # Stage E: rebuild the ensemble: the percentile summaries of every
    # definition, for the attributed warming and the observations, and of
    # the interim results (the prior warming and the forcing)
    attributed = calculate_attributed_percentiles(sample, cfg)
    observed = calculate_observed_percentiles(inputs.df_temp_Obs, cfg)
    priors, erf = calculate_prior_and_erf_percentiles(
        T_mod, forcing_vars, member_index, cfg, inputs)

    # Stage F: Write
    # Write the inputs as used and the interim results
    write_inputs_and_interims(cfg, inputs, priors, erf, folders)
    # Write the attribution outputs, and post-processed observation outputs
    write_outputs(cfg, sample.n_members, inputs.df_temp_Obs.shape[1],
                  attributed, observed, folders)

    # What the run cost, beside the 'Memory usage anticipated' block
    run_log.log_section('Memory usage recorded')
    compute_resources.report_memory_used(expected_memory)
    run_log.log_section('Total')
    print(f'Total time taken: {dt.datetime.now() - time_alpha}')


if __name__ == "__main__":
    main(sys.argv[1:])
