"""Script to generate global warming index."""

import os
import sys

import datetime as dt
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# import scipy.stats as ss

import src.method.attribution as attribution
import src.method.emulators as emulators
import src.method.members as members
import src.method.regression as regression
import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.infrastructure.run_log as run_log
import src.method.variables as variables
import src.method.warming_definitions as warming_definitions
import src.inputs.input_datasets as datasets

import models.AR5_IR.AR5_IR as AR5_IR


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def GWI_faster(
        model_choice, inc_reg_const, inc_pi_offset,
        df_forc, df_params, df_temp_PiC, df_temp_Obs,
        start_trunc, end_trunc, start_pi, end_pi,
        start_regress, end_regress, regress_vars, out=None):
    """Calculate the global warming index (GWI).

    If `out` is given, results are written into it instead of into a newly
    allocated array. That is how the parallel run avoids copying: `out` is
    one parameterisation's slice of a single array shared by all the worker
    processes, so nothing has to be handed back. See
    compute_resources.shared_results_array.
    """
    """Parallelise over FaIR parameterisations, exploit vectorisation of
    FaIR model by running all forcings at once through it, and separate
    regression against observations and piControl to add rather than multiply
    linear regressions' computational time."""

    # Preparing lists to ensure that order of variables and ensemble members
    # are consistent across the different dataframes. I'm pretty sure that
    # pandas keeps column order consistent, but this is just extra safety
    var_list_ERF = sorted(df_forc.columns.get_level_values(
        "variable").unique().to_list())
    ens_list_ERF = df_forc.columns.get_level_values(
        "ensemble").unique().to_list()
    ens_list_Obs = df_temp_Obs.columns.to_list()
    ens_list_PiC = df_temp_PiC.columns.to_list()

    # Prepare results #########################################################
    # Total sub-ensemble size: multiple number of ensemble members for each of:
    # HadCRUT sub-ensemble * piControl sub-ensemble * ERF sub-ensemble
    # Specify only 1 for the number of FaIR paramaterisations, as we are
    # parallelising across the parameterisations, and therefore only have one
    # for each instance of the function call.
    n = (len(ens_list_Obs) * len(ens_list_PiC) * len(ens_list_ERF) * 1)

    # The variables along the results array's variable axis: the forcing
    # variables (which may include sub-variables), then the variables
    # diagnosed after the regression (Ant, Tot, Res) that are not among them.
    output_vars = attribution.attribution_output_vars(df_forc, regress_vars)

    shape = (end_trunc - start_trunc + 1,  # number of years (after truncation)
             len(output_vars),  # variables dimension
             n)  # samples
    if out is None:
        temp_Att_Results = np.empty(
            shape,
            dtype=np.float32  # make the array smaller in memory
            )
    elif out.shape != shape:
        # The caller sized the shared array from the same inputs, so a
        # mismatch means those inputs disagree with what this emulation
        # actually produces. Stop rather than write into the wrong columns.
        raise ValueError(
            f'Emulation of {model_choice} produces an array of shape {shape}, '
            f'but was given somewhere of shape {out.shape} to write it.')
    else:
        temp_Att_Results = out

    # The years of each input
    forc_Yrs = df_forc.index.to_numpy()  # forcing
    temp_Yrs = df_temp_Obs.index.to_numpy()  # observed GMT
    PiC_Yrs = df_temp_PiC.index.to_numpy()  # piControl segments

    # Stage A: the modelled warming from each forcing variable alone, for
    # every ERF member, with the one parameter set model_choice. The adapter
    # runs every parameter set in the table it is given, in the table's
    # order, and returns them along the first axis (p) of T_mod, shape
    # (p, e, t, v). Given only model_choice's columns, it runs just that set:
    # p has length 1, and T_mod[0] is the warming with parameter set
    # model_choice.
    params_this_set = df_params[[model_choice]]
    T_mod = emulators.run_fair_v2_alpha(
        emulators.forcing_array(df_forc, var_list_ERF, ens_list_ERF),
        forc_Yrs, params_this_set)
    # Take that one parameter set, and reorder the axes from (e, t, v) to
    # the (t, v, e) the code below expects: (years, variables, ERF members).
    temp_Mod_array_all_years = T_mod[0].transpose(1, 2, 0)

    # Express the modelled warming relative to its pre-industrial mean, if
    # asked to: each ERF member and variable relative to its own mean over
    # the pre-industrial years (both included).
    if inc_pi_offset:
        # Which forcing years are pre-industrial
        in_pi = (forc_Yrs >= start_pi) & (forc_Yrs <= end_pi)
        # Subtract each (variable, ERF member)'s mean over those years:
        # (t, v, e) minus (v, e), broadcast over the years
        temp_Mod_array_all_years = (
            temp_Mod_array_all_years -
            temp_Mod_array_all_years[in_pi].mean(axis=0))

    # Stage B: the scaling factors. Every observed GMT member, and every
    # piControl segment, is fitted onto each ERF member's modelled warming,
    # all in one batched solve. The regressors X are the regressed variables
    # (plus a column of ones, if a constant is used) over the regression
    # years, for each ERF member: shape (e, t, k).

    # Where each regressed variable sits on the variable axis
    reg_indices = [var_list_ERF.index(var) for var in regress_vars]
    # Which forcing years are in the regression
    in_regress_forc = (forc_Yrs >= start_regress) & (forc_Yrs <= end_regress)
    # The regressed variables over the regression years: (t, k, e)
    X = temp_Mod_array_all_years[in_regress_forc][:, reg_indices, :]
    # Put the ERF members first, one regressor matrix each: (e, t, k)
    X = X.transpose(2, 0, 1)
    if inc_reg_const:
        # Add a column of ones to every matrix, for the constant: k + 1
        X = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    # The observed GMT members over the regression years: (t, o)
    Y_obs = df_temp_Obs[ens_list_Obs].to_numpy()[
        (temp_Yrs >= start_regress) & (temp_Yrs <= end_regress)]
    # The piControl segments over the regression years: (t, c)
    Y_pic = df_temp_PiC[ens_list_PiC].to_numpy()[
        (PiC_Yrs >= start_regress) & (PiC_Yrs <= end_regress)]
    # Every fit at once: coefs.obs (e, o, k), coefs.pic (e, c, k)
    coefs = regression.ols(X, Y_obs, Y_pic)

    # Stage C: rebuild every member (e, o, c) of this parameter set from the
    # pieces, and write it into the results array.

    # Which forcing years are in the output (the truncation years)
    in_trunc_forc = (forc_Yrs >= start_trunc) & (forc_Yrs <= end_trunc)
    # The modelled warming over those years, with a parameter-set axis of
    # length 1 added in front: (t, v, e) -> (p, e, t, v)
    T_mod_trunc = temp_Mod_array_all_years[in_trunc_forc].transpose(
        2, 0, 1)[np.newaxis]
    # The scaling factors, with the same parameter-set axis added:
    # obs (p, e, o, k) and pic (p, e, c, k)
    coefs = regression.Coefficients(obs=coefs.obs[np.newaxis],
                                    pic=coefs.pic[np.newaxis])
    # The observed members over the output years, NaN in years after the
    # observations end: (t, o)
    T_obs_trunc = df_temp_Obs[ens_list_Obs].reindex(
        np.arange(start_trunc, end_trunc + 1)).to_numpy()

    # The members for one ERF member: every (o, c) pair, with c varying
    # fastest, the order of the results array's member axis
    n_obs, n_pic = len(ens_list_Obs), len(ens_list_PiC)
    o_index = np.repeat(np.arange(n_obs), n_pic)
    c_index = np.tile(np.arange(n_pic), n_obs)
    # The parameter-set index: always 0, the only one here
    p_index = np.zeros(n_obs * n_pic, dtype=int)
    # One ERF member at a time, to keep the memory small
    for ens in range(len(ens_list_ERF)):
        # This ERF member's index, for each of its members
        e_index = np.full(n_obs * n_pic, ens)
        # Its block of the member axis: members ens·o·c to (ens+1)·o·c
        block = slice(ens * n_obs * n_pic, (ens + 1) * n_obs * n_pic)
        # Rebuild its members, and write them into their block
        temp_Att_Results[:, :, block] = members.rebuild(
            T_mod_trunc, coefs, T_obs_trunc,
            (p_index, e_index, o_index, c_index),
            var_list_ERF, regress_vars, inc_reg_const, output_vars)

    return temp_Att_Results, output_vars


# The three things below are filled in further down, in the main block, just
# before the worker processes are created. The workers inherit them as they
# are forked, which is how each one reaches the shared results array without
# anything being sent to it.
emulation_models = None      # the FaIR parameterisations, in order
emulation_settings = None    # the arguments every emulation shares
emulation_results = None     # the array they all write into


def emulate_one_model(model_number):
    """Run one FaIR parameterisation, in a worker process.

    The emulations are laid out side by side along the ensemble axis of the
    shared results array, in the order of `emulation_models`: the first
    parameterisation fills the first block of columns, the second fills the
    next, and so on. This function works out which columns belong to
    `model_number` and hands that slice of the array to GWI_faster to fill.

    `model_number` is the only thing the parent sends; everything large was
    inherited when this process was forked. Nothing is returned, because the
    results are already in the array the parent is holding.
    """
    n_per_model = emulation_results.shape[2] // len(emulation_models)
    my_columns = slice(model_number * n_per_model,
                       (model_number + 1) * n_per_model)
    GWI_faster(emulation_models[model_number],
               out=emulation_results[:, :, my_columns],
               **emulation_settings)


###############################################################################
# MAIN CODE BODY ##############################################################
###############################################################################

if __name__ == "__main__":

    time_alpha = dt.datetime.now()

    # Percentiles to calculate and use throughout analysis.
    # sigmas = [[32, 68], [5, 95], [0.3, 99.7]]
    # These are the percentile ranges used in the IPCC likelihood statements:
    # Typically, 17-83 is likely range, and 5-95 is very likely range.
    sigmas = [[17, 83], [5, 95]]
    sigmas_all = list(
        np.concatenate((np.sort(np.ravel(sigmas)), [50]), axis=0)
        )

    # model_choice = 'AR5_IR'
    model_choice = 'FaIR_V2'

    # Use command line arguments instead of interactivity, so that we can
    # run the script using nohup and later slurm.

    # TODO: switch to using docopt for command line arguments.
    # For now just use sys.argv while HPC conda is having installation issues.

    # argv format:
    # python gwi.py
    # --samples=<n> (default interactive)
    # --preindustrial-era=<start-end> (default 1850-1900)
    # --regress-range=<start-end> (default 1850-2023)
    # --truncate=<start-end> (default 1850-2023)
    # --include-rate=<y/n> (default interactive)
    # --erf-dataset=<id> (required, e.g. IGCC-2025)
    # --gmt-dataset=<id> (required, e.g. HadCRUT-2025)
    # --picontrol-dataset=<id> (required, e.g. CMIP6)
    # --erf-member=<member or all> (default all)
    # --gmt-member=<member or all> (default all)
    # (every option is listed in KNOWN_OPTIONS below; run-gwi.sh passes them
    # all)

    # Every option gwi.py reads. Anything else on the command line stops the
    # run: an unknown option is most likely a typo (e.g. --sample=10000),
    # which would otherwise be ignored silently, leaving the default in its
    # place.
    KNOWN_OPTIONS = (
        '--samples', '--seed', '--regress-range', '--truncate',
        '--include-rate', '--headline-toggles', '--headline-years',
        '--regress-variables', '--erf-dataset', '--gmt-dataset',
        '--picontrol-dataset', '--committed', '--preindustrial-era',
        '--include-reg-const', '--erf-member', '--gmt-member',
        '--include-sub-vars', '--calculate-priors-output', '--output-tag',
    )

    # Separate out the name and value of each option, --name=value, into a
    # dictionary for later use. Options left out are absent from it (most
    # then fall back to a default, or to asking interactively).
    argv_dict = {}
    for argv in sys.argv[1:]:
        name, equals, value = argv.partition('=')
        if not equals:
            raise ValueError(
                f'Invalid argument {argv!r}: give options as --name=value '
                '(e.g. --samples=10).')
        if name not in KNOWN_OPTIONS:
            raise ValueError(
                f'Unknown option {name!r}. The options are: '
                f'{", ".join(KNOWN_OPTIONS)}.')
        if name in argv_dict:
            raise ValueError(f'Option {name!r} is given more than once.')
        argv_dict[name] = value

    # Define pre-industrial period for temperature offset.
    # The average of this period is used as the offset, as standard in IPCC.
    # Both years are included: 1850-1900 is 51 years (IPCC SR1.5, Ch. 1).
    # This offset is applied to:
    # 1. Modelled output temperatures from FaIR
    # 2. Observed/Reference GMT dataset (df_temp_Obs)
    # 3. piControl internal variability slices (df_temp_PiC)
    if '--preindustrial-era' in argv_dict:
        if argv_dict['--preindustrial-era'].lower() in ['n', 'none', 'false']:
            inc_pi_offset = False
            start_pi, end_pi = None, None
        else:
            inc_pi_offset = True
            start_pi = int(argv_dict['--preindustrial-era'].split('-')[0])
            end_pi = int(argv_dict['--preindustrial-era'].split('-')[1])
    else:
        # Defaults to the standard in IPCC AR6 Ch.3
        inc_pi_offset = True
        start_pi, end_pi = 1850, 1900  

    # Generate a string to identify the pre-industrial period ('none' if
    # there is no offset).
    preindustrial_str = f'{start_pi}-{end_pi}' if inc_pi_offset else 'none'

    # Determine whether to include a constant term in the regression.
    if '--include-reg-const' in argv_dict:
        inc_reg_const = argv_dict['--include-reg-const']
        inc_reg_const = True if inc_reg_const.lower() in ['y', 'yes', 'true'] else False
    else:
        inc_reg_const = True

    if not inc_pi_offset and not inc_reg_const:
        run_log.note('Both the preindustrial offset and the regression '
                     'constant are disabled. Without these centring factors '
                     'the regression attribution may behave unexpectedly.')

    # Define the regression year range for the temperature attribution.
    # This is the range over which the regression coefficients are calculated.
    # For the full GWI, the regression range is the same as the length of the
    # HadCRUT dataset, which is 1850-current.
    # However, for other analyses, such as calculating a historical-only GWI,
    # the regression range will vary. For use with slurm on HPC, these are
    # provided as argvs by user.
    if '--regress-range' in argv_dict:
        start_regress = int(argv_dict['--regress-range'].split('-')[0])
        end_regress = int(argv_dict['--regress-range'].split('-')[1])
    else:
        # Fallback to requesting (interactive) input from user. This is safer
        # than hardcoding a fallback such as 1850-2023
        regress_input = input('Regression range (start-end): ')
        start_regress, end_regress = regress_input.split('-')

    # Determine truncation year range for the analysis.
    # Typically, for the full GWI calculation, we use the entire regressable
    # range of the HadCRUT observations, which is 1850-current.
    # Truncation has multiple benefits:
    # 1. It reduces the memory requirements of the analysis, which is the main
    #    bottleneck in the analysis.
    # 2. It allows for a separation between the output dataset and the
    #    regressed range, which is useful for using GWI to constrain future
    #    projections (using longer ERF inputs) using a shorter reference
    #    period for temperatures.
    if '--truncate' in argv_dict:
        start_trunc = int(argv_dict['--truncate'].split('-')[0])
        end_trunc = int(argv_dict['--truncate'].split('-')[1])
    else:
        # Fallback to clipping the data to the range of the regression.
        start_trunc = start_regress
        end_trunc = end_regress

    # Determine whether to calculate committed warming.
    if '--committed' in argv_dict:
        committed_arg = argv_dict['--committed']
    else:
        committed_arg = input('Calculate committed warming? (n or start-end): ')

    if committed_arg == 'n':
        year_committed_to = False
        year_committed_from = False
    
    else:
        # Parse the committed argument.
        # Format: start_year-end_year (e.g. 2024-2300 or end_regress-2300)
        if 'end_regress' in committed_arg:
            committed_arg = committed_arg.replace(
                'end_regress', str(end_regress))
        
        if '-' in committed_arg:
            year_committed_from = int(committed_arg.split('-')[0])
            year_committed_to = int(committed_arg.split('-')[1])
        
        else:
            raise ValueError(
                f'Invalid committed warming argument provided: {committed_arg}'
                )

    # Generate a string to identify the committed warming settings (empty if
    # there is no committed warming).
    committed_str = (
        f'_COMMITTED--{year_committed_from}-{year_committed_to}'
        if year_committed_to
        else '')

    # Determine the number of samples to take for each source.
    # This is the number of ensemble members to subsample from the available
    # ensemble members for each source. If this is larger than the ensemble for
    # a particular source, the entire ensemble is used. If it is smaller, a
    # random subset of the ensemble is used.
    if '--samples' in argv_dict:
        samples = int(argv_dict['--samples'])
    else:
        # If no command line argument passed, fallback to interactivity.
        samples = int(input('Max number of samples for each source (int): '))

    # Optional label that keeps one piece of work's output separate from
    # another's. With no tag, output goes to the plain results/ and plots/
    # folders exactly as before. Pass the same tag to
    # combine_results_iterations.py so that it reads what this run writes.
    output_tag = argv_dict.get('--output-tag')

    # Optional seed for the random sampling of ensemble members. With no seed
    # (no --seed, or --seed= left empty) every run samples afresh, so repeat
    # runs of the same configuration differ, as they always have. With a seed,
    # runs of the same configuration on the same inputs draw exactly the same
    # samples, and so give identical results: useful for checking that a code
    # change leaves the results unchanged.
    seed = argv_dict.get('--seed', '')
    if seed == '':
        seed = None
    elif seed.isdigit() and int(seed) < 2**32:
        seed = int(seed)
    else:
        raise ValueError(
            f'Invalid --seed {seed!r}: use a whole number from 0 to 2**32-1, '
            'or leave it empty for unseeded sampling.')

    # Determine whether to calculate rates (computaitonally expensive).
    if '--include-rate' in argv_dict:
        rate_toggle = argv_dict['--include-rate']
        rate_toggle = True if rate_toggle == 'y' else False
    else:
        # If no command line argument passed, use interactivity as a backup.
        rate_toggle = input('Calculate rates? (y/n): ')
        rate_toggle = True if rate_toggle == 'y' else False

    # Determine whether to include headline calculations in the output.
    # If no, then only the timeseries are calculated
    # Format: 'ANNUAL,SR1.5,AR6,CGWL' or some subset.
    if '--headline-toggles' in argv_dict:
        headline_toggles = argv_dict['--headline-toggles']
    else:
        headline_toggles = input(
            'Include headlines? (annual, SR1.5, AR6, CGWL, or n): ')

    if headline_toggles == 'n':
        headline_toggles = False
    else:
        tog_list = headline_toggles.split(',')
        tog_avail = ['annual', 'SR1.5', 'AR6', 'CGWL', '']
        # Check that the all elements of tog_list exist within tog_avail
        if not (all(tog in tog_avail for tog in tog_list)):
            raise ValueError('Invalid headline toggle(s) provided.')

    # Determine whether to include sub-variables in the output.
    if '--include-sub-vars' in argv_dict:
        include_sub_vars = argv_dict['--include-sub-vars']
        include_sub_vars = True if include_sub_vars == 'y' else False
    else:
        include_sub_vars = False

    # Determine whether to calculate the prior (pre-constrained) warming.
    # The priors depend only on the ERF ensemble and the FaIR parameters, NOT
    # on the reference temperature ensemble member, so when sweeping over
    # individual GMT members every run recomputes an identical prior dataset.
    # The priors are used only for plotting (in combine_results_iterations.py),
    # never for the attribution results themselves, so they can be switched off
    # for per-member sweeps. Defaults to 'y' to preserve previous behaviour.
    if '--calculate-priors-output' in argv_dict:
        calculate_priors_output = argv_dict['--calculate-priors-output']
        calculate_priors_output = (
            False if calculate_priors_output == 'n' else True)
    else:
        calculate_priors_output = True

    # Specify years for headlines calculations
    # Format '2024', '2022,2023,2024', etc.
    if '--headline-years' in argv_dict:
        headline_years = argv_dict['--headline-years']
    else:
        if headline_toggles:
            headline_years = input('Years for headlines: ')
        else:
            headline_years = None
    headline_years = warming_definitions.check_headlines(
        headline_years, end_regress, end_trunc)


    # Specify regression variables:
    if '--regress-variables' in argv_dict:
        regress_vars = sorted(argv_dict['--regress-variables'].split(','))
    else:
        regress_vars = sorted(['GHG', 'Nat', 'OHF'])

    # Specify the input datasets, by ID (see INPUT DATASETS in
    # src/inputs/input_datasets.py): the ERFs, the reference temperatures (GMT), and
    # the piControl simulations that sample internal variability.
    #   e.g. --erf-dataset=IGCC-2025 --gmt-dataset=HadCRUT-2025
    #   e.g. --erf-dataset=SSP_extension-2022-SSP245 --gmt-dataset=JK_multi-2025
    #   e.g. --erf-dataset=SMILE_ESM-SSP245 --gmt-dataset=SMILE_ESM-SSP245
    #   --picontrol-dataset=CMIP6 (the only one for now)

    # Stop if any of the three is missing
    for flag in ['--erf-dataset', '--gmt-dataset', '--picontrol-dataset']:
        if not argv_dict.get(flag):
            raise ValueError(
                f'{flag} is required, e.g. --erf-dataset=IGCC-2025 '
                '--gmt-dataset=HadCRUT-2025 --picontrol-dataset=CMIP6.')

    # Read the three dataset IDs
    erf_id = argv_dict['--erf-dataset']
    gmt_id = argv_dict['--gmt-dataset']
    picontrol_id = argv_dict['--picontrol-dataset']

    # Specify which ensemble members to use from each dataset: 'all' (the
    # default), or a single member, by its label in the dataset. One gwi.py
    # call takes one value for each; run-gwi.sh loops over member lists and
    # pairs members. An empty value means 'all'.
    ensemble_members = {
        'ERF': argv_dict.get('--erf-member', 'all') or 'all',
        'GMT': argv_dict.get('--gmt-member', 'all') or 'all',
    }

    # Stop now, before loading anything, if the datasets or members cannot
    # be attributed against each other (e.g. a model's ERF with observations).
    datasets.check_dataset_pairing(
        erf_id, gmt_id, ensemble_members['ERF'], ensemble_members['GMT'])

    # e.g. ERF-all_GMT-7 (always ERF first, as in ERF--<id>_GMT--<id>)
    ensemble_members_str = (
        f'ERF-{ensemble_members["ERF"]}_GMT-{ensemble_members["GMT"]}')

    # Create directory structure based on the input parameters.

    # GWI timeseries, headlines and rates
    output_path = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members_str}/' +
        f'VARIABLES--{"-".join(regress_vars)}/' +
        f'REGRESSED-YEARS--{start_regress}-{end_regress}/'
    )
    # Prior (unconstrained) warming
    output_path_priors = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members_str}/'
    )
    # ERF percentiles
    output_path_erfs = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members_str}/'
    )
    # Observation headlines, for the headline years of this regression
    out_path_obs = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members_str}/'
        f'REGRESSED-YEARS--{start_regress}-{end_regress}/'
    )
    # Observations themselves, which don't depend on the regression range
    out_path_obs_timeseries = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members_str}/'
    )
    # piControl segments, which depend only on the inputs (not on the members
    # or the regression range)
    out_path_picontrol = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}/'
    )

    # Create the results folders
    results_root = results_layout.tagged_folder('results', output_tag)
    results_folder = f'{results_root}/iterations/'
    if not os.path.exists(f'{results_folder}{output_path}'):
        os.makedirs(f'{results_folder}{output_path}',
        exist_ok=True)
    results_folder_picontrol = f'{results_root}/picontrol/'
    os.makedirs(f'{results_folder_picontrol}{out_path_picontrol}',
                exist_ok=True)
    results_folder_priors = f'{results_root}/priors/'
    if not os.path.exists(f'{results_folder_priors}{output_path_priors}'):
        os.makedirs(f'{results_folder_priors}{output_path_priors}',
        exist_ok=True)
    results_folder_erfs = f'{results_root}/erfs/'
    if not os.path.exists(f'{results_folder_erfs}{output_path_erfs}'):
        os.makedirs(f'{results_folder_erfs}{output_path_erfs}',
        exist_ok=True)
    results_folder_obs = f'{results_root}/observations/'
    if not os.path.exists(f'{results_folder_obs}{out_path_obs}'):
        os.makedirs(f'{results_folder_obs}{out_path_obs}',
        exist_ok=True)

    ###########################################################################
    # READ IN THE DATA ########################################################
    ###########################################################################

    # Effective Radiative Forcing
    df_forc = datasets.load_ERF(erf_id, regress_vars, ensemble_members['ERF'],
                                include_sub_vars=include_sub_vars)

    if year_committed_to and year_committed_from:
        # Ensure that the ERF data goes up to the committed year.
        df_forc = datasets.extend_ERF_to_committed_year(
            df_forc, year_committed_to, year_committed_from)

    forc_var_names = sorted(
        df_forc.columns.get_level_values('variable').unique())
    # Obtain the ERF_start and ERF_end from the dataframe.
    forc_Yrs = np.array(df_forc.index)
    forc_Yrs_min = forc_Yrs.min()
    forc_Yrs_max = forc_Yrs.max()

    # Check that the truncation years are within the ERF data range.
    # TODO: write down how and when the truncation happens (i.e. after the
    # regression).
    if start_trunc < forc_Yrs_min:
        start_trunc = forc_Yrs_min
        run_log.note('Truncation start year was before the ERF data range, so '
                     f'it was moved to the ERF minimum: {start_trunc}.')
    if end_trunc > forc_Yrs_max:
        end_trunc = forc_Yrs_max
        run_log.note('Truncation end year was after the ERF data range, so it '
                     f'was moved to the ERF maximum: {end_trunc}.')
    if year_committed_to and (year_committed_to != forc_Yrs_max):
        run_log.note('The ERF data end year does not match the committed '
                     'warming year.')
    if year_committed_to and (end_trunc < year_committed_to):
        run_log.note(f'Truncation end year {end_trunc} is before the '
                     f'committed warming end year {year_committed_to}.')

    trunc_Yrs = np.arange(start_trunc, end_trunc+1)

    # TEMPERATURE
    df_temp_Obs = datasets.load_GMT(
        gmt_id, ensemble_members['GMT'], start_pi, end_pi)
    # Obtain the maximum regressable years from the dataframe.
    temp_Yrs = np.array(df_temp_Obs.index)

    # Check that the regression years are within both the temperature and
    # the forcing data. Stop rather than shorten the range, which would
    # silently run a different regression from the one asked for (and,
    # before, filed it under the range asked for).
    for name, yrs_min, yrs_max in [
            (f'GMT dataset {gmt_id}', temp_Yrs.min(), temp_Yrs.max()),
            (f'ERF dataset {erf_id}', forc_Yrs_min, forc_Yrs_max)]:
        # Stop if the regression starts before, or ends after, this dataset
        if start_regress < yrs_min or end_regress > yrs_max:
            raise ValueError(
                f'Regression range {start_regress}-{end_regress} is not '
                f'within the {name}, which covers {yrs_min}-{yrs_max}.')

    # Check that the regression end year is not after the start of the
    # committed warming period.
    if year_committed_to and (end_regress > year_committed_from):
        raise ValueError(
            f'Regression end year ({end_regress}) cannot be after the '
            f'start of the committed warming period ({year_committed_from}). '
            'The regression must be performed on data consistent between the '
            'reference/observed temperatures and forcing data; once the'
            'constant ERF assumption begins, the scenario for forcing and'
            'reference warming are no longer aligned.')

    # Save the reference temperatures as used: the selected members, cut off
    # at the dataset's year, with the pre-industrial offset removed. The
    # combination script plots these, so it need not reload them from data/.
    # Identical for every iteration of a configuration, so (like the priors)
    # the file has no date and is simply overwritten.
    df_temp_Obs.to_csv(
        f'{results_folder_obs}{out_path_obs_timeseries}' +
        f'Obs_timeseries_ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{df_temp_Obs.shape[1]}.csv')

    compute_resources.log_slurm_configuration()

    run_log.log_section('Run configuration')
    print(f'Git commit: {run_log.describe_git_commit()}')
    print(f'Regressed variables: {regress_vars}')
    print(f'ERF dataset: {erf_id}')
    print(f'GMT dataset: {gmt_id}')
    print(f'piControl dataset: {picontrol_id}')
    if year_committed_to:
        print(f'Committed warming: forcing constant from:'
              f'{year_committed_to} to {year_committed_from}')
    print(f'Forcing range: {forc_Yrs_min}-{forc_Yrs_max}')
    print(f'Reference temperature range: {temp_Yrs.min()}-{temp_Yrs.max()}')
    print(f'Reference temperature realisation number: {ensemble_members["GMT"]}')
    print(f'ERF realisation number: {ensemble_members["ERF"]}')
    print(f'Pre-industrial era: {start_pi}-{end_pi}')
    print(f'Truncation range: {start_trunc}-{end_trunc}')
    print(f'Regression range: {start_regress}-{end_regress}')
    print(f'Number of samples: {samples}')
    print('Random seed: '
          f'{seed if seed is not None else "none (unseeded sampling)"}')
    print(f'Calculate and save rates: {rate_toggle}')
    print(f'Calculate and save priors: {calculate_priors_output}')
    print(f'Include pre-industrial offset: {inc_pi_offset}')
    print(f'Include constant term in regression: {inc_reg_const}')
    print(f'Using model: {model_choice}')
    print(f'Output folder: {results_root}/'
          f'{"" if output_tag else "  (no tag set)"}')

    # CMIP6 PI-CONTROL
    timeframes = [1, 3, 30]
    # Segments as long as the GMT record, indexed by its years
    df_temp_PiC = datasets.load_PiC(picontrol_id, temp_Yrs, start_pi, end_pi)
    df_temp_PiC = datasets.filter_PiControl(df_temp_PiC, timeframes)
    # NOTE: For 1850-2022, we get 183 realisations for piControl
    # NOTE: For 1850-2023, we get 181 realisations for piControl

    # Save the filtered piControl segments, for the internal-variability
    # figures drawn by combine_results_iterations.py. Identical for every
    # iteration of these inputs, so (like the priors) the file has no date
    # and is simply overwritten.
    df_temp_PiC.to_csv(
        f'{results_folder_picontrol}{out_path_picontrol}' +
        f'PiControl_timeseries_ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{df_temp_PiC.shape[1]}.csv')


    ###########################################################################
    # CARRY OUT GWI CALCULATION ###############################################
    ###########################################################################

    # Set model parameters ####################################################
    if model_choice == 'AR5_IR':
        # We only use a[10], a[11], a[15], a[16]
        # Defaults:
        # AR5 thermal sensitivity coeffs
        # a_ar5[10:12] = [0.631, 0.429]
        # AR5 thermal time-inc_Constants -- could use Geoffroy et al [4.1,249.]
        # a_ar5[15:17] = [8.400, 409.5]
        # a_ar5 = np.zeros(20, 16)
        # # Geoffrey 2013 paramters for a_ar5[15:17]
        Geoff = np.array([[4.0, 5.0, 4.5, 2.8, 5.2, 3.9, 4.2, 3.6,
                           1.6, 5.3, 4.0, 5.5, 3.5, 3.9, 4.3, 4.0],
                          [126, 267, 193, 132, 289, 200, 317, 197,
                           184, 280, 698, 286, 285, 164, 150, 218]])

    elif model_choice == 'FaIR_V2':
        # The original location of the FaIR tunings is here
        # CMIP6_param_csv = ('models/FaIR_V2/FaIRv2_0_0_alpha1/fair/util/' +
        #                    'parameter-sets/CMIP6_climresp.csv')
        # Which is simply copied to the following location and committed
        # to repo explicitly for transparency and convenience.
        CMIP6_param_csv = ('models/FaIR_V2/FaIR_CMIP6_climresp.csv')
        CMIP6_param_df = pd.read_csv(
            CMIP6_param_csv, index_col=[0], header=[0, 1])

    # Calculate GWI ###########################################################

    # Select random sub-set sampling of all ensemble members:

    # Note that we choose to select all possible parameterisations for FaIR,
    # as the number of parameterisations is relatively small, and we wish to
    # fully span model parameterisation uncertainty. For piControl (internal
    # variability), observed temperature data, and ERFs, we select an identical
    # number of ensemble members for each, (a) to evenly spread the uncertainty
    # across all sources, and (b) because 60 members for each source is the
    # maximum that can be handled by the HPC cluster RAM; 60 is already quite
    # low, and choosing lower for a particular source seems problematic.
    # However, for the resulting ensemble size (6 million), variance between
    # results from repeat random subsampling leads to consistent results, and
    # differences between samplings are well below the overall unceratinty
    # range. Variance is compensated for by making several repeat calculations
    # and averaging the resulting timeseries - see combine_results.py.

    # Seed the random sampling, if asked to. All three draws below (forcing,
    # temperature and internal variability members) use numpy's global
    # random state -- pandas' .sample() does too when given no random_state
    # -- and they are the only random draws in a run, so seeding here, just
    # before them, makes the whole run reproducible.
    if seed is not None:
        np.random.seed(seed)

    # 1. Select random samples of the forcing data
    run_log.log_notes()

    run_log.log_section('Ensemble sampling')
    print('Forcing ensemble all: ' +
          f'{len(df_forc.columns.get_level_values("ensemble").unique())}')
    # Select a random subset of the ensemble names from the forcing data.
    forc_sample = np.random.choice(
        df_forc.columns.get_level_values("ensemble").unique(),
        min(samples,
            len(df_forc.columns.get_level_values("ensemble").unique())),
        replace=False)
    # print('forc_sample:', forc_sample)
    # select all variables for first column level, and forc_sample for
    # second column level
    forc_subset = df_forc.loc[:, (slice(None), forc_sample)]
    # print(forc_subset)
    # forc_subset = df_forc.xs(tuple(forc_sample), axis=1, level=1)
    _nf = len(forc_subset.columns.get_level_values("ensemble").unique())
    print(f'Forcing ensemble pruned: {_nf}')

    # 2. Select ALL samples of the model parameters, no sub-sampling.
    print('FaIR Parameters: '
          f'{len(CMIP6_param_df.columns.levels[0].unique())}')
    if model_choice == 'AR5_IR':
        params_subset = Geoff[:, :min(samples, Geoff.shape[1])]
    elif model_choice == 'FaIR_V2':
        params_subset = CMIP6_param_df  # No sub-sampling
        # List the names of the CMIP6 models to be emulated. This list is used
        # to parallelise the GWI calculation across the different models.
        models = CMIP6_param_df.columns.levels[0].unique().to_list()

    # 3. Select random samples of the temperature data
    print(f'Temperature ensembles all: {df_temp_Obs.shape[1]}')
    df_temp_Obs_subset = df_temp_Obs.sample(
        n=min(samples, df_temp_Obs.shape[1]), axis=1)
    print(f'Temperature ensembles pruned: {df_temp_Obs_subset.shape[1]}')

    # 4. Select random samples of the internal variability
    print(f'Internal variability ensembles all: {df_temp_PiC.shape[1]}')
    df_temp_PiC_subset = df_temp_PiC.sample(
        n=min(samples, df_temp_PiC.shape[1]), axis=1)
    print('Internal variability ensembles pruned: '
          f'{df_temp_PiC_subset.shape[1]}')

    # Print the total available ensemble size
    _n_all = (
        len(df_forc.columns.get_level_values("ensemble").unique()) *
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs.shape[1] *
        df_temp_PiC.shape[1]
    )
    print(f'Max available ensemble: {_n_all:,}')

    # Print the randomly subsampled ensemble size
    _n_sub = (
        _nf *  # number of forcing ensembles
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs_subset.shape[1] *
        df_temp_PiC_subset.shape[1]
    )
    print(f'Sub-sampled ensemble size: {_n_sub:,} '
          f'({_n_sub / _n_all:.2%} of all possible combinations)')

    # Parallelise GWI calculation, with each thread corresponding to a
    # single (model) parameterisation for FaIR.
    #
    # Executor.map is a drop-in for Pool.imap here: it yields results in the
    # order of `models` as they arrive, so the ensemble axis is laid out
    # identically and results are unchanged.
    #
    # THE POINT: if the Monte Carlo ensemble is too big to fit in the job's
    # memory, runs using Executor.map fail within minutes with a clear
    # out-of-memory error, instead of silently hanging until the scheduler's
    # time limit kills it hours later, which is what Pool.imap does.
    #
    # Why the pool type decides that: the emulation below is the peak-memory
    # stage of the whole run, so when a job is too big it is these workers
    # that the kernel's OOM killer takes, not the parent process. mp.Pool
    # cannot recover from a worker dying. It quietly starts a replacement but
    # never fails the task the dead worker was holding, so that result never
    # arrives and map()/imap() waits for it forever -- burning the entire
    # walltime allocation at zero CPU, while still looking like a healthy
    # running job in squeue. ProcessPoolExecutor watches its workers and
    # raises BrokenProcessPool instead.
    #
    # Every emulation writes its ensemble members into one array that all the
    # workers share, rather than handing a block back for the parent to
    # collect. Sharing is what keeps the memory down -- returning a block
    # means copying it out of the worker and rebuilding it in the parent --
    # and it is also why nothing needs joining together afterwards. See
    # compute_resources.shared_results_array for the full explanation.
    #
    # The array's shape is worked out here, before the workers exist, from
    # the same inputs the emulations are given.
    vars_list = attribution.attribution_output_vars(forc_subset, regress_vars)
    n_per_model = attribution.ensemble_members_per_model(
        forc_subset, df_temp_Obs_subset, df_temp_PiC_subset)
    temp_Att_Results = compute_resources.shared_results_array(
        shape=(end_trunc - start_trunc + 1,     # years, after truncation
               len(vars_list),                  # attributed warming variables
               n_per_model * len(models)),      # ensemble members
        dtype=np.float32)

    # Settings shared by every emulation, and the array they all write into.
    # These are module-level so that the workers inherit them when they are
    # forked below; passing them as arguments would send a copy to each
    # worker, which for the results array is precisely what we are avoiding.
    emulation_settings = dict(
        inc_reg_const=inc_reg_const,
        inc_pi_offset=inc_pi_offset,
        df_forc=forc_subset,
        df_params=params_subset,
        df_temp_PiC=df_temp_PiC_subset,
        df_temp_Obs=df_temp_Obs_subset,
        start_trunc=start_trunc,
        end_trunc=end_trunc,
        start_pi=start_pi,
        end_pi=end_pi,
        start_regress=start_regress,
        end_regress=end_regress,
        regress_vars=regress_vars,
    )
    emulation_models = models
    emulation_results = temp_Att_Results

    run_log.log_section('Timeseries')
    print('Calculating GWI (parallelised)', end=' ')
    T1a = dt.datetime.now()
    # 'fork' is what lets the workers inherit the shared array. It is already
    # the default on Linux; naming it here means this keeps working if that
    # default ever changes.
    with ProcessPoolExecutor(compute_resources.n_workers(),
                             mp_context=mp.get_context('fork')) as p:
        # list() is what makes any failure in a worker surface here.
        list(p.map(emulate_one_model, range(len(models))))

    T1b = dt.datetime.now()
    print(f'... took {T1b - T1a}')
    # There is no longer a separate concatenation step to time, but T2a is
    # read below as the start marker for the PRIORS stage.
    T2a = T1b

    # Reminder: temp_Att_Results has shape (years, vars_list, n (ensembles members))
    n = temp_Att_Results.shape[2]

    # FILTER RESULTS ######################################################
    # For diagnosing: filter out results with particular regression
    # coefficients.
    # If you only want to study subsets of the results based on certain
    # constraints apply a mask here. The below mask is set to look at
    # results for different values of the coefficients.
    # Note to self about masking: coef_Reg_Results is the array of all
    # regression coefficients, with shape (4, n), where n is total number
    # of samplings. We select slice indices (forcing coefficients) we're
    # interested in basing the condition on:
    # AER is index 0, GHGs index 1, NAT index 2, Const index 3
    # Then choose whether you want any or all or the coefficients to meet
    # the condition (in this case being less than zero)
    mask_switch = False
    if mask_switch:
        coef_mask = np.all(coef_Reg_Results[[0, 2], :] <= 0, axis=0)
        # mask = np.any(coef_Reg_Results[[0], :] <= 0.0, axis=0)

        temp_Att_Results = temp_Att_Results[:, :, coef_mask]
        coef_Reg_Results = coef_Reg_Results[:, coef_mask]
        print('Shape of masked attribution results:', temp_Att_Results.shape)


    # CALCULATE PRIOR WARMING #################################################
    # NOTE: the ERF loading below is required even when the priors themselves
    # are not calculated, because df_forc_priors also feeds the ERF percentiles
    # output further down (which IS consumed by the post-processing).

    # Set new variable selection for priors calculation
    prior_vars = regress_vars.copy()
    # Add aggregate variables for prior calculation
    prior_vars.extend(variables.extra_vars(regress_vars))
    # Remove 'Res' from prior vars if present (it is added in extra_vars)
    if 'Res' in prior_vars:
        prior_vars.remove('Res')

    # Load in the forcings for the prior calculation, specifying sub-variables
    df_forc_priors = datasets.load_ERF(
        erf_id, prior_vars, ensemble_members['ERF'],
        include_sub_vars=include_sub_vars)
    # This is a two level column MultiIndex dataframe, with levels
    # 'variable' and 'ensemble'.
    prior_vars_all = sorted(
        df_forc_priors.columns.get_level_values("variable").unique().to_list())

    if year_committed_to:
        # Ensure that the ERF data goes up to the committed year.
        df_forc_priors = datasets.extend_ERF_to_committed_year(
            df_forc_priors, year_committed_to, year_committed_from)

    if calculate_priors_output:
        print('Calculating PRIORS (parallelised)', end=' ')
        # ProcessPoolExecutor rather than mp.Pool for the same reason as the
        # emulation pool above: an ensemble too big for the job's memory fails
        # here with a clear out-of-memory error, rather than hanging until the
        # scheduler's time limit. Executor.map returns a lazy iterator rather
        # than a list, so it is consumed into one here to keep
        # np.concatenate's input unchanged.
        with ProcessPoolExecutor(compute_resources.n_workers()) as p:
            # print('Partialising Function')
            partial_priors = functools.partial(
                attribution.model_prior_warming,
                df_params=params_subset,
                df_forc=df_forc_priors)
            results = list(p.map(partial_priors, models))

        # Combine results from all models into one array
        temp_Priors = np.concatenate(results, axis=2)
        T2b = dt.datetime.now()
        print(f'... took {T2b - T2a}')
    else:
        print('Skipping PRIORS (--calculate-priors-output=n)')
        # T2b is read further down as the start marker for the GWI percentile
        # timing, so it must be defined on both branches.
        T2b = T2a

    # PRODUCE FINAL RESULTS DATASETS ######################################

    # For multiple runs, we want to save the results with a unique identifier.
    # Use the current date and time of calculation for simplicity.
    current_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    iteration_id = current_time

    variation = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'VARIABLES--{"-".join(regress_vars)}_' +
        f'ENSEMBLE-SIZE--{n}_' +
        f'REGRESSED-YEARS--{start_regress}-{end_regress}_' +
        f'DATE-CALCULATED--{iteration_id}'
    )

    full_prior_size = (
        len(models) *
        len(df_forc.columns.get_level_values("ensemble").unique()))
    variation_priors = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{full_prior_size}_'
    )
    full_erf_size = (
        len(df_forc.columns.get_level_values("ensemble").unique())
    )
    variation_erfs = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{full_erf_size}_'
    )
    full_obs_size = (df_temp_Obs.shape[1])
    variation_obs = (
        f'ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{full_obs_size}'
    )

    # NOTE TO SELF: multidimensional np.percentile() changes the order of
    # the axes, so that the axis along which you took the percentiles is
    # now the first axis, and the other axes are the remaining axes...

    # TIMESERIES RESULTS ######################################################
    print('Calculating percentiles for GWI', end=' ')
    gwi_timeseries_array = compute_resources.percentile_threaded(
        temp_Att_Results, sigmas_all, axis=2)
    dict_Results = {
        (var, sigma):
        gwi_timeseries_array[sigmas_all.index(sigma), :, vars_list.index(var)]
        for var in vars_list for sigma in sigmas_all
    }
    df_Results = pd.DataFrame(dict_Results, index=trunc_Yrs)
    df_Results.columns.names = ['variable', 'percentile']
    df_Results.index.name = 'Year'
    gwi_filename = (f'{results_folder}{output_path}' +
                    f'GWI_results_timeseries_{variation}.csv')
    df_Results.to_csv(gwi_filename)

    T3a = dt.datetime.now()
    print(f'... took {T3a - T2b}')

    if calculate_priors_output:
        print('Calculating percentiles for priors', end=' ')
        priors_timeseries_array = compute_resources.percentile_threaded(
            temp_Priors, sigmas_all, axis=2)
        dict_Results_priors = {
            (var, sigma):
            priors_timeseries_array[
                sigmas_all.index(sigma), :, prior_vars_all.index(var)]
            for var in prior_vars_all for sigma in sigmas_all
        }
        df_Results_priors = pd.DataFrame(
            dict_Results_priors, index=df_forc_priors.index.to_numpy()
            )
        df_Results_priors.columns.names = ['variable', 'percentile']
        df_Results_priors.index.name = 'Year'
        priors_filename = (
            f'{results_folder_priors}{output_path_priors}' +
            f'PRIOR_results_timeseries_{variation_priors}.csv'
        )
        # Note that every iteration of GWI will result in an identical prior
        # dataset, so if running multiple iterations, this will overwrite the
        # previously saved prior dataset (there is no date identifier in the
        # filename), but note this is intended behaviour to avoid unnecessary
        # duplication of identical datasets.
        df_Results_priors.to_csv(priors_filename)

    print('Calculating percentiles for ERF', end=' ')
    erf_vars = sorted(
        df_forc_priors.columns.get_level_values('variable').unique())
    # Calculate percentiles for each variable and stack into a 3D array
    # (n_sigmas, years, n_vars), to keep consistent with GWI and Priors logic.
    erf_timeseries_array = np.stack([
        np.percentile(df_forc_priors[v].to_numpy(), sigmas_all, axis=1)
        for v in erf_vars
    ], axis=2)
    dict_Results_ERF = {
        (var, sigma):
        erf_timeseries_array[sigmas_all.index(sigma), :, erf_vars.index(var)]
        for var in erf_vars for sigma in sigmas_all
    }
    df_Results_ERF = pd.DataFrame(
        dict_Results_ERF, index=df_forc_priors.index.to_numpy())
    df_Results_ERF.columns.names = ['variable', 'percentile']
    df_Results_ERF.index.name = 'Year'

    erf_filename = (
        f'{results_folder_erfs}{output_path_erfs}' +
        f'ERF_results_timeseries_{variation_erfs}.csv'
    )
    df_Results_ERF.to_csv(erf_filename)

    T3b = dt.datetime.now()
    print(f'... took {T3b - T3a}')

    # HEADLINE RESULTS ########################################################
    # TODO: Update this.
    # NOTE: Currently, the headline results are calculated for two periods:
    # 1. The same years as the most recent IPCC reports (2017, and 2010-2019)
    # 2. The end of the truncation period, NOT the end of the regressed range.
    # The preferred range dependes on usage context, and can be changed later.

    if headline_toggles:
        run_log.log_section(f'Headlines ({headline_years})')

        hl_years = warming_definitions.generate_headline_years(
            headline_years, end_regress, end_trunc)

        dfs = []

    if 'annual' in headline_toggles:
        T1 = dt.datetime.now()
        # GWI-ANNUAL DEFINITION (SIMPLE VALUE IN A GIVEN YEAR) ################
        print('Reading annual mean definition temps', end=' ')

        hl_years_annual = [y for y in hl_years if y in trunc_Yrs]
        if ((headline_years == 'IGCC') and (2017 not in hl_years_annual)):
            hl_years_annual.append(2017)

        dfs = [df_Results.loc[[y]] for y in hl_years_annual]

        T2 = dt.datetime.now()
        print(f'... took {T2 - T1}')

    if 'SR1.5' in headline_toggles:
        T1 = dt.datetime.now()
        # SR15 DEFINITION (CENTRE OF 30-YEAR TREND) ###########################
        # Calculate the linear trend of the final 15 years of the timeseries
        # and use this to calculate the present-day warming
        print('Calculating SR15-definition temps', end=' ')

        hl_years_annual = [y for y in hl_years if y in trunc_Yrs]
        if ((headline_years == 'IGCC') and (2017 not in hl_years_annual)):
            hl_years_annual.append(2017)

        for year in hl_years_annual:
            if ((year not in trunc_Yrs) and (year-15 not in trunc_Yrs)):
                raise ValueError(
                    f'SR15 definition requires the years {year} and {year-15} '
                    'to be in the truncation years: '
                    f'({min(trunc_Yrs)}-{max(trunc_Yrs)})')

            years_SR15 = warming_definitions.contiguous_slice(
                (year-15 <= trunc_Yrs) * (trunc_Yrs <= year))
            temp_Att_Results_SR15_recent = temp_Att_Results[years_SR15, :, :]

            # Calculate SR15-definition warming for each var-ens combination
            # See SR15 Ch1 1.2.1
            # This previously looped over variables, creating an mp.Pool per
            # variable and mapping a per-series np.polyfit function over a
            # list of millions of individual 16-year series. The pickling of
            # those series dominated the runtime; the vectorised form below
            # fits every trend in one weighted sum. It agrees with the
            # per-series fit to ~1e-7 (float32 rounding) rather than exactly
            # -- see final_value_of_trend_vectorised for why.
            temp_Att_Results_SR15 = (
                warming_definitions.final_value_of_trend_vectorised(
                    temp_Att_Results_SR15_recent))

            # Obtain statistics
            gwi_headline_array = compute_resources.percentile_threaded(
                temp_Att_Results_SR15, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma): gwi_headline_array[sigmas_all.index(sigma),
                                                 vars_list.index(var)]
                for var in vars_list for sigma in sigmas_all
            }
            df_headlines_i = pd.DataFrame(
                dict_Results, index=[f'{year} (SR15 definition)'])
            df_headlines_i.columns.names = ['variable', 'percentile']
            df_headlines_i.index.name = 'Year'
            dfs.append(df_headlines_i)

        T2 = dt.datetime.now()
        print(f'... took {T2 - T1}')

    if 'AR6' in headline_toggles:
        T1 = dt.datetime.now()
        # AR6 DEFINITION (DECADE MEAN) ########################################
        hl_years_decadal = [y for y in hl_years
                            if ((y in trunc_Yrs) and (y-9 in trunc_Yrs))]
        if ((headline_years == 'IGCC') and (2019 not in hl_years_decadal)):
            hl_years_decadal.append(2019)

        print('Calculating AR6-definition temps', end=' ')
        # if ((2010 in trunc_Yrs) and (2019 in trunc_Yrs) and (end_regress != 2019)):
        #     # The final condition is to avoid duplicate calculations when
        #     # the end_regress is 2019.
        #     years_headlines = [[2010, 2019], [end_regress-9, end_regress]]
        # else:
        #     years_headlines = [[end_regress-9, end_regress]]

        for year in hl_years_decadal:
            if ((year not in trunc_Yrs) and (year-9 not in trunc_Yrs)):
                raise ValueError(
                    f'AR6 definition requires the years {year-9} and '
                    f'{year} to be in the truncation years: '
                    f'({min(trunc_Yrs)}-{max(trunc_Yrs)})')
            recent_years = warming_definitions.contiguous_slice(
                (year-9 <= trunc_Yrs) * (trunc_Yrs <= year))
            temp_Att_Results_AR6 = \
                temp_Att_Results[recent_years, :, :].mean(axis=0)

            # Obtain statistics
            gwi_headline_array = compute_resources.percentile_threaded(
                temp_Att_Results_AR6, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma): gwi_headline_array[sigmas_all.index(sigma),
                                                 vars_list.index(var)]
                for var in vars_list for sigma in sigmas_all
            }
            df_headlines_i = pd.DataFrame(
                dict_Results, index=['-'.join([str(year-9), str(year)])])
            df_headlines_i.columns.names = ['variable', 'percentile']
            df_headlines_i.index.name = 'Year'
            dfs.append(df_headlines_i)

        T2 = dt.datetime.now()
        print(f'... took {T2 - T1}')

    if 'CGWL' in headline_toggles:
        T1 = dt.datetime.now()
        # CGWL DEFINITION (20YR MEAN CENTERED WITH PROJECTIONS) ###############
        # This definition was propsoed in Betts et al., 2023, and is not
        # currently used in the IPCC/IGCC assessments.

        # Check that we have the years needed for the CGWL definition
        for year in hl_years:
            if not ((year-9 in trunc_Yrs) and (year+10 in trunc_Yrs)):
                # Often we can produce SR1.5 and AR6 definitions but not CGWL
                # due to not having data off the end of the truncation period.
                # Therefore, we simply print a warning rather than raising an
                # error, so that we can still produce the headlines csv without
                # the CGWL definition.
                print(
                    f'CGWL definition requires the years {year-9} and '
                    f'{year+10} to be in the truncation years: '
                    f'({min(trunc_Yrs)}-{max(trunc_Yrs)})', end=' ')

            else:
                print('Calculating CGWL-definition temps', end=' ')

                recent_years = warming_definitions.contiguous_slice(
                    (year-9 <= trunc_Yrs) * (trunc_Yrs <= year+10))
                temp_Att_Results_CGWL = \
                    temp_Att_Results[recent_years, :, :].mean(axis=0)
                # Obtain statistics
                gwi_headline_array = compute_resources.percentile_threaded(
                    temp_Att_Results_CGWL, sigmas_all, axis=1)
                dict_Results = {
                    (var, sigma): gwi_headline_array[sigmas_all.index(sigma),
                                                    vars_list.index(var)]
                    for var in vars_list for sigma in sigmas_all
                }
                df_headlines_i = pd.DataFrame(
                    dict_Results, index=[
                        f"{'-'.join([str(year-9), str(year+10)])} (CGWL definition)"
                    ])
                df_headlines_i.columns.names = ['variable', 'percentile']
                df_headlines_i.index.name = 'Year'
                dfs.append(df_headlines_i)

        T2 = dt.datetime.now()
        print(f'... took {T2 - T1}')

    df_headlines = pd.concat(dfs, axis=0)
    df_headlines.to_csv(f'{results_folder}{output_path}' +
                        f'GWI_results_headlines_{variation}.csv')

    # OBSERVATIONS HEADLINE RESULTS ###########################################
    if headline_toggles:
        run_log.log_section(f'Headlines for observations ({headline_years})')

        dfs_obs = []

        if 'annual' in headline_toggles:
            print('Reading annual mean definition temps (Obs)', end=' ')

            hl_years_annual = [y for y in hl_years
                               if y in df_temp_Obs.index]
            if ((headline_years == 'IGCC') and
                (2017 not in hl_years_annual) and
                (2017 in df_temp_Obs.index)
                ):
                hl_years_annual.append(2017)

            for year in hl_years_annual:
                temp_Obs_annual = df_temp_Obs.loc[year].to_numpy()
                obs_headline_array = np.percentile(temp_Obs_annual, sigmas_all)

                dict_Results_obs = {
                    ('Obs', sigma): obs_headline_array[sigmas_all.index(sigma)]
                    for sigma in sigmas_all
                }

                df_headlines_obs_i = pd.DataFrame(
                    dict_Results_obs, index=[year])
                df_headlines_obs_i.columns.names = ['variable', 'percentile']
                df_headlines_obs_i.index.name = 'Year'
                dfs_obs.append(df_headlines_obs_i)
            print('... done')

        if 'SR1.5' in headline_toggles:
            print('Calculating SR15-definition temps (Obs)', end=' ')

            hl_years_annual = [y for y in hl_years
                               if y in df_temp_Obs.index]
            if ((headline_years == 'IGCC') and
                (2017 not in hl_years_annual) and
                (2017 in df_temp_Obs.index)
                ):
                hl_years_annual.append(2017)

            for year in hl_years_annual:
                if ((year in df_temp_Obs.index) and (year-15 in df_temp_Obs.index)):
                    temp_Obs_SR15_recent = df_temp_Obs.loc[year-15:year].to_numpy()
                    # print(temp_Obs_SR15_recent)

                    # Vectorised for consistency with the attributed SR15
                    # temps above: both paths must use the same estimator, or
                    # they would disagree at the ~1e-7 rounding level.
                    results = (
                        warming_definitions.final_value_of_trend_vectorised(
                            temp_Obs_SR15_recent))

                    obs_headline_array = np.percentile(results, sigmas_all)

                    dict_Results_obs = {
                        ('Obs', sigma): obs_headline_array[sigmas_all.index(sigma)]
                        for sigma in sigmas_all
                    }

                    df_headlines_obs_i = pd.DataFrame(
                        dict_Results_obs, index=[f'{year} (SR15 definition)'])
                    df_headlines_obs_i.columns.names = ['variable', 'percentile']
                    df_headlines_obs_i.index.name = 'Year'
                    dfs_obs.append(df_headlines_obs_i)
            print('... done')

        if 'AR6' in headline_toggles:
            print('Calculating AR6-definition temps (Obs)', end=' ')

            hl_years_decadal = [y for y in hl_years
                                if ((y in df_temp_Obs.index) and
                                    (y-9 in df_temp_Obs.index))]
            if ((headline_years == 'IGCC') and
                (2019 not in hl_years_decadal) and
                (2019 in df_temp_Obs.index) and
                (2010 in df_temp_Obs.index)):

                hl_years_decadal.append(2019)

            for year in hl_years_decadal:
                temp_Obs_AR6 = df_temp_Obs.loc[year-9:year].mean(axis=0
                                                                 ).to_numpy()

                obs_headline_array = np.percentile(temp_Obs_AR6, sigmas_all)

                dict_Results_obs = {
                    ('Obs', sigma): obs_headline_array[sigmas_all.index(sigma)]
                    for sigma in sigmas_all
                }

                df_headlines_obs_i = pd.DataFrame(
                    dict_Results_obs, index=['-'.join([str(year-9), str(year)])])
                df_headlines_obs_i.columns.names = ['variable', 'percentile']
                df_headlines_obs_i.index.name = 'Year'
                dfs_obs.append(df_headlines_obs_i)
            print('... done')

        if 'CGWL' in headline_toggles:
            print('Calculating CGWL-definition temps (Obs)', end=' ')

            for year in hl_years:
                if ((year-9 in df_temp_Obs.index) and (year+10 in df_temp_Obs.index)):
                    temp_Obs_CGWL = df_temp_Obs.loc[year-9:year+10].mean(axis=0).to_numpy()

                    obs_headline_array = np.percentile(temp_Obs_CGWL, sigmas_all)

                    dict_Results_obs = {
                        ('Obs', sigma): obs_headline_array[sigmas_all.index(sigma)]
                        for sigma in sigmas_all
                    }

                    df_headlines_obs_i = pd.DataFrame(
                        dict_Results_obs, index=[f"{'-'.join([str(year-9), str(year+10)])} (CGWL definition)"])
                    df_headlines_obs_i.columns.names = ['variable', 'percentile']
                    df_headlines_obs_i.index.name = 'Year'
                    dfs_obs.append(df_headlines_obs_i)
            print('... done')

        if dfs_obs:
            df_headlines_obs = pd.concat(dfs_obs, axis=0)
            df_headlines_obs.to_csv(
                f'{results_folder_obs}{out_path_obs}' +
                f'Obs_results_headlines_{variation_obs}.csv')

    # RATE: AR6 DEFINITION
    if rate_toggle:
        T7 = dt.datetime.now()
        run_log.log_section('Warming rates')
        print('Calculating AR6-definition attributed warming rates', end=' ')
        dfs_rates = []
        for year in np.arange(1950, end_trunc+1):
            recent_years = warming_definitions.contiguous_slice(
                (year-9 <= trunc_Yrs) * (trunc_Yrs <= year))
            ten_slice = temp_Att_Results[recent_years, :, :]

            # Calculate AR6-definition warming rate for each var-ens
            # combination. See AR6 WGI Chapter 3 Table 3.1.
            # This loop runs once per year from 1950, and previously created an
            # mp.Pool per variable within each of those years -- several
            # hundred pools per member, each pickling millions of 10-year
            # series. That is why enabling rates used to cost days. The
            # vectorised form fits every trend in one weighted sum, and agrees
            # with the per-series np.polyfit fit to ~1e-16.
            temp_Rate_Results = warming_definitions.rate_func_vectorised(
                ten_slice)

            # Obtain statistics
            gwi_rate_array = compute_resources.percentile_threaded(
                temp_Rate_Results, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma):
                gwi_rate_array[sigmas_all.index(sigma), vars_list.index(var)]
                for var in vars_list for sigma in sigmas_all
            }
            df_rates_i = pd.DataFrame(
                dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
            df_rates_i.columns.names = ['variable', 'percentile']
            df_rates_i.index.name = 'Year'
            dfs_rates.append(df_rates_i)

        df_rates = pd.concat(dfs_rates, axis=0)
        df_rates.to_csv(f'{results_folder}{output_path}' +
                        f'GWI_results_rates_{variation}.csv')
        T8 = dt.datetime.now()
        print(f'... took {T8 - T7}')

    # What the run actually cost, to sit alongside the 'Memory usage
    # anticipated' block printed when the results array was made.
    run_log.log_section('Memory usage recorded')
    compute_resources.report_memory_used(temp_Att_Results.nbytes)

    time_omega = dt.datetime.now()
    total_time = time_omega - time_alpha
    run_log.log_section('Total')
    print(f'Total time taken: {total_time}')
