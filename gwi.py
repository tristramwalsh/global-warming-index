"""Script to generate global warming index."""

import os
import sys

import datetime as dt

import numpy as np
import pandas as pd

# import scipy.stats as ss

import src.method.attribution as attribution
import src.method.emulators as emulators
import src.method.members as members
import src.method.regression as regression
import src.method.sampling as sampling
import src.infrastructure.compute_resources as compute_resources
import src.postprocessing.results_layout as results_layout
import src.infrastructure.run_log as run_log
import src.method.warming_definitions as warming_definitions
import src.inputs.input_datasets as datasets

import models.AR5_IR.AR5_IR as AR5_IR


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def modelled_warming(df_forc, df_params, inc_pi_offset, start_pi, end_pi):
    """Stage A: the modelled warming from each forcing variable alone,

        T_mod[p, e, t, v]

    for every FaIR parameter set p, every ERF member e, every forcing year t
    and every forcing variable v. Both the attribution (stages B and C) and
    the priors are built from it.

    df_forc: forcing, columns (variable, ensemble), for every ERF member,
        in the order of the e axis
    df_params: FaIR thermal parameter sets, columns (set name, parameter),
        in the order of the p axis
    inc_pi_offset: whether to express the modelled warming relative to its
        mean over the pre-industrial years, start_pi to end_pi (both ends
        included); without it, it is relative to the first forcing year
    Returns (T_mod, forcing_vars): T_mod is an array (p, e, t, v) over the
    forcing years (df_forc's index), and forcing_vars the names along v,
    sorted.
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


def attribute(T_mod, forcing_vars, forc_Yrs, df_temp_Obs, df_temp_PiC,
              member_index, regress_vars, inc_reg_const,
              start_regress, end_regress, start_trunc, end_trunc):
    """Attribute the observed warming: every ensemble member's attributed
    warming, by stages B and C of the method, from stage A's modelled
    warming (see modelled_warming):

        Stage A  T_mod[p, e, t, v]   the modelled warming from each forcing
                                     variable alone (emulators)
        Stage B  β_obs[p, e, o, k]   the scaling factors that fit it to each
                 β_pic[p, e, c, k]   observed member and to each piControl
                                     segment (regression)
        Stage C  the attributed warming of the members (p, e, o, c) chosen
                 (sampling), rebuilt from the pieces (members)

    Stage B runs on every member of every source given; only stage C
    depends on which members were chosen.

    The axes:

        p  FaIR parameter set
        e  ERF ensemble member
        o  observed GMT member
        c  piControl segment
        t  year
        v  forcing variable (e.g. GHG, Nat, OHF, and any sub-variables)
        k  regressor: one per regressed variable, plus one for the constant
           term if it is used

    T_mod: array (p, e, t, v), from modelled_warming
    forcing_vars: the names along its v axis
    forc_Yrs: the years along its t axis
    df_temp_Obs: every observed GMT member (columns, in the order of the o
        axis), indexed by year
    df_temp_PiC: every piControl segment (columns, in the order of the c
        axis), indexed by the same years
    member_index: the members to rebuild, (p, e, o, c): four integer arrays
        of length N, positions along those axes, e.g. from a sampling design
        (see src/method/sampling.py); member n goes to position n of the
        results' member axis
    regress_vars: the regressed variables, e.g. ['GHG', 'Nat', 'OHF']
    inc_reg_const: whether the regression has a constant term
    start_regress, end_regress: the regression years
    start_trunc, end_trunc: the output years
    (Every year range includes both its ends.)
    Returns (temp_Att_Results, output_vars): every member's attributed
    warming over the output years, an array (t, variables, N) in float32,
    and the names along its variable axis.
    """
    # Every member of the observed and piControl sources, in order: the o
    # and c axes
    ens_list_Obs = df_temp_Obs.columns.to_list()
    ens_list_PiC = df_temp_PiC.columns.to_list()
    # The variables along the results array's variable axis: the forcing
    # variables, then the diagnosed variables (Ant, Tot, Res) that are not
    # among them
    output_vars = attribution.attribution_output_vars(forcing_vars,
                                                      regress_vars)

    # The years of the observations
    temp_Yrs = df_temp_Obs.index.to_numpy()  # observed GMT
    PiC_Yrs = df_temp_PiC.index.to_numpy()  # piControl segments

    # Stage B: the scaling factors. Every observed GMT member, and every
    # piControl segment, is fitted onto each (parameter set, ERF member)'s
    # modelled warming, all in one batched solve. The regressors X are the
    # regressed variables (plus a column of ones, if a constant is used)
    # over the regression years: one matrix (t, k) per (p, e).

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
    Y_obs = df_temp_Obs[ens_list_Obs].to_numpy()[
        (temp_Yrs >= start_regress) & (temp_Yrs <= end_regress)]
    # The piControl segments over the regression years: (t, c)
    Y_pic = df_temp_PiC[ens_list_PiC].to_numpy()[
        (PiC_Yrs >= start_regress) & (PiC_Yrs <= end_regress)]
    # Every fit at once: coefs.obs (p, e, o, k), coefs.pic (p, e, c, k)
    coefs = regression.ols(X, Y_obs, Y_pic)

    # Stage C: rebuild each chosen member from the pieces.

    # Which forcing years are in the output (the truncation years)
    in_trunc_forc = (forc_Yrs >= start_trunc) & (forc_Yrs <= end_trunc)
    # The modelled warming over those years: (p, e, t, v)
    T_mod_trunc = T_mod[:, :, in_trunc_forc, :]
    # The observed members over the output years, NaN in years after the
    # observations end: (t, o)
    T_obs_trunc = df_temp_Obs[ens_list_Obs].reindex(
        np.arange(start_trunc, end_trunc + 1)).to_numpy()
    # The results array, after checking it fits in memory: (t, variables, N).
    # float32 halves its memory against float64; its precision, about 7
    # significant digits (~1e-7 °C on warming of order 1 °C), is far finer
    # than any output reports
    temp_Att_Results = compute_resources.results_array(
        shape=(end_trunc - start_trunc + 1,  # years, after truncation
               len(output_vars),  # attributed warming variables
               len(member_index[0])),  # members
        dtype=np.float32)
    # Rebuild every member into it, in blocks, over the job's CPUs
    members.rebuild_into(temp_Att_Results, T_mod_trunc, coefs, T_obs_trunc,
                         member_index, forcing_vars, regress_vars,
                         inc_reg_const, output_vars,
                         n_threads=compute_resources.n_workers())

    return temp_Att_Results, output_vars


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

    # The emulator: FaIR v2.0 ('FaIR_V2'). The AR5 impulse response model
    # ('AR5_IR') has no adapter in src/method/emulators.py, so it can't run.
    # emulator_choice = 'AR5_IR'
    emulator_choice = 'FaIR_V2'

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
        '--include-sub-vars', '--output-tag',
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
    # the file has no date: each run overwrites it, atomically, since
    # several jobs may write it at once (see results_layout.write_csv).
    results_layout.write_csv(
        df_temp_Obs,
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
    print(f'Include pre-industrial offset: {inc_pi_offset}')
    print(f'Include constant term in regression: {inc_reg_const}')
    print(f'Using model: {emulator_choice}')
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
    # iteration of these inputs, so (like the priors) the file has no date:
    # each run overwrites it, atomically, since several jobs may write it at
    # once (see results_layout.write_csv).
    results_layout.write_csv(
        df_temp_PiC,
        f'{results_folder_picontrol}{out_path_picontrol}' +
        f'PiControl_timeseries_ERF--{erf_id}_GMT--{gmt_id}_' +
        f'PREINDUSTRIAL--{preindustrial_str}{committed_str}_' +
        f'ENSEMBLE-SIZE--{df_temp_PiC.shape[1]}.csv')


    ###########################################################################
    # CARRY OUT GWI CALCULATION ###############################################
    ###########################################################################

    # Set model parameters ####################################################
    if emulator_choice == 'AR5_IR':
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

    elif emulator_choice == 'FaIR_V2':
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
    # random state, and they are the only random draws in a run, so seeding
    # here, just before them, makes the whole run reproducible.
    if seed is not None:
        np.random.seed(seed)

    # The draws choose which members are rebuilt (stage C). Stages A and B
    # run on every member of every source, whatever is drawn. Each draw
    # gives positions along its source's axis, in the order drawn:
    # np.random.choice(n, k, replace=False) is the draw the legacy code made
    # (np.random.choice on the ERF members' names, and DataFrame.sample on
    # the GMT members and piControl segments, which draw positions the same
    # way), so the same seed draws the same members, in the same order.

    run_log.log_notes()
    run_log.log_section('Ensemble sampling')

    # 1. A random subset of the ERF members
    # Every ERF member, in the order of the forcing's columns: the e axis
    erf_members = df_forc.columns.get_level_values('ensemble').unique()
    print(f'Forcing ensemble all: {len(erf_members)}')
    # The positions drawn along the e axis: (up to samples,)
    erf_drawn = np.random.choice(
        len(erf_members), min(samples, len(erf_members)), replace=False)
    print(f'Forcing ensemble pruned: {len(erf_drawn)}')

    # 2. Every parameter set, no sub-sampling.
    print('FaIR Parameters: '
          f'{len(CMIP6_param_df.columns.levels[0].unique())}')
    if emulator_choice == 'AR5_IR':
        params_subset = Geoff[:, :min(samples, Geoff.shape[1])]
    elif emulator_choice == 'FaIR_V2':
        # Every parameter set, no sub-sampling, sorted by name: the order of
        # the p axis, and so of the blocks along the results' member axis
        params_subset = CMIP6_param_df.sort_index(
            axis=1, level=0, sort_remaining=False)
        # The names of the CMIP6 models whose parameter sets are used
        models = params_subset.columns.get_level_values(0).unique().to_list()
    # Every position along the p axis: (P,)
    params_drawn = np.arange(len(models))

    # 3. A random subset of the observed GMT members
    print(f'Temperature ensembles all: {df_temp_Obs.shape[1]}')
    # The positions drawn along the o axis: (up to samples,)
    obs_drawn = np.random.choice(
        df_temp_Obs.shape[1], min(samples, df_temp_Obs.shape[1]),
        replace=False)
    print(f'Temperature ensembles pruned: {len(obs_drawn)}')

    # 4. A random subset of the piControl segments
    print(f'Internal variability ensembles all: {df_temp_PiC.shape[1]}')
    # The positions drawn along the c axis: (up to samples,)
    pic_drawn = np.random.choice(
        df_temp_PiC.shape[1], min(samples, df_temp_PiC.shape[1]),
        replace=False)
    print(f'Internal variability ensembles pruned: {len(pic_drawn)}')

    # The members to rebuild: every combination of the drawn members (the
    # cube design), as four arrays of positions (p, e, o, c), each (N,)
    member_index = sampling.cube(params_drawn, erf_drawn, obs_drawn,
                                 pic_drawn)

    # Print the total available ensemble size
    _n_all = (
        len(erf_members) *
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs.shape[1] *
        df_temp_PiC.shape[1]
    )
    print(f'Max available ensemble: {_n_all:,}')

    # Print the randomly subsampled ensemble size
    _n_sub = len(member_index[0])
    print(f'Sub-sampled ensemble size: {_n_sub:,} '
          f'({_n_sub / _n_all:.2%} of all possible combinations)')

    T1a = dt.datetime.now()
    # Stage A, once, for both the attribution and the priors: the modelled
    # warming for every parameter set and ERF member, (p, e, t, v)
    T_mod, forcing_vars = modelled_warming(
        df_forc=df_forc,
        df_params=params_subset,
        inc_pi_offset=inc_pi_offset,
        start_pi=start_pi,
        end_pi=end_pi,
    )
    # Attribute every member: stages B and C (see attribute)
    temp_Att_Results, vars_list = attribute(
        T_mod=T_mod,
        forcing_vars=forcing_vars,
        forc_Yrs=df_forc.index.to_numpy(),
        df_temp_Obs=df_temp_Obs,
        df_temp_PiC=df_temp_PiC,
        member_index=member_index,
        regress_vars=regress_vars,
        inc_reg_const=inc_reg_const,
        start_regress=start_regress,
        end_regress=end_regress,
        start_trunc=start_trunc,
        end_trunc=end_trunc,
    )
    T1b = dt.datetime.now()
    run_log.log_section('Timeseries')
    print(f'Calculating GWI ... took {T1b - T1a}')
    # The start of the PRIORS stage
    T2a = T1b
    # The number of members, for the output file names
    n = temp_Att_Results.shape[2]


    # PRIOR WARMING ###########################################################
    # The prior warming is stage A's modelled warming itself, T_mod: the
    # warming before the observational constraint (every scaling factor 1),
    # so it needs no calculation of its own. Its summary is taken over every
    # (parameter set, ERF member) pair, each counted once: the prior of the
    # full ensemble, which may differ from the pairs the attribution's
    # members use, which depend on the sampling design and its size. The
    # priors are always written: they cost a second or so.

    print('Adding the aggregate variables to the PRIORS', end=' ')
    # Every forcing variable plus the aggregates (Ant, Tot), as the
    # attribution's outputs hold them: (p, e, t, v'), and the names along
    # v', sorted
    T_prior, prior_vars_all = attribution.add_aggregate_variables(
        T_mod, forcing_vars, regress_vars)
    T2b = dt.datetime.now()
    print(f'... took {T2b - T2a}')
    # How many (parameter set, ERF member) pairs there are, and how many the
    # attribution's members use (each pair as one number, p · E + e)
    n_erf = T_mod.shape[1]
    n_pairs = T_mod.shape[0] * n_erf
    pairs_used = np.unique(member_index[0] * n_erf + member_index[1])
    print(f'The priors are every (parameter set, ERF member) pair, '
          f'{T_mod.shape[0]} x {n_erf} = {n_pairs:,}, each counted once, '
          f'which may differ from the pairs the attribution samples: its '
          f'members use {len(pairs_used):,} of them.')

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
    results_layout.write_csv(df_Results, gwi_filename)

    T3a = dt.datetime.now()
    print(f'... took {T3a - T2b}')

    print('Calculating percentiles for priors', end=' ')
    T_prior_start = dt.datetime.now()
    # The percentiles over every parameter set and ERF member:
    # (n_sigmas, years, v'), laid out as for the GWI
    priors_timeseries_array = np.percentile(
        T_prior, sigmas_all, axis=(0, 1))
    dict_Results_priors = {
        (var, sigma):
        priors_timeseries_array[
            sigmas_all.index(sigma), :, prior_vars_all.index(var)]
        for var in prior_vars_all for sigma in sigmas_all
    }
    df_Results_priors = pd.DataFrame(
        dict_Results_priors, index=df_forc.index.to_numpy())
    df_Results_priors.columns.names = ['variable', 'percentile']
    df_Results_priors.index.name = 'Year'
    priors_filename = (
        f'{results_folder_priors}{output_path_priors}' +
        f'PRIOR_results_timeseries_{variation_priors}.csv'
    )
    # Every iteration of a configuration gives identical priors, so the
    # file has no date: each run overwrites it, atomically, since several
    # jobs may write it at once (see results_layout.write_csv).
    results_layout.write_csv(df_Results_priors, priors_filename)
    print(f'... took {dt.datetime.now() - T_prior_start}')

    print('Calculating percentiles for ERF', end=' ')
    # The forcing of every variable, for every ERF member: (t, e, v)
    erf_all = emulators.forcing_array(
        df_forc, forcing_vars,
        df_forc.columns.get_level_values('ensemble').unique())
    # ... plus the aggregates (Ant, Tot), as sums of their parts, as for the
    # priors: (t, e, v'), and the names along v', sorted
    erf_all, erf_vars = attribution.add_aggregate_variables(
        erf_all, forcing_vars, regress_vars)
    # The percentiles over the ERF members: (n_sigmas, years, v'), laid out
    # as for the GWI and the priors
    erf_timeseries_array = np.percentile(erf_all, sigmas_all, axis=1)
    dict_Results_ERF = {
        (var, sigma):
        erf_timeseries_array[sigmas_all.index(sigma), :, erf_vars.index(var)]
        for var in erf_vars for sigma in sigmas_all
    }
    df_Results_ERF = pd.DataFrame(
        dict_Results_ERF, index=df_forc.index.to_numpy())
    df_Results_ERF.columns.names = ['variable', 'percentile']
    df_Results_ERF.index.name = 'Year'

    erf_filename = (
        f'{results_folder_erfs}{output_path_erfs}' +
        f'ERF_results_timeseries_{variation_erfs}.csv'
    )
    results_layout.write_csv(df_Results_ERF, erf_filename)

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
    results_layout.write_csv(
        df_headlines,
        f'{results_folder}{output_path}' +
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
            results_layout.write_csv(
                df_headlines_obs,
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
        results_layout.write_csv(
            df_rates,
            f'{results_folder}{output_path}' +
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
