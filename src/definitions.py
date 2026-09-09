import sys
import os
import re
import shutil
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
import functools
import xarray as xr
import glob
from pathlib import Path
import pymagicc
import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################


# lru_cache makes Python work the answer out on the first call and hand back
# that same stored number on every later call. Four of the mp.Pool sites sit
# inside loops, so without it we would fork a scontrol subprocess hundreds of
# times per run. It also guarantees the count cannot change mid-run.
@functools.lru_cache(maxsize=1)
def n_workers():
    """Return the number of CPUs actually allocated to this job.

    os.cpu_count() reports the whole node rather than the allocation, so a Pool
    sized by it oversubscribes the job's memory cgroup and the workers get
    OOM-killed; a Pool worker killed mid-task then makes Pool.map() hang
    forever.

    No single source covers every way these scripts get run, so three are tried
    in descending order of trustworthiness:

      1. Environment variables - set by sbatch/srun job steps (gwi.py).
      2. scontrol              - interactive shells attach to step_extern,
                                 which sets none of those variables, so ask
                                 the scheduler directly (combine script).
      3. CPU affinity          - not under SLURM at all (login node, laptop).
    """
    # 1. Believe any count already stated explicitly. GWI_NUM_WORKERS is the
    #    manual override; the SLURM_* pair is set inside real job steps.
    #    CPUS_PER_TASK is preferred because CPUS_ON_NODE reports the node's
    #    whole allocation, which over-counts if a job runs >1 task per node.
    #    .get(var, '') with .isdigit() rejects missing/empty/malformed at once.
    for var in ('GWI_NUM_WORKERS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        if os.environ.get(var, '').isdigit():
            return max(1, int(os.environ[var]))

    # 2. Inside a SLURM job, but the variables above are absent - this is an
    #    interactive (step_extern) shell, so ask the scheduler itself. Note
    #    that sched_getaffinity and the cpuset cgroup are NOT restricted to the
    #    allocation on arc-htc, so they cannot stand in here. Any failure falls
    #    through to step 3 rather than killing the run.
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        try:
            output = subprocess.run(
                ['scontrol', 'show', 'job', job_id],
                capture_output=True, text=True, timeout=10).stdout
            match = re.search(r'NumCPUs=(\d+)', output)
            if match:
                return max(1, int(match.group(1)))
        except (OSError, subprocess.SubprocessError):
            pass

    # 3. No SLURM allocation at all. Prefer the CPUs this process is permitted
    #    to run on, since unlike os.cpu_count() that respects taskset/cpuset
    #    pinning where it exists. sched_getaffinity is Linux-only though, so on
    #    macOS/Windows fall back to the plain core count.
    if hasattr(os, 'sched_getaffinity'):
        n_available = len(os.sched_getaffinity(0))
    else:
        n_available = os.cpu_count() or 1

    # Finding scontrol on PATH but no job ID means we are on a cluster machine
    # outside any allocation, i.e. a login/head node, where nothing caps us and
    # the damage lands on other users. An ordinary machine sees no warning.
    if shutil.which('scontrol'):
        print(
            f'WARNING: no SLURM allocation found, so falling back to '
            f'{n_available} workers.\n'
            f'         On a login/head node this spawns {n_available} '
            f'processes and degrades it for everyone else.\n'
            f'         Request an interactive node instead, e.g.:\n'
            f'             srun -p interactive -c 16 --mem-per-cpu=2G '
            f'--pty /bin/bash\n'
            f'         Or cap the workers explicitly:\n'
            f'             GWI_NUM_WORKERS=4 python <script>.py',
            file=sys.stderr)

    return n_available


SUB_VAR_MAPPING = {
    'GHG': ['co2', 'ch4', 'n2o', 'halogen'],
    'OHF': ['aerosol-radiation_interactions', 'aerosol-cloud_interactions',
            'contrails', 'land_use', 'bc_snow', 'h2o_strat', 'o3'],
    'Nat': ['solar', 'volcanic'],
    'Ant': ['GHG', 'OHF'],
    'Tot': ['Ant', 'Nat']
}


VAR_NAMES = {
    'Obs': 'Observed warming',
    'Tot': 'Total forced warming',
    'Ant': 'Human-induced warming',
    'GHG': 'Well-mixed greenhouse gases',
    'OHF': 'Other human forcings',
    'Nat': 'Solar and volcanic drivers',
    'Res': 'Residual (Internal variability)',
    'co2': 'Carbon dioxide',
    'ch4': 'Methane',
    'n2o': 'Nitrous oxide',
    'halogen': 'Halogenated gases',
    'aerosol-radiation_interactions': 'Aerosol-radiation interactions',
    'aerosol-cloud_interactions': 'Aerosol-cloud interactions',
    'land_use': 'Land-use reflectance',
    'bc_snow': 'Black carbon on snow',
    'h2o_strat': 'Stratospheric water vapour',
    'o3': 'Ozone',
    'solar': 'Solar',
    'volcanic': 'Volcanic',
    'contrails': 'Aviation contrails'
}

def load_ERF(scenario, regress_vars, ensemble_members, include_sub_vars=False):
    """Load the ERFs for the specified scenario and variables."""

    if 'observed-20' in scenario:
        df_ERF = load_ERF_CMIP6(scenario, include_sub_vars=include_sub_vars)
    elif 'observed_JK' in scenario:
        df_ERF = load_ERF_SSP(scenario, regress_vars)
    elif 'observed-SSP' in scenario:
        df_ERF = load_ERF_SSP(scenario, regress_vars)
    elif 'SMILE_ESM' in scenario:
        df_ERF = load_ERF_SMILE(scenario, regress_vars)
    elif 'NorESM' in scenario:
        df_ERF = load_ERF_NorESM(scenario, regress_vars)
    else:
        raise ValueError('Invalid scenario for ERF data.')

    # Extract just the required variables
    df_ERF = extract_variables(df_ERF,
                               regress_vars,
                               include_sub_vars=include_sub_vars)
    # Extract just the required ensemble members
    df_ERF = extract_ensembles(df_ERF, ensemble_members)

    return df_ERF


def extract_variables(df_ERF, regress_vars, include_sub_vars=False):
    """Extract specific variables from the ERF dataframe."""

    # Check that the regress_vars are present in the dataframe
    available_vars = df_ERF.columns.get_level_values('variable').unique()
    missing_vars = set(regress_vars) - set(available_vars)
    if missing_vars:
        raise ValueError(
            'The following regression variables are not available in the '
            f'ERF data: {missing_vars}')

    # Select vars:
    if include_sub_vars:
        # If including sub-variables, we need to keep the regression variables
        # AND their sub-variables.

        # Helper to recursively find all sub-variables
        def get_all_sub_vars(var):
            sub_vars = []
            if var in SUB_VAR_MAPPING:
                direct_subs = SUB_VAR_MAPPING[var]
                sub_vars.extend(direct_subs)
                for sv in direct_subs:
                    sub_vars.extend(get_all_sub_vars(sv))
            return sub_vars

        # 1. We trivially need to keep the regress_vars
        vars_to_keep = list(regress_vars)

        # 2. For each regress_var, find and add its sub-variables
        for rv in regress_vars:
            vars_to_keep.extend(get_all_sub_vars(rv))

        vars_to_keep = list(set(vars_to_keep))

        # Filter dataframe
        available_vars = df_ERF.columns.get_level_values('variable').unique()
        vars_to_keep = [v for v in vars_to_keep if v in available_vars]

        df_ERF = df_ERF.loc[:, (vars_to_keep, slice(None))]

    else:
        df_ERF = df_ERF.loc[:, (regress_vars, slice(None))]

    return df_ERF


def extract_ensembles(df_ERF, ensemble_members):
    """Extract specific ensemble members from the ERF dataframe."""

    available_ens = df_ERF.columns.get_level_values('ensemble').unique()

    # Select ensemble members:
    if ensemble_members == 'all':
        ens_mems = slice(None)
    elif ((len(available_ens) == 1) and (ensemble_members != 'all')):
        # This is for SMILE_ESM scenarios that have multiple temperatures for
        # a single input forcing
        ens_mems = available_ens[0]
    elif ((len(available_ens) > 1) and (ensemble_members in available_ens)):
        # This is for NorESM scenarios that have multiple temperature
        # timeseries and multiple forcing timeseries, but a 1-1 correspondance
        # between the single ensemble number in the forcing and temperature.
        ens_mems = ensemble_members
    else:
        print(f'Invalid ensemble members {ensemble_members} for ensemble: '
              f'{df_ERF.columns.get_level_values("ensemble").unique()}')
        raise ValueError('Invalid ensemble member {ensemble_member} for data.')

    return df_ERF.loc[:, (slice(None), ens_mems)]


def extend_ERF_to_committed_year(df_ERF, year_committed_to,
                                 year_committed_from=None):
    """Extend the ERF dataframe to the committed year by holding
    the ERF constant from the specified year."""

    if year_committed_from is not None:
        df_ERF = df_ERF.loc[:year_committed_from]

    last_year = df_ERF.index.max()
    if year_committed_to > last_year:
        years_to_add = np.arange(last_year + 1, year_committed_to + 1)
        df_extension = pd.DataFrame(
            index=years_to_add,
            columns=df_ERF.columns,
            data=np.tile(df_ERF.loc[last_year].values,
                         (len(years_to_add), 1))
        )
        df_ERF = pd.concat([df_ERF, df_extension], axis=0)
    return df_ERF


def aggregate_missing_forcings(df_ERF, SUB_VAR_MAPPING=SUB_VAR_MAPPING):
    """Calculate aggregate forcings from sub-variables."""

    # Iterate over SUB_VAR_MAPPING to calculate any possible aggregates
    # This works because the mapping is ordered (GHG/OHF/Nat -> Ant -> Tot)
    for agg_var, sub_vars in SUB_VAR_MAPPING.items():
        # print('Preparing variable:', agg_var)

        # Check that this doesn't already exist first
        if agg_var not in df_ERF.columns:
            # print("..", agg_var, "doesn't yet exist: calculating it now.")

            # Check which sub-variables are present
            present_sub_vars = [v for v in sub_vars if v in df_ERF.columns]

            # If we have sub-variables, calculate the aggregate
            if present_sub_vars:
                # print("....", agg_var, 'has subvariables available:',
                #       present_sub_vars)

                # Check for missing sub-variables and warn if we have a partial
                # set
                missing_sub_vars = set(sub_vars) - set(present_sub_vars)

                if missing_sub_vars:
                    print(f"Warning: Missing sub-variables for {agg_var}: " +
                          f"{missing_sub_vars}. "
                          f"Aggregating only present variables:" +
                          f"{present_sub_vars}")
                else:
                    # Sum across the columns (variables) for each row.
                    df_ERF[agg_var] = df_ERF[present_sub_vars].sum(axis=1)
            else:
                pass
                # print("....", agg_var,
                #       ' has no sub-variables present: skipping aggregation.')
        else:
            pass
            # print("..", agg_var, 'already exists; skipping aggregation.')

    return df_ERF


def check_ensemble_matching(df_ERF):
    """Check that all variables have the same ensemble sets."""

    forc_var_names = sorted(df_ERF.columns.get_level_values(
        'variable').unique().to_list())

    # Check that the ensemble names are the same for all variables.
    dict_ensemble_names = {}
    for var in forc_var_names:
        # Select the variable 'OHF' from the dataframe, and get ensemble names.
        forc_subset = df_ERF.loc[:, (var, slice(None))]
        # print(forc_subset.head())
        forc_ens_names = sorted(
            list(forc_subset.columns.get_level_values("ensemble").unique()))
        dict_ensemble_names[var] = forc_ens_names

    check_ens = all(
        [dict_ensemble_names[var] == dict_ensemble_names[forc_var_names[0]]
            for var in forc_var_names]
        )

    if not check_ens:
        raise ValueError('Ensemble names are not the same for all variables.')

    return check_ens


def load_ERF_CMIP6(scenario, include_sub_vars=False):
    """Load the ERFs from Chris."""

    # ERF location
    here = Path(__file__).parent
    end = scenario.split('-')[-1]

    if include_sub_vars:
        file_ERF = here / f'../data/{scenario}/ERF/Chris/ERF_DAMIP_1000_1750-{end}_full.nc'
    else:
        file_ERF = here / f'../data/{scenario}/ERF/Chris/ERF_DAMIP_1000_1750-{end}.nc'

    # import ERF_file to xarray dataset and convert to pandas dataframe
    df_ERF = xr.open_dataset(file_ERF).to_dataframe()
    # assign the columns the name 'variable'
    df_ERF.columns.names = ['variable']

    # Drop total if exists
    if 'total' in df_ERF.columns:
        df_ERF = df_ERF.drop(columns='total')

    # Rename columns from file names to internal names
    RENAME_MAP = {'wmghg': 'GHG', 'other_ant': 'OHF', 'natural': 'Nat'}
    df_ERF = df_ERF.rename(columns=RENAME_MAP)

    # Calculate aggregates if missing (works both with/without sub-var toggle)
    df_ERF = aggregate_missing_forcings(df_ERF, SUB_VAR_MAPPING)

    # move the multi-index 'ensemble' level to a column,
    # and then set the 'ensemble' column to second column level
    df_ERF = df_ERF.reset_index(level='ensemble')
    df_ERF['ensemble'] = 'ens' + df_ERF['ensemble'].astype(str)
    df_ERF = df_ERF.pivot(columns='ensemble')

    # Check that the ensembles names are all matching across variables
    check_ensemble_matching(df_ERF)

    return df_ERF


def load_ERF_SMILE(scenario, regress_vars=None):
    """Load the data from John Nicklas for Thorne et al., analyis."""

    # ERF location
    here = Path(__file__).parent
    lower_scen = scenario.split('-')[-1].lower()
    file_ERF = here / f'../data/{scenario}/ERF_ESM1-2-LR_{lower_scen}.csv'

    df_ERF = pd.read_csv(file_ERF
                         ).rename(columns={'year': 'Year',
                                           'ERF_anthro': 'Ant',
                                           'ERF_natural': 'Nat',
                                           'ERF_other_human': 'OHF',
                                           'ERF_wmghg': 'GHG',
                                           'ERF_CO2': 'co2'
                                           }
                                  ).set_index('Year')
    # Drop the column named 'CO2' as this is concentrations, not ERF
    if 'CO2' in df_ERF.columns:
        df_ERF = df_ERF.drop(columns='CO2')

    # Calculate aggregates if missing
    df_ERF = aggregate_missing_forcings(df_ERF)

    # Add a second level to the column names ,and set the name of the second
    # level to 'ensemble'. Make the value of this 'single' for all of the
    # columns. This keeps the data structure the same as the multi-ensemble
    # data.
    df_ERF.columns = pd.MultiIndex.from_tuples(
        [(col, 'single') for col in df_ERF.columns],
        names=['variable', 'ensemble'])

    return df_ERF


def load_ERF_NorESM(scenario, regress_vars=None):
    """Load the data from John Nicklas for Thorne et al., analyis."""

    here = Path(__file__).parent
    volc_scen = scenario.split('-')[-1]

    if volc_scen == 'VolcConst':
        # In this case, the ERF components are all in one file, and there is
        # only one ensemble member.

        # ERF location
        here = Path(__file__).parent
        volc_scen = scenario.split('-')[-1]
        file_ERF = here / f'../data/{scenario}/ERF_NorESM_rcp45-{volc_scen}.csv'

        df_ERF = pd.read_csv(file_ERF
                             ).rename(columns={'year': 'Year',
                                               'ERF_anthro': 'Ant',
                                               'ERF_natural': 'Nat',
                                               'ERF_other_human': 'OHF',
                                               'ERF_wmghg': 'GHG',
                                               'ERF_CO2': 'co2'
                                               }
                                      ).set_index('Year')

        # Drop the column named 'CO2' as this is concentrations, not ERF
        if 'CO2' in df_ERF.columns:
            df_ERF = df_ERF.drop(columns='CO2')

        # Calculate aggregates if missing
        df_ERF = aggregate_missing_forcings(df_ERF)

        df_ERF.columns = pd.MultiIndex.from_tuples(
            [(col, 'single') for col in df_ERF.columns],
            names=['variable', 'ensemble'])

    elif volc_scen == 'Volc':
        # In this case, there is a single timeseries for the various anthro
        # components, and a whole ensemble of natural components.

        # ERF location for NATURAL
        file_ERF_natural = here / f'../data/{scenario}/ERF_natural_NorESM_rcp45-{volc_scen}.csv'
        df_ERF_natural = pd.read_csv(file_ERF_natural, index_col=0)
        # Rename index to 'Year'
        df_ERF_natural.index.name = 'Year'
        # Add a first level to the column names ,and set the name of the first
        # level to 'variable'. Make the value of this 'Nat' for all of the
        # columns. This keeps the data structure the same as the multi-ensemble
        # data.
        ens_num = df_ERF_natural.columns.to_list()

        df_ERF_natural.columns = pd.MultiIndex.from_tuples(
            [('Nat', col) for col in df_ERF_natural.columns],
            names=['variable', 'ensemble'])

        # There is only a single-level column name at the moment, with just
        # ensemble numbers. Move this to the second level, and add in the
        # first level the variable name]

        # ERF location for ANTHRO
        file_ERF_anthro = here / f'../data/{scenario}/ERF_anthro_NorESM_rcp45-{volc_scen}.csv'
        df_ERF_anthro = pd.read_csv(file_ERF_anthro
                             ).rename(columns={'year': 'Year',
                                               'ERF_anthro': 'Ant',
                                               'ERF_other_human': 'OHF',
                                               'ERF_wmghg': 'GHG',
                                               'ERF_CO2': 'co2'
                                               }
                                      ).set_index('Year')

        # Drop CO2 column if exists, as this is concentrations not ERF
        if 'CO2' in df_ERF_anthro.columns:
            df_ERF_anthro = df_ERF_anthro.drop(columns='CO2')

        # At the moment we have a single level column name, with just variable
        # names. Keep this in the first level, and add a second level with
        # 'ensemble' as the name, and '1' as the  value:
        df_ERF_anthro.columns = pd.MultiIndex.from_tuples(
            [(col, 'single') for col in df_ERF_anthro.columns],
            names=['variable', 'ensemble'])

        # We currently have (var, 'single') as the column names. We need to
        # copy this data to a new column, with the same variable name, but with
        # 'ensemble' as the second level, and '1' as the value. Copy it 60
        # times, so that the second levels are '0', '2', '3', ..., '59'.

        copies = []
        for ii in ens_num:
            df_ERF_anthro_repeat = df_ERF_anthro.copy()
            # Rename the values in the 'ensemble' column to 'ii'
            df_ERF_anthro_repeat = df_ERF_anthro_repeat.rename(
                columns={'single': str(ii)}, level=1)
            copies.append(df_ERF_anthro_repeat)
        df_ERF_anthro = pd.concat(copies, axis=1)        
        # Check that '(GHG, i)' column is the same, regardless, of the number i:
        # Check that I haven't made a mistake in copying.
        check_ens = all(
            [df_ERF_anthro['GHG', str(ens)].equals(df_ERF_anthro['GHG', '1'])
             for ens in ens_num])
        if not check_ens:
            raise ValueError(
                'Ensemble values are not the same for all variables.')

        # Combine the two dataframes
        df_ERF = pd.concat([df_ERF_anthro, df_ERF_natural], axis=1)

        # Calculate aggregates if missing
        # Stack to get variables as columns, aggregate, then unstack
        df_ERF = df_ERF.stack(level='ensemble')
        df_ERF = aggregate_missing_forcings(df_ERF)
        df_ERF = df_ERF.unstack(level='ensemble')

    return df_ERF


def load_ERF_SSP(scenario, regress_vars=['GHG', 'OHF', 'Nat']):
    """Load observed ERF with SSP extensions from Chris Smith."""
    # ERF location
    here = Path(__file__).parent
    scen = scenario.split('-')[-1]
    file_ERF = here / f'../data/observed_SSP-extension/ERF/ssp_forcing_fair2.1.3_cal1.4.5.nc'
    # import ERF_file to xarray dataset and convert to pandas dataframe
    df_ERF = xr.open_dataset(file_ERF).to_dataframe()
    # Rename the 'config' column in the index to be 'ensemble'
    df_ERF.index.names = ['year', 'scenario', 'ensemble']
    # assign the columns the name 'variable'
    df_ERF.columns.names = ['variable']
    # Move the ensemble column from the index to the columns
    df_ERF = df_ERF.unstack(level=2)
    # Change the values of the index 'year' to be rounded down (ie. remove 0.5)
    df_ERF.index = df_ERF.index.set_levels(
        df_ERF.index.levels[0].astype(int), level=0)

    # Select the scenario
    df_ERF = df_ERF.loc[(slice(None), scen.lower()), :]
    # Drop the scenario level from the index
    df_ERF.index = df_ERF.index.droplevel(1)

    # rename the variable columns
    df_ERF = df_ERF.rename(columns={'ghg': 'GHG',
                                    'natural': 'Nat',
                                    'anthro': 'Ant',
                                    'total': 'Tot'})

    # Add 'OHF' variable as the sum of 'aerosol', and 'other':
    df_ERF_OHF = df_ERF[['aerosol', 'other']].groupby(level='ensemble', axis=1
                                                      ).sum()
    # Add a new level to the columns, with the variable name 'OHF'
    df_ERF_OHF.columns = pd.MultiIndex.from_product(
        [['OHF'], df_ERF_OHF.columns])
    df_ERF = pd.concat([df_ERF, df_ERF_OHF], axis=1)

    # Check that the ensemble names are all matching across variables
    check_ensemble_matching(df_ERF)

    return df_ERF


def load_Temp(scenario, ensemble_members, start_pi, end_pi):
    """Load temperature scenario data, and remove pre-industrial baseline."""

    if 'observed-20' in scenario:
        df_temp = load_Temp_HadCRUT(scenario, start_pi, end_pi)
    elif 'observed_JK' in scenario:
        df_temp = load_Temp_JK(scenario, start_pi, end_pi)
    elif 'observed-SSP' in scenario:
        df_temp = load_Temp_HadCRUT('observed-2024', start_pi, end_pi)
    elif 'SMILE_ESM' in scenario:
        df_temp = load_Temp_SMILE(scenario, start_pi, end_pi)
    elif 'NorESM' in scenario:
        df_temp = load_Temp_NorESM(scenario, start_pi, end_pi)
    else:
        raise ValueError('Invalid scenario for temperature data.')

    # Select ensemble members:
    if ensemble_members == 'all':
        df_temp = df_temp
    elif ensemble_members in df_temp.columns.to_list():
        df_temp = df_temp[[ensemble_members]]
    else:
        print(f'Invalid ensemble members {ensemble_members} for ensemble:'
              + f'{df_temp.columns.to_list()}')
        raise ValueError(f'Invalid ensemble member {ensemble_member} for data.')

    # Remove pre-industrial baseline from temperature data
    df_temp = preindustrial_baseline(df_temp, start_pi, end_pi)

    return df_temp


def load_Temp_HadCRUT(scenario, start_pi, end_pi):
    """Load HadCRUT5 observations and remove PI baseline."""

    here = Path(__file__).parent
    temp_dir = here / f'../data/{scenario}/Temp/HadCRUT/'
    matches = sorted(
        temp_dir.glob('HadCRUT.*.analysis.ensemble_series.global.annual.csv')
    )
    if not matches:
        raise FileNotFoundError(
            f'No HadCRUT file found in {temp_dir} matching pattern '
            "'HadCRUT.*.analysis.ensemble_series.global.annual.csv'."
        )
    if len(matches) > 1:
        raise ValueError(
            f'Multiple HadCRUT files found in {temp_dir}; expected one match: '
            f'{[m.name for m in matches]}'
        )
    temp_ens_Path = matches[0]
    # read temp_Path into pandas dataframe, rename column 'Time' to 'Year'
    # and set the index to 'Year', keeping only columns with 'Realization' in
    # the column name, since these are the ensembles
    df_temp_Obs = pd.read_csv(temp_ens_Path,
                              ).rename(columns={'Time': 'Year'}
                                       ).set_index('Year'
                                                   ).filter(regex='Realization'
                                                            )

    # Rename the columns called "Realization_x" to just "x"
    df_temp_Obs.columns = [col.split(' ')[-1] for col in df_temp_Obs.columns]

    return df_temp_Obs


def load_Temp_JK(scenario, start_pi, end_pi):
    """Load multi-dataset observations from John Kennedy and
    remove PI baseline."""

    here = Path(__file__).parent
    temp_ens_Path = (
        f'../data/observed_JK-2024/Temp/JohnKennedy/' +
        'sst_pseudo.csv')

    # Read the csv file into a pandas dataframe. But note that, unlike the
    # HadCRUT dataset, there are no column names in this csv; we need the
    # first column to be named 'Year', and the rest to be named 'Realization_x'
    # for x=0,1,2,...,N where N is the number of columns that are not 'Year'.
    temp_ens_Path = here / temp_ens_Path
    df_temp_Obs = pd.read_csv(temp_ens_Path, header=None)
    n_ens = df_temp_Obs.shape[1] - 1
    col_names = ['Year'] + [str(realisation) for realisation in range(n_ens)]
    df_temp_Obs.columns = col_names
    df_temp_Obs = df_temp_Obs.set_index('Year')
    return df_temp_Obs

# def load_PiC_Old(n_yrs):
#     """Load piControl data from Stuart's ERF datasets."""
#     here = Path(__file__).parent
#     file_PiC = here / '../data/piControl/piControl.csv'

#     df_temp_PiC = pd.read_csv(file_PiC
#                           ).rename(columns={'year': 'Year'}
#                                    ).set_index('Year')
#     # model_names = list(set(['_'.join(ens.split('_')[:1])
#     #                         for ens in list(df_temp_PiC)]))

#     temp_IV_Group = {}

#     for ens in list(df_temp_PiC):
#         # pi Control data located all over the place in csv; the following
#         # lines strip the NaN values, and limits slices to the same length as
#         # observed temperatures
#         temp = df_temp_PiC[ens].dropna().to_numpy()[:n_yrs]

#         # Remove pre-industrial mean period; this is done because the models
#         # use different "zero" temperatures (eg 0, 100, 287, etc).
#         # An alternative approach would be to simply subtract the first value
#         # to start all models on 0; the removal of the first 50 years
#         # is used here in case the models don't start in equilibrium (and
#         # jump up by x degrees at the start, for example), and the baseline
#         # period is just defined as the same as for the observation PI
#         # period.
#         temp -= temp[:start_pi-end_pi+1].mean()

#         if len(temp) == n_yrs:
#             temp_IV_Group[ens] = temp

#     return pd.DataFrame(temp_IV_Group)


def load_Temp_SMILE(scenario, start_pi, end_pi):
    """Load the temperature data from John Nicklas for Thorne et al., analyis."""
    # Temp location
    here = Path(__file__).parent
    lower_scen = scenario.split('-')[-1].lower()
    file_temp = here / f'../data/{scenario}/ts_ESM1-2-LR_{lower_scen}.csv'

    df_temp = pd.read_csv(file_temp, index_col=0)

    # Rename index to 'Year'
    df_temp.index.name = 'Year'

    return df_temp


def load_Temp_NorESM(scenario, start_pi, end_pi):
    """Load the temperature data from John Nicklas for Thorne et al., analyis."""
    # Temp location
    here = Path(__file__).parent
    volc_scen = scenario.split('-')[-1]
    file_temp = here / f'../data/{scenario}/ts_NorESM_rcp45-{volc_scen}.csv'

    df_temp = pd.read_csv(file_temp, index_col=0)
    # Rename index to 'Year'
    df_temp.index.name = 'Year'

    return df_temp


def preindustrial_baseline(df_temp, start_pi, end_pi):
    """Remove PI baseline from temperature data."""
    # Check if offsets are explicitly bypassed.
    if start_pi is None or end_pi is None:
        return df_temp
        
    # Check that start_pi and end_pi are within the range of the data

    if ((start_pi in df_temp.index) and (end_pi in df_temp.index)):
        # Find PI offset that is the PI-mean of the median (HadCRUT best estimate)
        # of the ensemble and substract this from entire ensemble. Importantly,
        # the same offset is applied to the entire ensemble to maintain accurate
        # spread of HadCRUT (ie it is wrong to subtract the PI-mean for each
        # ensemble member from itself).
        ofst_Obs = df_temp.median(axis=1).loc[
            (df_temp.index >= start_pi) &
            (df_temp.index <= end_pi),
            ].mean(axis=0)
        df_temp -= ofst_Obs
    else:
        print(f'{start_pi} and {end_pi} not in {df_temp.index}')
        raise ValueError('PI offsetting period not in temperature data.')

    return df_temp


def load_PiC(scenario, n_yrs, start_pi, end_pi):
    if 'observed' in scenario:
        return load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    elif 'SMILE_ESM' in scenario:
        return load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    elif 'NorESM' in scenario:
        return load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    else:
        raise ValueError('Invalid scenario for piControl data.')


def load_PiC_CMIP6(n_yrs, start_pi, end_pi):
    """Create DataFrame of piControl data from .MAG files."""
    # Create list of all .MAG files recursively inside the directory
    # data/piControl/CMIP6. These files are simply as extracted from zip
    # downloaded from https://cmip6.science.unimelb.edu.au/results?experiment_id=piControl&normalised=&mip_era=CMIP6&timeseriestype=average-year-mid-year&variable_id=tas&region=World#download
    # (ie a CMIP6 archive for pre-meaned data, saving data/time.)
    here = Path(__file__).parent
    path_PiC = here / '../data/piControl/CMIP6/**/*.MAG'
    path_PiC = str(path_PiC)
    mag_files = sorted(glob.glob(path_PiC, recursive=True))
    dict_temp = {}
    for file in mag_files:
        # Adopt nomenclature format that matches earlier csv from Stuart
        group = file.split('/')[6]
        model = file.split('/')[-1].split('_')[3]
        member = file.split('/')[-1].split('_')[5]
        var = file.split('/')[-1].split('_')[1]
        experiment = file.split('/')[-1].split('_')[4]
        model_name = '_'.join([group, model, member, var, experiment])

        # use pymagicc to read the .MAG file
        df_PiC = pymagicc.io.MAGICCData(file).to_xarray().to_dataframe()
        # select only the data with keyword 'world' in the level 1 index
        df_PiC = df_PiC.xs('World', level=1)
        # replace the cftime index with an integer for the cftime year
        df_PiC.index = df_PiC.index.year

        temp = df_PiC.dropna().to_numpy().ravel()

        # Create multiple segments with 50% overlap from each other.
        # ie 0:173, 86:259, 172:345, etc
        segments = (temp.shape[0] - (n_yrs - n_yrs//2)) // (n_yrs//2)
        for s in range(segments):
            # print(s*(n_yrs//2), s*(n_yrs//2)+n_yrs)
            temp_s = temp[s*(n_yrs//2):s*(n_yrs//2)+n_yrs]
            if start_pi is not None and end_pi is not None:
                temp_s = temp_s - temp_s[:(end_pi-start_pi)].mean()
            dict_temp[
                f'{model_name}_slice-{s*(n_yrs//2)}:{s*(n_yrs//2)+n_yrs}'
                ] = temp_s

    return pd.DataFrame(dict_temp)


def filter_PiControl(df, timeframes):
    """Remove simulations that correspond poorly with observations."""
    dict_temp_PiC = {}
    for ens in list(df):
        # Establish inclusion condition, which is that the smoothed internal
        # variability of a CMIP6 ensemble must operate within certain bounds:
        # 1. there must be a minimum level of variation (to remove those models
        # that are clearly wrong, eg oscillating between 0.01 and 0 warming)
        # 2. they must not exceed a certain min or max temperature bound; the
        # 0.3 value is roughly similar to a 0.15 drift per century limit as
        # used in Haustein et al 2017, and Leach et al 2021.
        #
        # The final ensemble distribution are plotted against HadCRUT5 median
        # in gwi.py, to check that the percentiles of this median run are
        # similar to the percentiles on the entire CMIP5 ensemble. ie, if the
        # observed internal variability is essentially a sampling of the
        # climate each year, you would expect the percentiles over the observed
        # history to be similar to the percentiles across the ensemble (ie
        # multiple parallel realisations of reality) in any given year. We
        # allow the ensemble to be slightly broader, to reasonably allow for a
        # wider range of behaviours than we have so far seen in the real world.
        temp = df[ens].to_numpy()
        temp_ma_3 = moving_average(temp, 3)
        temp_ma_30 = moving_average(temp, 30)
        _cond = (
                 (max(temp_ma_3) < 0.3 and min(temp_ma_3) > -0.3)
                 and ((max(temp_ma_3) - min(temp_ma_3)) > 0.06)
                 and (max(temp_ma_30) < 0.1 and min(temp_ma_30) > -0.1)
                 )

        # Approve actual (ie not smoothed) data if the corresponding smoothed
        # data is approved.
        if _cond:
            dict_temp_PiC[ens] = temp

    return pd.DataFrame(dict_temp_PiC)


def moving_average(data, w):
    """Calculate a moving average of data with window size w."""
    # data_padded = np.pad(data, (w//2, w-1-w//2),
    #                      mode='constant', constant_values=(0, 1.5))
    return np.convolve(data, np.ones(w), 'valid') / w


def temp_signal(data, w, method):
    """Calculate the temperature signal as moving average of window w."""
    # Sensibly extend data (to avoid shortening the length of moving average)

    # These are the lengths of the pads to add before and after the data.
    start_pad = w//2
    end_pad = w-1-w//2

    if method == 'constant':
        # Choices are:
        # - 0 before 1850 (we are defining this as preindustrial)
        # - 1.5 between 2022 and 2050 (the line through the middle)
        data_padded = np.pad(data, (start_pad, end_pad),
                             mode='constant',
                             constant_values=(0, 1.5))

    elif method == 'extrapolate':
        # Add zeros to the beginning (corresponding to pre-industrial state)
        extrap_start = np.zeros(start_pad)

        # Extrapolate the final w years to the end of the data
        A = np.vstack([np.arange(w), np.ones(w)]).T
        coef = np.linalg.lstsq(A, data[-w:], rcond=None)[0]
        B = np.vstack([np.arange(w + end_pad), np.ones(w + end_pad)]).T
        extrap_end = np.sum(coef*B, axis=1)[-end_pad:]
        data_padded = np.concatenate((extrap_start, data, extrap_end), axis=0)

    return moving_average(data_padded, w)
    return np.convolve(data_padded, np.ones(w), 'valid') / w


def final_value_of_trend(temp):
    """Used for calculating the SR1.5 definition of present-day warming."""

    """Pass a 15-year long timeseries to this function and it will compute
    a linear trend through it, and return the final value of the trend. This
    corresponds to the SR15 definition of warming, if the 'present-day' in
    consideration is the final observable year; the SR15 definition would
    extrapolate this linear trend for 15 more years and take the mid-value,
    which is simply the end value of the first 15 years."""

    """SR1.5 definition: 'warming at a given point in time is defined as the
    global average temperatures for a 30-year period centred on that time,
    extrapolating into the future if necessary'. For these calculations,
    therefore, we take the final 15 years of the timeseries, take the trend
    through it, and then warming is given by the value of the trend in the
    final (present-day) year."""

    time = np.arange(temp.shape[0])
    fit = np.poly1d(np.polyfit(time, temp, 1))
    return fit(time)[-1]


def rate_func(array):
    # Instead of passing years array, just set the start year for the slice
    # to zero
    times = np.arange(array.shape[0])
    fit = np.polyfit(x=times, y=array, deg=1)
    return fit[0]


def rate_HadCRUT5(start_pi, end_pi, start_yr, end_yr, sigmas_all):
    # Load the HadCRUT5 dataset
    df_temp_Obs = load_Temp_HadCRUT(start_pi, end_pi, start_yr, end_yr)
    temp_Yrs = df_temp_Obs.index.values
    arr_temp_Obs = df_temp_Obs.values
    # Apply the function defs.rate_calc to each column of this dataframe

    dfs_rates = []
    for year in np.arange(1950, end_yr+1):
        print(year, end='\r')
        recent_years = ((year-9 <= temp_Yrs) * (temp_Yrs <= year))
        ten_slice = arr_temp_Obs[recent_years, :]

        with mp.Pool(n_workers()) as p:
            single_series = [ten_slice[:, ii]
                             for ii in range(ten_slice.shape[-1])]
            results = p.map(rate_func, single_series)
        forc_Rate_results = np.array(results)

        # Obtain statistics
        obs_rate_array = np.percentile(
            forc_Rate_results, sigmas_all, axis=0)
        dict_Results = {
            ('Obs', str(sigma)): obs_rate_array[sigmas_all.index(sigma)]
            for sigma in sigmas_all}
        df_rates_i = pd.DataFrame(
            dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
        df_rates_i.columns.names = ['variable', 'percentile']
        df_rates_i.index.name = 'Year'
        dfs_rates.append(df_rates_i)
    df_rates = pd.concat(dfs_rates, axis=0)
    return df_rates


def rate_ERF(end_yr, sigmas_all):
    rate_vars = ['Nat', 'GHG', 'OHF', 'Ant', 'Tot']
    df_forc = load_ERF_CMIP6()
    forc_Group_names = sorted(
        df_forc.columns.get_level_values('variable').unique())
    forc_Ens_names = sorted(
        df_forc.columns.get_level_values('ensemble').unique())
    forc_Yrs = df_forc.index.values

    # Apply the function defs.rate_calc to each column of this dataframe
    dfs_rates = []
    arr_forc = np.empty(
        (len(forc_Yrs), len(forc_Group_names)+2, len(forc_Ens_names)))
    # Move the data for each forcing group into a separate array dimension
    for vv in forc_Group_names:
        arr_forc[:, rate_vars.index(vv), :] = df_forc[vv].values
    arr_forc[:, rate_vars.index('Ant'), :] = (
        arr_forc[:, rate_vars.index('GHG'), :] +
        arr_forc[:, rate_vars.index('OHF'), :])
    arr_forc[:, rate_vars.index('Tot'), :] = (
        arr_forc[:, rate_vars.index('Ant'), :] +
        arr_forc[:, rate_vars.index('Nat'), :]
    )

    for year in np.arange(1950, end_yr+1):
        print(f'Calculating AR6-definition ERF rate: {year}', end='\r')
        recent_years = ((year-9 <= forc_Yrs) * (forc_Yrs <= year))
        ten_slice = arr_forc[recent_years, :, :]

        # Calculate AR6-definition ERF rate for each var-ens combination
        forc_Rate_results = np.empty(
            ten_slice.shape[1:])
        # Only include 'Ant'
        for vv in range(ten_slice.shape[1]):
            # Parallelise over ensemble members
            with mp.Pool(n_workers()) as p:
                single_series = [ten_slice[:, vv, ii]
                                 for ii in range(ten_slice.shape[2])]
                # final_value_of_trend is from src/definitions.py
                results = p.map(rate_func, single_series)
            forc_Rate_results[vv, :] = np.array(results)

        # Obtain statistics
        forc_rate_array = np.percentile(
            forc_Rate_results, sigmas_all, axis=1)
        dict_Results = {
            (var, str(sigma)):
            forc_rate_array[sigmas_all.index(sigma), rate_vars.index(var)]
            for var in rate_vars for sigma in sigmas_all
        }
        df_rates_i = pd.DataFrame(
            dict_Results, index=[f'{year-9}-{year} (AR6 rate definition)'])
        df_rates_i.columns.names = ['variable', 'percentile']
        df_rates_i.index.name = 'Year'
        dfs_rates.append(df_rates_i)
    print('')

    df_forc_rates = pd.concat(dfs_rates, axis=0)
    return df_forc_rates


def en_dash_ify(df):
    r"""Replace - with \N{EN DASH} in date danges in dataframes."""
    """This is required by ESSD formatting"""
    # List the rows with a - character in them
    rows_to_rename = [r for r in df.index if '-' in r]
    # Rename those rows, replacing the - with a \N{EN DASH}
    df.rename(
        index={r: r.replace('-', '\N{EN DASH}') for r in rows_to_rename},
        inplace=True)
    return df


def un_en_dash_ify(df):
    r"""Replace \N{EN DASH} with - in date danges in dataframes."""
    """For the purposes of saving to csv, where a normal '-' is likely safest
    for people to use, and most consistent with files from collaborators."""
    # List the rows with a - character in them
    rows_to_rename = [r for r in df.index if '\N{EN DASH}' in r]
    # Rename those rows, replacing the - with a \N{EN DASH}
    df.rename(
        index={r: r.replace('\N{EN DASH}', '-') for r in rows_to_rename},
        inplace=True)
    return df


def extra_vars(forc_vars):
    r"""Return diagnosable variables for a set of regressed variables."""
    # 1-way regression of Tot against Obs
    if 'Tot' in forc_vars and len(forc_vars) == 1:
        extra_vars = ['Res']

    # 2-way regression of Ant and Nat against Obs
    elif 'Ant' in forc_vars and 'Nat' in forc_vars and len(forc_vars) == 2:
        extra_vars = ['Tot', 'Res']

    # 3-way regression
    elif 'Ant' not in forc_vars and len(forc_vars) == 3:
        extra_vars = ['Ant', 'Tot', 'Res']

    else:
        # Raise error:
        raise ValueError('Invalid combination of variables for regression.')

    return extra_vars


def map_var_to_regression_aggregate(var_list_ERF, regress_vars):
    """
    Creates a mapping from each variable in var_list_ERF to its corresponding
    regression variable in regress_vars.
    """

    # First, remove the extra variables that are a higher-level combination of
    # the regression variables, so that the recursive search only finds the
    # correct regression variable, and not anything "above it" which will be
    # calculated as a linear combination later.
    extra_vars_list = extra_vars(regress_vars)
    reduced_mapping = {k: v for k, v in SUB_VAR_MAPPING.items()
                       if k not in extra_vars_list}

    def get_highest_parent(target, mapping):
        for parent, children in mapping.items():
            if target in children:
                return get_highest_parent(parent, mapping)
        return target

    # Explicitly ensure regression variables map to themselves
    # First, turn the inverse map function above into a dictionary map
    mapped = {}
    for v in var_list_ERF:
        mapped[v] = get_highest_parent(v, reduced_mapping)
    # Second, apply the identity mapping for regression variables
    for rv in regress_vars:
        mapped[rv] = rv

    return mapped


def check_steps(all_reg_ranges):
    """Check that the years are in steps of 1."""
    end_yrs = sorted([
        int(regressed_years.split('-')[1])
        for regressed_years in all_reg_ranges
    ])
    all_year_steps = all(np.diff(end_yrs) == 1)

    out_dict = {
        'check_bool': all_year_steps,
        'range': f'{min(all_reg_ranges)} to {max(all_reg_ranges)}',
    }

    return out_dict


def check_headlines(hy, end_regress, end_trunc):
    """Check that the headline years are in the correct format."""

    hy = hy.replace('end_regress', str(end_regress))
    hy = hy.replace('end_trunc', str(end_trunc))
    if hy in ['IGCC', 'end_regress', 'end_trunc']:
        return hy
    elif hy.isnumeric():
        return hy
    elif all([y.isnumeric() for y in hy.split(',')]):
        return hy
    else:
        raise ValueError('Invalid headline year format.')


def generate_headline_years(headline_years, end_regress, end_trunc):
    """Generate the headline years for the analysis."""

    headline_years = headline_years.replace('end_regress', str(end_regress))
    headline_years = headline_years.replace('end_trunc', str(end_trunc))

    if headline_years == 'end_regress':
        hl_years = [end_regress]
    elif headline_years == 'end_trunc':
        hl_years = [end_trunc]

    elif headline_years == 'IGCC':
        hl_years = [end_regress]

    elif headline_years.isnumeric():
        hl_years = [int(headline_years)]

    elif all([y.isnumeric() for y in headline_years.split(',')]):
        hl_years = [int(y) for y in headline_years.split(',')]
    else:
        print(headline_years)
        raise ValueError('Invalid headline year format.')

    return hl_years


def model_prior_warming(
        model_choice, df_params, df_forc):
    """Calculate prior (pre-constrained) warming."""
    """Parallelise over FaIR parameterisations, exploit vectorisation of
    FaIR model by running all forcings at once through it, and sample over all
    ERF and model uncertainty."""

    # Preparing lists to ensure that order of variables and ensemble members
    # are consistent across the different dataframes. I'm pretty sure that
    # pandas keeps column order consistent, but this is just extra safety
    var_list_ERF = sorted(df_forc.columns.get_level_values(
        "variable").unique().to_list())
    ens_list_ERF = df_forc.columns.get_level_values(
        "ensemble").unique().to_list()

    # Prepare results #########################################################
    # Total sub-ensemble size: multiple number of ensemble members for ERF:
    n_ens = len(ens_list_ERF)
    n_yrs = df_forc.shape[0]
    forc_Yrs = df_forc.index.to_numpy()  # Full forcing years

    # Prepare FaIR parameters for this particular model.
    params_FaIR = df_params[model_choice]
    params_FaIR.columns = pd.MultiIndex.from_product(
        [[model_choice], params_FaIR.columns])

    # Prepare results array for temperatures. Note that temp_Mod naming refers
    # to the fact that these temperatures are outputs from the model.
    temp_Mod_array = np.zeros(shape=(forc_Yrs.shape[0],
                                    #  -1 to get rid of Res
                                    #  len(var_list_ERF) + len(vars_extra) - 1,
                                     len(var_list_ERF),
                                     len(ens_list_ERF)))

    # Calculate temperatures from forcings for all ensembles at once,
    # leveraging FaIR's vectorisation
    for var in var_list_ERF:
        # FaIR won't run without emissions or concentrations, so specify
        # no zero emissions for input.
        emis_FAIR = fair.return_empty_emissions(
            df_to_copy=False,
            start_year=min(forc_Yrs), end_year=max(forc_Yrs), timestep=1,
            scen_names=ens_list_ERF)
        # Prepare a FaIR-compatible forcing dataframe
        forc_FaIR = fair.return_empty_forcing(
            df_to_copy=False,
            start_year=min(forc_Yrs), end_year=max(forc_Yrs), timestep=1,
            scen_names=ens_list_ERF)
        for ens in ens_list_ERF:
            forc_FaIR[ens] = df_forc.loc[:, (var, ens)].to_numpy()

        # Run FaIR. Convert output to numpy array for later regression.
        temp_All = fair.run_FaIR(emissions_in=emis_FAIR,
                                 forcing_in=forc_FaIR,
                                 thermal_parameters=params_FaIR,
                                 show_run_info=False)['T'].to_numpy()
        temp_Mod_array[:, var_list_ERF.index(var), :] = temp_All

    return temp_Mod_array
