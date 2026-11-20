import sys
import os
import re
import mmap
import shutil
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import functools
import xarray as xr
import glob
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import pymagicc
import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################


# lru_cache makes Python work the answer out on the first call and hand back
# that same stored number on every later call. Several of the pool sites sit
# inside loops, so without it we would fork a scontrol subprocess hundreds of
# times per run. It also guarantees the count cannot change mid-run.
@functools.lru_cache(maxsize=1)
def n_workers():
    """Return the number of CPUs actually allocated to this job.

    os.cpu_count() reports the whole node rather than the allocation, so a pool
    sized by it oversubscribes the job's memory cgroup and the workers get
    OOM-killed.

    Worker death used to be unrecoverable: mp.Pool respawns the dead worker but
    never fails the task it was holding, so map()/imap() blocked forever and the
    job burned its whole walltime at zero CPU. The pool sites now use
    ProcessPoolExecutor, which raises BrokenProcessPool instead -- but sizing
    the pool correctly is still what stops the workers being killed in the
    first place.

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
        note(f'No SLURM allocation found, so falling back to {n_available} '
             f'workers. On a login/head node that spawns {n_available} '
             f'processes and degrades it for everyone else. Request an '
             f'interactive node instead (srun -p interactive -c 16 '
             f'--mem-per-cpu=2G --pty /bin/bash), or cap the workers '
             f'explicitly (GWI_NUM_WORKERS=4 python <script>.py).')

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

# INPUT DATASETS ##############################################################
#
# Each run attributes one ERF dataset against one GMT (reference temperature)
# dataset, with internal variability sampled from one piControl dataset.
#
# Every input is filed as data/<collection>/<kind>/<name>/, with a SOURCE.md
# on where it came from:
#     observed:   data/observed-<year>/<ERF or GMT>/<product>/
#                 e.g. data/observed-2025/GMT/HadCRUT/
#     simulated:  data/<experiment>/<ERF or GMT>/<model>/
#                 e.g. data/SMILE_ESM/ERF/MPI-ESM1.2-LR/
#     piControl:  data/piControl/GMT/<archive>/
#                 e.g. data/piControl/GMT/CMIP6/
#
# ERF and GMT datasets are chosen by ID:
#
#     <product or experiment>[-<year>][-<variant>]
#
# - product (observed) or experiment (simulated): uses '_', never '-'. A
#   simulated ID names the experiment, not the model (model names contain
#   dashes); its loader knows the model's folder, as each experiment has one
#   model for now.
# - year: for observed datasets only, the year their history ends.
# - variant: a version within the product, e.g. the scenario it follows.
#
# e.g. IGCC-2025, SSP_extension-2022-SSP245, SMILE_ESM-SSP245,
#      FutureVolc_ESM-rcp45-Volc
#
# A simulated dataset's ERF and GMT pair only with each other.
#
# To add a new year of an existing observed product (e.g. HadCRUT-2026), put
# its files in data/observed-2026/GMT/HadCRUT/, named as before (the version
# number in a HadCRUT file name may change): no code changes are needed.
# To add a new product, write its loader (load_ERF_<product> or
# load_GMT_<product>), and call that loader from load_ERF or load_GMT.

# Simulated datasets with one ERF member for each GMT member (with the same
# labels), which must be attributed one-to-one
DATASETS_WITH_PAIRED_MEMBERS = ['FutureVolc_ESM-rcp45-Volc']

# The data folder, next to src/
DATA_DIR = Path(__file__).parent / '../data'


@dataclass(frozen=True)
class DatasetMetadata:
    """What a dataset ID says about the dataset, and where its files are
    (see INPUT DATASETS). Not the data itself."""
    id: str                 # e.g. 'SSP_extension-2022-SSP245'
    kind: str               # 'ERF' or 'GMT'
    product: str            # e.g. 'SSP_extension'; for simulated datasets,
                            # the experiment, e.g. 'SMILE_ESM'
    year: Optional[int]     # e.g. 2022; None for simulated datasets
    variant: str            # e.g. 'SSP245'; '' if the product has none

    @property
    def is_observed(self):
        """True for observed datasets, which are filed by year."""
        return self.year is not None

    @property
    def folder(self):
        """e.g. data/observed-2022/ERF/SSP_extension/, or data/SMILE_ESM/ERF/
        (whose loader adds the model folder, MPI-ESM1.2-LR/)"""
        # Observed: filed by year, then kind, then product
        if self.is_observed:
            return (DATA_DIR / f'observed-{self.year}' / self.kind
                    / self.product)
        # Simulated: filed by experiment, then kind (then model)
        else:
            return DATA_DIR / self.product / self.kind


def dataset_metadata(dataset_id, kind):
    """The DatasetMetadata for a dataset ID, read from its parts:
        'SSP_extension-2022-SSP245' -> product 'SSP_extension', year 2022,
                                       variant 'SSP245'
        'IGCC-2025'                 -> product 'IGCC', year 2025
        'FutureVolc_ESM-rcp45-Volc' -> product 'FutureVolc_ESM',
                                       variant 'rcp45-Volc'

    kind: 'ERF' or 'GMT'
    """
    # Split the ID at each dash
    parts = dataset_id.split('-')

    # The product is the first part
    product = parts[0]
    rest = parts[1:]

    # The year, if there is one, is the next part: a four-digit number
    year = None
    if rest and len(rest[0]) == 4 and rest[0].isdigit():
        year = int(rest[0])
        rest = rest[1:]

    # Whatever is left is the variant, which may itself contain dashes ('' if
    # nothing is left)
    variant = '-'.join(rest)

    return DatasetMetadata(dataset_id, kind, product, year, variant)


def check_dataset_pairing(erf_id, gmt_id, erf_member, gmt_member):
    """Check, before loading anything, that the ERF and GMT datasets, and
    their members, can be attributed against each other.

    erf_id, gmt_id: e.g. 'IGCC-2025' and 'HadCRUT-2025'
    erf_member, gmt_member: 'all', or one member's label, e.g. '7'
    """
    erf = dataset_metadata(erf_id, 'ERF')
    gmt = dataset_metadata(gmt_id, 'GMT')

    # A simulated dataset (one whose ID has no year) pairs only with itself
    if (not erf.is_observed or not gmt.is_observed) and erf_id != gmt_id:
        raise ValueError(
            f'ERF dataset {erf_id!r} cannot be paired with GMT dataset '
            f'{gmt_id!r}: a simulated dataset (one whose ID has no year) '
            'pairs only with itself.')

    # Some simulated datasets need their members paired one-to-one
    if erf_id in DATASETS_WITH_PAIRED_MEMBERS and (
            erf_member == 'all' or erf_member != gmt_member):
        raise ValueError(
            f'{erf_id} has one ERF member for each GMT member, so they must '
            'be attributed one-to-one: give the same single member for both '
            f'(got ERF member {erf_member!r}, GMT member {gmt_member!r}). In '
            'run-gwi.sh, set ERF_MEMBERS=paired_with_GMT_member.')


def find_dataset_file(metadata, pattern):
    """The one file in the dataset's folder matching pattern.

    metadata: the DatasetMetadata, from dataset_metadata
    pattern: a file name, with * for any part that changes between versions,
        and any subfolder in front, e.g. 'sst_pseudo.csv',
        'HadCRUT.*.analysis.ensemble_series.global.annual.csv', or
        'MPI-ESM1.2-LR/ts_ESM1-2-LR_ssp245.csv'
    Stops if there is no such file (e.g. for an unknown product, or a year
    with no data), or more than one.
    """
    folder = metadata.folder
    matches = sorted(folder.glob(pattern))

    # Exactly one match: the file to read
    if len(matches) == 1:
        return matches[0]

    # Otherwise stop, saying what the folder holds
    if folder.is_dir():
        present = sorted(str(f.relative_to(folder)) for f in folder.rglob('*')
                         if f.is_file() and f.name != 'SOURCE.md')
    else:
        present = 'nothing: there is no such folder'
    raise FileNotFoundError(
        f'{metadata.kind} dataset {metadata.id!r}: expected one file matching '
        f'{pattern!r} in {folder.resolve()}, but found {len(matches)}. The '
        f'folder holds {present}.')


def describe_members(labels):
    """A short description of a list of member labels, for error messages:
    all of them if there are up to ten, otherwise the first and last few."""
    labels = list(labels)
    if len(labels) <= 10:
        return f'{labels}'
    first = ', '.join(map(str, labels[:5]))
    last = ', '.join(map(str, labels[-2:]))
    return f'{len(labels)} members: {first}, ..., {last}'


def load_ERF(erf_id, regress_vars, member, include_sub_vars=False):
    """Load an ERF metadata, keeping only the variables and member asked for.

    erf_id: e.g. 'IGCC-2025' or 'SMILE_ESM-SSP245' (see INPUT DATASETS)
    regress_vars: e.g. ['GHG', 'Nat', 'OHF']
    member: 'all', or one member's label, e.g. 'ens100' (IGCC) or '7'
    include_sub_vars: also keep the variables' sub-variables, e.g. co2 for
        GHG (see SUB_VAR_MAPPING)

    Returns a DataFrame indexed by year, with columns (variable, ensemble).
    """
    metadata = dataset_metadata(erf_id, 'ERF')

    # Load the whole dataset with its product's loader
    if metadata.product == 'IGCC':
        df_ERF = load_ERF_IGCC(metadata, include_sub_vars=include_sub_vars)
    elif metadata.product == 'SSP_extension':
        df_ERF = load_ERF_SSP_extension(metadata)
    elif metadata.product == 'SMILE_ESM':
        df_ERF = load_ERF_SMILE_ESM(metadata)
    elif metadata.product == 'FutureVolc_ESM':
        df_ERF = load_ERF_FutureVolc_ESM(metadata)
    else:
        raise ValueError(f'No ERF loader for {erf_id!r}.')

    # Extract just the required variables
    df_ERF = extract_ERF_variables(df_ERF,
                                   regress_vars,
                                   include_sub_vars=include_sub_vars)
    # Extract just the required ensemble member(s)
    df_ERF = extract_ERF_members(df_ERF, member, erf_id)

    return df_ERF


def extract_ERF_variables(df_ERF, regress_vars, include_sub_vars=False):
    """Keep only some variables of an ERF DataFrame.

    df_ERF: columns (variable, ensemble), as from the load_ERF_* loaders
    regress_vars: e.g. ['GHG', 'Nat', 'OHF']
    include_sub_vars: also keep their sub-variables (see SUB_VAR_MAPPING)
    """
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


def extract_ERF_members(df_ERF, member, erf_id):
    """Keep all members of an ERF DataFrame, or just one.
    (For GMT members, see extract_GMT_members.)

    df_ERF: columns (variable, ensemble), as from the load_ERF_* loaders
    member: 'all', or one member's label, e.g. 'ens100' (IGCC) or '7'.
        Datasets with a single ERF timeseries (SMILE_ESM,
        FutureVolc_ESM-rcp45-VolcConst) label it 'single'.
    erf_id: e.g. 'IGCC-2025', for error messages
    """
    # Keep everything
    if member == 'all':
        return df_ERF

    # Find the member, matched as text, because the member comes from the
    # command line but some labels are numbers (SSP_extension)
    available = df_ERF.columns.get_level_values('ensemble').unique()
    matches = [ens for ens in available if str(ens) == member]

    # Stop if it isn't there
    if not matches:
        raise ValueError(
            f'ERF member {member!r} is not in {erf_id}, which has '
            f'{describe_members(available)}.')

    # Keep that member, for every variable
    return df_ERF.loc[:, (slice(None), matches[0])]


def extend_ERF_to_committed_year(df_ERF, year_committed_to,
                                 year_committed_from=None):
    """Extend an ERF DataFrame to year_committed_to by holding the ERF
    constant from year_committed_from (or from its last year, if None).

    e.g. year_committed_from=2025, year_committed_to=2300
    """

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
    """Add any aggregate variable that is missing but whose sub-variables
    are all present, e.g. GHG from co2, ch4, n2o and halogen.

    df_ERF: one column per variable (the loaders call this before adding the
        ensemble column level)
    """

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
                    note(f'Missing sub-variables for {agg_var}: '
                         f'{missing_sub_vars}. Aggregating only the '
                         f'variables present: {present_sub_vars}.')
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
    """Check that every variable of an ERF DataFrame has the same members."""

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


def load_ERF_IGCC(metadata, include_sub_vars=False):
    """Load IGCC historical ERFs (supplied by Chris Smith).

    metadata: e.g. the DatasetMetadata for 'IGCC-2025'
    include_sub_vars: read the _full file, which also holds the
        sub-variables (only some years have one)
    """
    # Find the file: the _full file also holds the sub-variables
    full = '_full' if include_sub_vars else ''
    file_ERF = find_dataset_file(
        metadata, f'ERF_DAMIP_1000_1750-{metadata.year}{full}.nc')

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


def load_ERF_SMILE_ESM(metadata):
    """Load SMILE_ESM ERFs: the MPI-ESM1.2-LR large ensemble (supplied by
    John Nicklas, for Thorne et al.); a single timeseries per SSP, labelled
    'single'.

    metadata: e.g. the DatasetMetadata for 'SMILE_ESM-SSP245'
    """
    # Find the file for this SSP (named in lower case), in the model's folder
    file_ERF = find_dataset_file(
        metadata,
        f'MPI-ESM1.2-LR/ERF_ESM1-2-LR_{metadata.variant.lower()}.csv')

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


def load_ERF_FutureVolc_ESM(metadata):
    """Load FutureVolc_ESM ERFs: the NorESM1-M RCP4.5 large ensemble with
    future volcanic eruptions (Bethke et al., 2017; supplied by John Nicklas,
    for Thorne et al.).

    metadata: the DatasetMetadata for 'FutureVolc_ESM-rcp45-Volc' (a natural
        ERF member for each temperature member, labelled '1' to '60') or
        'FutureVolc_ESM-rcp45-VolcConst' (a single timeseries, labelled
        'single')
    """
    if metadata.variant == 'rcp45-VolcConst':
        # In this case, the ERF components are all in one file, and there is
        # only one ensemble member.
        file_ERF = find_dataset_file(
            metadata, f'NorESM1-M/ERF_NorESM_{metadata.variant}.csv')

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

    elif metadata.variant == 'rcp45-Volc':
        # In this case, there is a single timeseries for the various anthro
        # components, and a whole ensemble of natural components.

        # ERF location for NATURAL
        file_ERF_natural = find_dataset_file(
            metadata, f'NorESM1-M/ERF_natural_NorESM_{metadata.variant}.csv')
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
        file_ERF_anthro = find_dataset_file(
            metadata, f'NorESM1-M/ERF_anthro_NorESM_{metadata.variant}.csv')
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

    # Any other variant
    else:
        raise ValueError(
            f'Unknown FutureVolc_ESM variant {metadata.variant!r}: use '
            'rcp45-Volc or rcp45-VolcConst.')

    return df_ERF


def load_ERF_SSP_extension(metadata):
    """Load observed ERFs extended along an SSP (supplied by Chris Smith).

    metadata: e.g. the DatasetMetadata for 'SSP_extension-2022-SSP245'.
        The SSPs share their history to 2022 and diverge from 2023, so the
        data is filed with the 2022 data.
    """
    # Find the file, whose name holds the FaIR and calibration versions
    file_ERF = find_dataset_file(metadata, 'ssp_forcing_*.nc')

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

    # Select the SSP (named in lower case in the file), stopping if it isn't
    # there
    ssp = metadata.variant.lower()
    available = df_ERF.index.get_level_values('scenario').unique().tolist()
    if ssp not in available:
        raise ValueError(
            f'{metadata.variant!r} is not in {file_ERF.name}, which has '
            f"{[a.replace('ssp', 'SSP') for a in available]}.")
    df_ERF = df_ERF.loc[(slice(None), ssp), :]
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


def load_GMT(gmt_id, member, start_pi, end_pi):
    """Load a GMT metadata, keep the member asked for, and remove the
    pre-industrial baseline.

    gmt_id: e.g. 'HadCRUT-2025' or 'SMILE_ESM-SSP245' (see INPUT DATASETS)
    member: 'all', or one member's label, e.g. '7'
    start_pi, end_pi: the pre-industrial period, e.g. 1850, 1900; None for
        no offset

    Returns a DataFrame indexed by year, with one column per member.
    """
    metadata = dataset_metadata(gmt_id, 'GMT')

    # Load the whole dataset with its product's loader
    if metadata.product == 'HadCRUT':
        df_temp = load_GMT_HadCRUT(metadata)
    elif metadata.product == 'JK_multi':
        df_temp = load_GMT_JK_multi(metadata)
    elif metadata.product == 'SMILE_ESM':
        df_temp = load_GMT_SMILE_ESM(metadata)
    elif metadata.product == 'FutureVolc_ESM':
        df_temp = load_GMT_FutureVolc_ESM(metadata)
    else:
        raise ValueError(f'No GMT loader for {gmt_id!r}.')

    # Extract just the required ensemble member(s)
    df_temp = extract_GMT_members(df_temp, member, gmt_id)

    # Remove pre-industrial baseline from temperature data
    df_temp = preindustrial_baseline(df_temp, start_pi, end_pi)

    return df_temp


def extract_GMT_members(df_temp, member, gmt_id):
    """Keep all members of a GMT DataFrame, or just one.
    (For ERF members, see extract_ERF_members.)

    df_temp: one column per member, as from the load_GMT_* loaders
    member: 'all', or one member's label, e.g. '7'
    gmt_id: e.g. 'HadCRUT-2025', for error messages
    """
    # Keep everything
    if member == 'all':
        return df_temp

    # Stop if the member isn't there
    if member not in df_temp.columns:
        raise ValueError(
            f'GMT member {member!r} is not in {gmt_id}, which has '
            f'{describe_members(df_temp.columns)}.')

    # Keep that member
    return df_temp[[member]]


def load_GMT_HadCRUT(metadata):
    """Load the HadCRUT5 ensemble, members labelled '1' to '200'.

    metadata: e.g. the DatasetMetadata for 'HadCRUT-2025'
    """
    # Find the ensemble file, whose name holds the HadCRUT version
    temp_ens_Path = find_dataset_file(
        metadata, 'HadCRUT.*.analysis.ensemble_series.global.annual.csv')
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

    # Truncate to the final year in the dataset ID (the "vintage" of the
    # metadata, eg 'HadCRUT-2025' means observations through 2025).
    #
    # HadCRUT ships a row for the year currently in progress, which is a mean
    # over an incomplete set of months and is therefore not a meaningful annual
    # value for us (eg HadCRUT 5.1.0.0 in data/observed-2025/ carries a 2026 row
    # whose coverage uncertainty is ~30x that of the complete years).

    return truncate_to_end_year(df_temp_Obs, metadata, temp_ens_Path.name)


def truncate_to_end_year(df_temp_Obs, metadata, file_name):
    """Cut observations off at the year in their dataset ID, e.g. at 2025
    for 'HadCRUT-2025'.

    metadata: the DatasetMetadata of an observed dataset
    file_name: its file, for error messages
    """
    # Stop if the file ends before the year in the ID
    end_obs = metadata.year
    if end_obs not in df_temp_Obs.index:
        # The file is short of what the dataset ID claims: fail loudly
        # rather than silently analysing a shorter record than intended.
        raise ValueError(
            f"Dataset '{metadata.id}' declares observations through "
            f"{end_obs}, but {file_name} only covers "
            f"{df_temp_Obs.index.min()}-{df_temp_Obs.index.max()}.")

    # Drop any later rows
    return df_temp_Obs.loc[:end_obs]


def load_GMT_JK_multi(metadata):
    """Load John Kennedy's multi-dataset observational ensemble, members
    labelled '1' to '100'.

    metadata: e.g. the DatasetMetadata for 'JK_multi-2025'
    """
    # Find the file
    temp_ens_Path = find_dataset_file(metadata, 'sst_pseudo.csv')

    # Read the csv file into a pandas dataframe. But note that, unlike the
    # HadCRUT metadata, there are no column names in this csv; we need the
    # first column to be named 'Year', and the rest to be named 'Realization_x'
    # for x=1,2,...,N where N is the number of columns that are not 'Year'.
    df_temp_Obs = pd.read_csv(temp_ens_Path, header=None)
    n_ens = df_temp_Obs.shape[1] - 1
    col_names = ['Year'] + [str(realisation + 1)
                            for realisation in range(n_ens)]
    df_temp_Obs.columns = col_names
    df_temp_Obs = df_temp_Obs.set_index('Year')

    # Same cut-off as HadCRUT (see load_GMT_HadCRUT)
    return truncate_to_end_year(df_temp_Obs, metadata, temp_ens_Path.name)


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


def load_GMT_SMILE_ESM(metadata):
    """Load SMILE_ESM temperatures: the MPI-ESM1.2-LR large ensemble
    (supplied by John Nicklas, for Thorne et al.), members labelled '0' to
    '49'.

    metadata: e.g. the DatasetMetadata for 'SMILE_ESM-SSP245'
    """
    # Find the file for this SSP (named in lower case), in the model's folder
    file_temp = find_dataset_file(
        metadata, f'MPI-ESM1.2-LR/ts_ESM1-2-LR_{metadata.variant.lower()}.csv')

    df_temp = pd.read_csv(file_temp, index_col=0)

    # Rename index to 'Year'
    df_temp.index.name = 'Year'

    return df_temp


def load_GMT_FutureVolc_ESM(metadata):
    """Load FutureVolc_ESM temperatures: the NorESM1-M RCP4.5 large ensemble
    with future volcanic eruptions (Bethke et al., 2017; supplied by John
    Nicklas, for Thorne et al.), members labelled '1' to '60'.

    metadata: e.g. the DatasetMetadata for 'FutureVolc_ESM-rcp45-Volc'
    """
    # Find the file for this variant, in the model's folder
    file_temp = find_dataset_file(
        metadata, f'NorESM1-M/ts_NorESM_{metadata.variant}.csv')

    df_temp = pd.read_csv(file_temp, index_col=0)
    # Rename index to 'Year'
    df_temp.index.name = 'Year'

    return df_temp


def preindustrial_baseline(df_temp, start_pi, end_pi):
    """Remove the pre-industrial baseline from temperatures.

    df_temp: one column per member, indexed by year
    start_pi, end_pi: e.g. 1850, 1900; None to leave the data unchanged
    """
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
        raise ValueError(
            f'Pre-industrial offsetting period {start_pi}-{end_pi} is not '
            f'within the temperature data, which covers '
            f'{df_temp.index.min()}-{df_temp.index.max()}.')

    return df_temp


def load_PiC(picontrol_id, n_yrs, start_pi, end_pi):
    """Load piControl temperatures, in n_yrs-long slices.

    picontrol_id: e.g. 'CMIP6' (the only one for now)
    n_yrs: slice length, the number of years of GMT data, e.g. 176
    start_pi, end_pi: e.g. 1850, 1900; None for no offset
    """
    # Load with the dataset's loader
    if picontrol_id == 'CMIP6':
        return load_PiC_CMIP6(n_yrs, start_pi, end_pi)
    else:
        raise ValueError(
            f'No piControl loader for {picontrol_id!r}.')


def load_PiC_CMIP6(n_yrs, start_pi, end_pi):
    """Slice the CMIP6 piControl simulations (.MAG files) into n_yrs-long
    segments, overlapping by half, one column per segment."""
    # Create list of all .MAG files recursively inside the directory
    # data/piControl/GMT/CMIP6. These files are simply as
    # extracted from zip
    # downloaded from https://cmip6.science.unimelb.edu.au/results?experiment_id=piControl&normalised=&mip_era=CMIP6&timeseriestype=average-year-mid-year&variable_id=tas&region=World#download
    # (ie a CMIP6 archive for pre-meaned data, saving data/time.)
    dir_PiC = (DATA_DIR / 'piControl/GMT/CMIP6').resolve()
    path_PiC = str(dir_PiC / '**/*.MAG')
    mag_files = sorted(glob.glob(path_PiC, recursive=True))
    dict_temp = {}
    for file in mag_files:
        # Adopt nomenclature format that matches earlier csv from Stuart.
        # The group is the institution, read from the path inside
        # data/piControl/GMT/CMIP6/, which is laid out as
        # average-year-mid-year/CMIP6/<activity>/<institution>/<model>/...
        group = Path(file).relative_to(dir_PiC).parts[3]
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
    """Remove piControl segments whose variability is implausible next to
    the observations.

    df: one column per segment, as from load_PiC
    timeframes: e.g. [1, 3, 30] (not used at present)
    """
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


def _trend_weights(n):
    """Ordinary-least-squares slope weights for a fixed x-axis of 0..n-1.

    Fitting a straight line by least squares gives the slope

        b = sum_i (x_i - xbar)(y_i - ybar) / sum_i (x_i - xbar)^2

    Here x is always the integers 0..n-1, so everything involving x can be
    worked out once: with w_i = (x_i - xbar) / sum_j (x_j - xbar)^2 the slope
    is just the dot product w . y. That turns a per-series curve fit into a
    single weighted sum, which numpy can apply across a whole array at once.

    Returns (w, dx), where dx = x_last - xbar is the offset needed to evaluate
    the fitted line at its final point (see final_value_of_trend_vectorised).
    """
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    return xc / (xc @ xc), x[-1] - x.mean()


def final_value_of_trend_vectorised(array):
    """Vectorised final_value_of_trend, applied along axis 0.

    Takes an array of shape (n, ...) and returns shape (...), fitting an
    independent trend down axis 0 for every remaining position. This replaces
    looping final_value_of_trend over each series individually.

    A least-squares line passes through (xbar, ybar), so its value at the final
    point is ybar + b * (x_last - xbar).

    NOTE: this agrees with final_value_of_trend to ~1e-7 rather than exactly.
    np.polyfit solves the least-squares problem via SVD (numpy.linalg.lstsq)
    instead of the closed form above; the two are mathematically identical but
    order the floating-point operations differently. The residual difference is
    float32 rounding on values of order 1.
    """
    w, dx = _trend_weights(array.shape[0])
    return array.mean(axis=0) + np.tensordot(w, array, axes=(0, 0)) * dx


def rate_func(array):
    # Instead of passing years array, just set the start year for the slice
    # to zero
    times = np.arange(array.shape[0])
    fit = np.polyfit(x=times, y=array, deg=1)
    return fit[0]


def rate_func_vectorised(array):
    """Vectorised rate_func, applied along axis 0.

    Takes an array of shape (n, ...) and returns shape (...), fitting an
    independent trend down axis 0 for every remaining position and returning
    its slope. This replaces looping rate_func over each series individually.

    The slope is the dot product of the precomputed weights with the data --
    see _trend_weights for the derivation.

    NOTE: this agrees with rate_func to ~1e-16, i.e. float64 machine epsilon,
    which is far tighter than final_value_of_trend_vectorised manages. The
    slope is a single weighted sum, whereas the trend endpoint additionally
    reconstructs the intercept, and that extra arithmetic on float32 inputs is
    what costs the latter its precision.
    """
    w, _ = _trend_weights(array.shape[0])
    return np.tensordot(w, array, axes=(0, 0))


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


def shared_results_array(shape, dtype=np.float32):
    """Return an empty array that forked worker processes can write into.

    Why this exists
    ---------------
    A worker process normally has its own private memory, so anything it
    hands back to the parent has to be copied: packed up in the worker, sent
    down a pipe, and rebuilt in the parent. For a moment the same numbers
    exist twice. With one worker per FaIR parameterisation, all finishing at
    roughly the same time, that copying is the single largest memory cost of
    a GWI run. Measured on a 4 GB result: 4.0x its size when each worker
    returned its own block and the parent joined them with np.concatenate,
    3.5x when the parent wrote the returned blocks into a pre-made array,
    and 1.0x using this.

    How it works
    ------------
    The array is placed in memory the operating system marks as shareable,
    so when the parent forks a worker, the worker sees the very same memory
    rather than a private copy of it. Anything a worker writes is
    immediately visible to the parent and to every other worker. Nothing is
    sent down a pipe and workers need not return anything at all.

    Two conditions, both of which the caller must arrange:

      1. Create the array BEFORE creating the worker processes. Workers
         inherit it as they are forked; an array made afterwards is not
         shared.
      2. Start the workers with the 'fork' method, which is the Linux
         default but is worth stating explicitly, e.g.
         ProcessPoolExecutor(n, mp_context=mp.get_context('fork')).

    Each worker should write only to its own part of the array. Two workers
    writing to the same elements would race, exactly as two threads would.

    The memory holds no filename and belongs to no process in particular:
    the operating system reclaims it once the parent and all its workers
    have exited, including if they are killed outright. So there is nothing
    to clean up and nothing that can be left behind on the node.
    """
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    _check_and_report_memory(n_bytes, shape)
    # mmap(-1, ...) asks the OS for anonymous shared memory: no file and no
    # name. The request itself is cheap -- the pages are only charged to the
    # job as they are written -- but it can still fail outright if the array
    # is larger than the whole machine, since shareable memory is backed by
    # the node's RAM. Hence both this and the check above.
    try:
        buffer = mmap.mmap(-1, n_bytes)
    except OSError as err:
        raise MemoryError(
            f'Could not reserve {n_bytes / 1024**3:.1f} GiB for the results '
            f'of this run. That is more memory than this compute node has in '
            f'total, so no memory request would make it fit; lower the '
            f'number of samples, or use a larger node.') from err
    return np.ndarray(shape, dtype=dtype, buffer=buffer)


def _job_cgroup_value(filename):
    """Read one of Slurm's memory counters for this job, in bytes.

    Slurm confines each job to a 'cgroup', a kernel accounting box, and
    reports its memory through small files. filename is the counter's name
    under cgroup v2. Nodes running the older cgroup v1, such as ARC's, keep
    the same counters elsewhere under other names, and those are read
    instead. Returns None when not running under Slurm, or if neither can be
    read.

    In the pipeline: behind job_memory_limit and job_memory_peak, which
    gwi.py uses to report memory in its log, and to refuse up front a run
    whose results cannot fit.

    Replaces a version that read only the cgroup v2 layout, so on ARC both
    came back as None: the log showed '?' and the up-front check was
    skipped.
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        return None
    v1_names = {'memory.max': 'memory.limit_in_bytes',
                'memory.peak': 'memory.max_usage_in_bytes'}
    paths = [
        f'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job_id}/'
        f'{filename}',
        f'/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/'
        f'{v1_names[filename]}',
        ]
    for path in paths:
        try:
            with open(path) as fh:
                return int(fh.read())
        except (OSError, ValueError):
            continue
    return None


def job_memory_limit():
    """Return the memory this job may use, in bytes, or None outside Slurm.

    Set by --cpus-per-task x --mem-per-cpu. Exceeding it is what gets a run
    killed, so it is the number worth checking against before starting work.
    """
    return _job_cgroup_value('memory.max')


def job_memory_peak():
    """Return the most memory this job has used at once, in bytes.

    This is the kernel's own high-water mark for the job. It is a better
    figure than the MaxRSS that `sacct` reports, which adds up the memory of
    every worker process separately (so double-counts whatever they share
    with the parent) and only samples every 10 seconds (so misses brief
    peaks). This counts shared memory once and misses nothing.
    """
    return _job_cgroup_value('memory.peak')


def describe_bytes(n):
    """Format a number of bytes as GiB, or '?' if it is not known."""
    return '?' if n is None else f'{n / 1024**3:.1f} GiB'


def tagged_folder(base, tag):
    """Return the output folder for a run, optionally tagged.

    Keeps separate pieces of work from overwriting one another, and from
    being swept up together by the combination script. Without a tag the
    folders are the plain ones they have always been, so existing behaviour
    and existing directories are untouched:

        tagged_folder('results', None)          -> 'results'
        tagged_folder('results', 'Thorne2025')  -> 'results_Thorne2025'
        tagged_folder('plots', 'IGCC')          -> 'plots_IGCC'

    Pass the same --output-tag to gwi.py and to combine_results_iterations.py
    so that the second reads what the first wrote.
    """
    if tag in (None, '', 'none'):
        return base
    if not re.fullmatch(r'[A-Za-z0-9._-]+', tag):
        raise ValueError(
            f'Invalid --output-tag {tag!r}. It becomes part of a directory '
            f'name, so use only letters, digits, dots, dashes and '
            f'underscores.')
    return f'{base}_{tag}'


_run_notes = []


def note(message):
    """Record something about this run that the reader should know.

    For things that change what the run does but do not stop it: a year range
    clipped to the available data, a setting that disables a safeguard, a
    resource that could not be determined. They arise in several places and
    at several stages -- argument parsing, data loading, range checking --
    and printing them where they arise would scatter them through the log
    above the first section header. Collecting them means log_notes() can put
    them together, under a heading, in a predictable place.

    Not for errors. Anything that stops the run should raise, with the detail
    in the exception message where a traceback will carry it.
    """
    _run_notes.append(message)


def log_notes():
    """Print everything note() collected, as its own section."""
    log_section('Notes and adjustments')
    if not _run_notes:
        print('None', flush=True)
        return
    for message in _run_notes:
        print(message, flush=True)


def log_section(title, width=76):
    """Start a new section of the run log.

    The log is long and is usually read after the fact, often to work out why
    a run died, so it is broken into labelled sections that can be found by
    eye or with grep.
    """
    print(f'\n── {title.upper()} '.ljust(width, '─'), flush=True)


def log_slurm_configuration():
    """Report the resources Slurm gave this job.

    Every one of these has at some point been the explanation for why two
    runs of the same configuration behaved differently: which node (their
    CPUs and memory bandwidth differ), how much memory (what gets a run
    killed), how many CPUs (the pool size), and the time limit (what turns a
    slow run into a TIMEOUT). The job ID ties the log back to `sacct` and to
    the batch directory it ends up in.
    """
    log_section('Slurm configuration')
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id is None:
        print(f'Not running under Slurm (host {os.uname().nodename})')
        return
    cpu_model = '?'
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    print(f'Job ID: {job_id}'
          f'{" (" + os.environ["SLURM_JOB_NAME"] + ")" if os.environ.get("SLURM_JOB_NAME") else ""}\n'
          f'Node: {os.uname().nodename}, partition '
          f'{os.environ.get("SLURM_JOB_PARTITION", "?")}\n'
          f'CPU: {cpu_model}\n'
          f'CPUs allocated to job: {n_workers()} '
          f'(node has {os.cpu_count()})\n'
          f'Memory allocated to job: {describe_bytes(job_memory_limit())}\n'
          f'Walltime limit: '
          f'{os.environ.get("SLURM_JOB_END_TIME") and _walltime_remaining() or "?"}',
          flush=True)


def _walltime_remaining():
    """Return the job's time limit as H:MM:SS, from Slurm's end-time stamp."""
    try:
        import datetime as _dt
        end = int(os.environ['SLURM_JOB_END_TIME'])
        start = int(os.environ.get('SLURM_JOB_START_TIME', end))
        return str(_dt.timedelta(seconds=end - start))
    except (KeyError, ValueError):
        return '?'


def _check_and_report_memory(n_bytes, shape):
    """Report what this run will cost, and stop if it cannot possibly work.

    Without this the run would start, spend minutes emulating, and only then
    be killed by the operating system with no explanation of what went wrong.
    Checking first turns that into an immediate, readable message, and
    putting the figures in the log means a run that later dies can still be
    understood from its log alone.

    The peak comes to roughly TWICE the size of the results array, because
    np.percentile copies whatever it is given (see percentile_threaded). So
    needing more than the allocation for the array alone is hopeless and
    stops the run; needing more than the allocation for twice it will almost
    certainly die later at the percentile step, which is worth saying plainly
    but not worth refusing outright, since the earlier stages may still be of
    interest.
    """
    limit = job_memory_limit()
    advice = ('lower the number of samples, or raise CPU_MEM in '
              'run-gwi.sh')

    if limit is not None and n_bytes > limit:
        raise MemoryError(
            f'This run cannot fit in the memory it has been given. The '
            f'results array alone needs {describe_bytes(n_bytes)} '
            f'({shape[0]} years x {shape[1]} variables x {shape[2]:,} '
            f'members), and this job may use only {describe_bytes(limit)}. '
            f'To fix, {advice}.')

    if limit is None:
        verdict = 'cannot tell, not running under Slurm'
    elif 2 * n_bytes > limit:
        verdict = (f'LIKELY -- the expected peak is more than the allocation, '
                   f'so this run will probably be killed at the percentile '
                   f'step. To avoid, {advice}')
    else:
        verdict = 'not expected'

    log_section('Memory usage anticipated')
    print(f'Results array: {describe_bytes(n_bytes)} '
          f'({shape[0]} years x {shape[1]} variables x {shape[2]:,} members)\n'
          f'Expected memory peak: near {describe_bytes(2 * n_bytes)} '
          f'(np.percentile copies the input from results array, doubling the '
          f'peak)\n'
          f'Slurm allocated memory for job: {describe_bytes(limit)}\n'
          f'Memory errors: {verdict}', flush=True)


def report_memory_used(results_nbytes):
    """Report what the run actually cost, to close the loop on the estimate.

    The partner of the 'Memory usage anticipated' block printed when the
    results array was made. Printing both means a log can be read on its own
    to see whether the estimate was right, without going back to Slurm's
    accounting -- which is purged eventually, and which in any case
    double-counts memory shared between worker processes.
    """
    peak, limit = job_memory_peak(), job_memory_limit()
    if peak is None:
        print('Not available (not running under Slurm)')
        return
    share = f' ({peak / limit:.0%} used)' if limit else ''
    print(f'Peak memory (usage): {describe_bytes(peak)} of '
          f'{describe_bytes(limit)} allocated{share}\n'
          f'Peak memory (vs results array): '
          f'{peak / results_nbytes:.2f}x the {describe_bytes(results_nbytes)} '
          f'results array, against the 2.00x anticipated', flush=True)


def attribution_output_vars(df_forc, regress_vars):
    """Return the variable names along axis 1 of the attribution array.

    These are the forcing variables actually present in df_forc, followed by
    the aggregates diagnosed after the regression (see extra_vars) that are
    not already among them. The order matters: it is the order of the
    variable axis of temp_Att_Results, and of the output CSV columns.

    Both the emulation and the code that sizes the shared results array need
    this list, so it lives here rather than being worked out twice.
    """
    forc_vars = sorted(
        df_forc.columns.get_level_values('variable').unique().to_list())
    diagnosed = extra_vars(regress_vars)
    return forc_vars + [v for v in diagnosed if v not in forc_vars]


def ensemble_members_per_model(df_forc, df_temp_Obs, df_temp_PiC):
    """Return how many ensemble members one FaIR parameterisation produces.

    Each emulation crosses every sampled ERF ensemble member with every
    sampled observational temperature member and every sampled piControl
    member, so the count is simply the product of the three.
    """
    n_erf = len(df_forc.columns.get_level_values('ensemble').unique())
    return n_erf * df_temp_Obs.shape[1] * df_temp_PiC.shape[1]


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
        raise ValueError(
            f'Invalid headline year format: {headline_years!r}.')

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


def contiguous_slice(mask):
    """Convert a boolean mask with no gaps in it into an equivalent slice.

    "No gaps" is a property of a SINGLE mask: the elements it selects must be
    adjacent to one another, so that start:stop picks out exactly the same
    set. It says nothing about how one mask relates to the next. The AR6 rate
    windows, for instance, overlap heavily from year to year (1950-1959, then
    1951-1960, ...) but each individual window is still an unbroken run and so
    converts cleanly. A mask selecting, say, indices 100, 101 and 105 could
    not, and raises.

    The headline and rate calculations select year windows with masks of the
    form (lo <= yrs) & (yrs <= hi), which by construction have no gaps.
    Indexing an array with such a mask is fancy indexing, so it COPIES the
    selected block; indexing with the equivalent slice returns a view.

    That matters at scale: on the full attribution array (~69 GB at
    samples=80, GMT-all) each mask copy costs ~1.1 s and several GB, and the
    AR6 rate loop performs one per year from 1950.

    Slicing axis 0 of a C-contiguous array yields a C-contiguous view with the
    same strides as the copy would have had, so every downstream reduction
    sees an identical memory layout and results are bitwise unchanged.

    Raises ValueError if the mask has gaps, rather than silently returning a
    slice that selects different elements.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return slice(0, 0)
    start, stop = int(idx[0]), int(idx[-1]) + 1
    if idx.size != stop - start:
        raise ValueError(
            'contiguous_slice requires a mask selecting a contiguous run; '
            f'got {idx.size} elements spanning {start}:{stop}.')
    return slice(start, stop)


def percentile_threaded(array, q, axis, n_threads=None):
    """np.percentile, parallelised by splitting the leading axis over threads.

    numpy is not automatically parallel: it only threads operations that
    dispatch to a threaded BLAS/LAPACK (matmul, decompositions). Reductions and
    sorts -- and so np.percentile, which is built on np.partition -- are plain
    single-threaded C loops.

    Those loops do release the GIL, however, so ordinary threads parallelise
    them. Threads are used rather than processes because the attribution array
    runs to tens of GB: sharing it costs nothing, whereas mp.Pool would pickle
    a copy per worker.

    Results are bitwise identical to np.percentile(array, q, axis), because
    each leading-axis slice is reduced independently of the others and the
    per-slice computation is untouched.

    Parameters
    ----------
    array : ndarray, reduced along `axis`, split along axis 0.
    q, axis : as np.percentile. `axis` must not be 0, which is the split axis.
    n_threads : defaults to n_workers().
    """
    if axis == 0:
        raise ValueError(
            'percentile_threaded splits along axis 0, so it cannot also '
            'reduce along it; use np.percentile directly.')

    if n_threads is None:
        n_threads = n_workers()
    n_chunks = min(n_threads, array.shape[0])

    # Not worth the thread overhead, and keeps a trivially correct fallback.
    if n_chunks <= 1:
        return np.percentile(array, q, axis=axis)

    # np.percentile prepends the q axis, so the array's axis 0 becomes axis 1
    # of the result (axis != 0 is guaranteed above, so it is never consumed).
    out = None
    # Chunk with slices, NOT index arrays. array[slice] is a view, whereas
    # fancy indexing with np.array_split(np.arange(n), k) would COPY each
    # chunk -- and with every thread holding its copy at once that doubles the
    # peak memory of the whole call. On the full attribution array (~69 GB at
    # samples=80, GMT-all) that extra copy was enough to OOM a 224 GB job.
    bounds = np.linspace(0, array.shape[0], n_chunks + 1).astype(int)
    chunks = [slice(bounds[i], bounds[i + 1]) for i in range(n_chunks)]

    def _worker(sl):
        return sl, np.percentile(array[sl], q, axis=axis)

    with ThreadPoolExecutor(n_chunks) as pool:
        for sl, result in pool.map(_worker, chunks):
            if out is None:
                out = np.empty(
                    (result.shape[0], array.shape[0]) + result.shape[2:],
                    dtype=result.dtype)
            out[:, sl, ...] = result

    return out

