"""Loading the input datasets: ERF, GMT (reference temperatures) and
piControl, chosen by dataset ID. See INPUT DATASETS below.

Used by gwi.py.
"""
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pymagicc
import xarray as xr

import src.infrastructure.run_log as run_log
import src.method.variables as variables
import src.method.warming_definitions as warming_definitions


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
DATA_DIR = Path(__file__).parent / '../../data'


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
            f'(got ERF member {erf_member!r}, GMT member {gmt_member!r}), '
            'with --erf-member and --gmt-member. In a run-gwi.sh recipe, set '
            'RECIPE_ERF_MEMBERS=paired_with_GMT_member.')


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
        GHG (see variables.SUB_VAR_MAPPING)

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
    include_sub_vars: also keep their sub-variables (see
        variables.SUB_VAR_MAPPING)
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
            if var in variables.SUB_VAR_MAPPING:
                direct_subs = variables.SUB_VAR_MAPPING[var]
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


def aggregate_missing_forcings(df_ERF,
                               SUB_VAR_MAPPING=variables.SUB_VAR_MAPPING):
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
                    run_log.note(f'Missing sub-variables for {agg_var}: '
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
    df_ERF = aggregate_missing_forcings(df_ERF, variables.SUB_VAR_MAPPING)

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


def load_PiC(picontrol_id, years, start_pi, end_pi):
    """Load piControl temperatures, in segments as long as the GMT record.

    picontrol_id: e.g. 'CMIP6' (the only one for now)
    years: the years of the GMT data, e.g. 1850 to 2025. Each segment is
        this long, and is indexed by these years.
    start_pi, end_pi: e.g. 1850, 1900, both included; None for no
        pre-industrial offset (each segment is then offset by its own mean)
    """
    # Load with the dataset's loader
    if picontrol_id == 'CMIP6':
        return load_PiC_CMIP6(years, start_pi, end_pi)
    else:
        raise ValueError(
            f'No piControl loader for {picontrol_id!r}.')


def load_PiC_CMIP6(years, start_pi, end_pi):
    """Slice the CMIP6 piControl simulations (.MAG files) into segments as
    long as `years`, overlapping by half, one column per segment.

    Each segment stands in for internal variability over `years`, and is
    indexed by them. Internal variability is a fluctuation about zero, so
    each segment is always made an anomaly, even when the other temperatures
    are not:
    - With a pre-industrial period, a segment is offset by its mean over the
      pre-industrial years, start_pi to end_pi with both included, as the
      modelled and observed temperatures are.
    - Without one (start_pi and end_pi None), the observations and the model
      keep their own levels (e.g. absolute temperatures, around 14 °C), and a
      regression constant takes up the difference. A segment is then offset
      by its own mean. Left in absolute temperature, every segment would
      fail filter_PiControl, which assumes anomalies.
    """
    years = np.asarray(years)
    n_yrs = len(years)
    if start_pi is not None and end_pi is not None:
        in_pi = (years >= start_pi) & (years <= end_pi)
    else:
        in_pi = np.ones(n_yrs, dtype=bool)  # the whole segment
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
            temp_s = temp_s - temp_s[in_pi].mean()
            dict_temp[
                f'{model_name}_slice-{s*(n_yrs//2)}:{s*(n_yrs//2)+n_yrs}'
                ] = temp_s

    return pd.DataFrame(dict_temp, index=years)


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
        temp_ma_3 = warming_definitions.moving_average(temp, 3)
        temp_ma_30 = warming_definitions.moving_average(temp, 30)
        _cond = (
                 (max(temp_ma_3) < 0.3 and min(temp_ma_3) > -0.3)
                 and ((max(temp_ma_3) - min(temp_ma_3)) > 0.06)
                 and (max(temp_ma_30) < 0.1 and min(temp_ma_30) > -0.1)
                 )

        # Approve actual (ie not smoothed) data if the corresponding smoothed
        # data is approved.
        if _cond:
            dict_temp_PiC[ens] = temp

    return pd.DataFrame(dict_temp_PiC, index=df.index)
