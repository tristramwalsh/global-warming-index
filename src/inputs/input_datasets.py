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
# load_GMT_<product>), returning a DataArray as the others do, and call that
# loader from load_ERF or load_GMT.

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
    """Load an ERF dataset, keeping only the variables and member asked for.

    erf_id: e.g. 'IGCC-2025' or 'SMILE_ESM-SSP245' (see INPUT DATASETS)
    regress_vars: e.g. ['GHG', 'Nat', 'OHF']
    member: 'all', or one member's label, e.g. 'ens100' (IGCC) or '7'
    include_sub_vars: also keep the variables' sub-variables, e.g. co2 for
        GHG (see variables.SUB_VAR_MAPPING)

    Returns the forcing, a DataArray (year, erf_member, forcing_variable):
    year first, then members, then variables, as every input is (see
    gwi.py's NAMING), the variables sorted, and the ERF members in the
    order their product's loader gives them (the order of the e axis).
    """
    metadata = dataset_metadata(erf_id, 'ERF')

    # Load the whole dataset with its product's loader
    if metadata.product == 'IGCC':
        erf = load_ERF_IGCC(metadata, include_sub_vars=include_sub_vars)
    elif metadata.product == 'SSP_extension':
        erf = load_ERF_SSP_extension(metadata)
    elif metadata.product == 'SMILE_ESM':
        erf = load_ERF_SMILE_ESM(metadata)
    elif metadata.product == 'FutureVolc_ESM':
        erf = load_ERF_FutureVolc_ESM(metadata)
    else:
        raise ValueError(f'No ERF loader for {erf_id!r}.')

    # Add any aggregate the file lacks but whose parts it has (e.g. GHG
    # from co2, ch4, n2o and halogen)
    erf = aggregate_missing_forcings(erf)
    # Keep just the variables asked for, and the member(s)
    erf = extract_ERF_variables(erf, regress_vars,
                                include_sub_vars=include_sub_vars)
    erf = extract_ERF_members(erf, member, erf_id)

    # The variables sorted, and the dimensions year first
    return erf.sortby('forcing_variable').transpose(
        'year', 'erf_member', 'forcing_variable').rename('erf')


def forcing_from_table(df_ERF, erf_member='single'):
    """One ERF member's forcing, from a table read from a CSV file.

    df_ERF: a DataFrame indexed by year, one column per variable, as the
        simulated products' files hold them
    erf_member: the member's label ('single' for a dataset with one)
    Returns a DataArray (year, erf_member, forcing_variable), with one
    member.
    """
    return xr.DataArray(
        df_ERF.to_numpy()[:, np.newaxis, :],
        dims=('year', 'erf_member', 'forcing_variable'),
        coords={'year': df_ERF.index.to_numpy(),
                'erf_member': [erf_member],
                'forcing_variable': df_ERF.columns.to_list()})


def extract_ERF_variables(erf, regress_vars, include_sub_vars=False):
    """Keep the regressed variables of the forcing, and, if asked, their
    sub-variables (those the dataset has).

    erf: the forcing, a DataArray (year, erf_member, forcing_variable), as
        from the load_ERF_* loaders
    regress_vars: e.g. ['GHG', 'Nat', 'OHF']
    include_sub_vars: also keep their sub-variables, e.g. co2, ch4, n2o and
        halogen for GHG (see variables.SUB_VAR_MAPPING), and theirs in turn
    """
    available_vars = erf['forcing_variable'].to_numpy().tolist()
    # Stop if a regressed variable isn't there
    missing_vars = set(regress_vars) - set(available_vars)
    if missing_vars:
        raise ValueError(
            'The following regression variables are not available in the '
            f'ERF data: {missing_vars}')

    # The variables to keep: the regressed ones ...
    vars_to_keep = list(regress_vars)
    # ... and, if asked, every variable below each of them in the mapping
    # (e.g. GHG's co2, ch4, n2o and halogen), found by walking down it
    if include_sub_vars:
        to_visit = list(regress_vars)
        while to_visit:
            sub_vars = variables.SUB_VAR_MAPPING.get(to_visit.pop(), [])
            vars_to_keep += sub_vars
            to_visit += sub_vars
    # ... of those the dataset has, each once
    vars_to_keep = [var for var in dict.fromkeys(vars_to_keep)
                    if var in available_vars]
    return erf.sel(forcing_variable=vars_to_keep)


def extract_ERF_members(erf, member, erf_id):
    """Keep all members of the forcing, or just one.
    (For GMT members, see extract_GMT_members.)

    erf: the forcing, a DataArray (year, erf_member, forcing_variable), as
        from the load_ERF_* loaders
    member: 'all', or one member's label, e.g. 'ens100' (IGCC) or '7'.
        Datasets with a single ERF timeseries (SMILE_ESM,
        FutureVolc_ESM-rcp45-VolcConst) label it 'single'.
    erf_id: e.g. 'IGCC-2025', for error messages
    """
    # Keep everything
    if member == 'all':
        return erf

    # Find the member, matched as text, because the member comes from the
    # command line but some labels are numbers (SSP_extension)
    available = erf.erf_member.to_numpy().tolist()
    matches = [label for label in available if str(label) == member]

    # Stop if it isn't there
    if not matches:
        raise ValueError(
            f'ERF member {member!r} is not in {erf_id}, which has '
            f'{describe_members(available)}.')

    # Keep that member, for every variable
    return erf.sel(erf_member=matches[:1])


def extend_ERF_to_committed_year(erf, year_committed_to,
                                 year_committed_from=None):
    """Extend the forcing to year_committed_to by holding it constant from
    year_committed_from (or from its last year, if None).

    erf:
        the forcing, a DataArray (year, erf_member, forcing_variable), as
        from load_ERF
    year_committed_to, year_committed_from:
        e.g. 2300 and 2025
    Returns the forcing over the years up to year_committed_to, each year
    after year_committed_from the same as year_committed_from.
    """
    # The forcing up to the year it is held from
    if year_committed_from is not None:
        erf = erf.sel(year=slice(None, year_committed_from))

    # The years to add, each with the last year's forcing
    last_year = int(erf.year.max())
    if year_committed_to > last_year:
        years_to_add = np.arange(last_year + 1, year_committed_to + 1)
        # The last year's forcing, repeated along the new years, with the
        # dimensions in the forcing's order
        extension = erf.sel(year=last_year, drop=True).expand_dims(
            year=years_to_add).transpose(*erf.dims)
        erf = xr.concat([erf, extension], dim='year')
    return erf


def aggregate_missing_forcings(erf):
    """Add any aggregate variable the forcing lacks but whose sub-variables
    are all present, e.g. GHG from co2, ch4, n2o and halogen.

    erf: the forcing, a DataArray (year, erf_member, forcing_variable)
    Returns it with the aggregates added, after its other variables.

    The mapping is ordered so that an aggregate's parts come before it
    (GHG, OHF, Nat, then Ant, then Tot), so an aggregate made here (e.g.
    Ant) can be a part of a later one (Tot). If only some of an aggregate's
    parts are present, it is not made, and a note says which are missing.
    """
    for aggregate, sub_vars in variables.SUB_VAR_MAPPING.items():
        present = erf['forcing_variable'].to_numpy().tolist()
        # Nothing to do if the aggregate is already there
        if aggregate in present:
            continue
        # The parts the forcing has
        present_sub_vars = [var for var in sub_vars if var in present]
        if not present_sub_vars:
            continue
        # Only some parts: say so, and leave the aggregate out
        missing_sub_vars = set(sub_vars) - set(present_sub_vars)
        if missing_sub_vars:
            run_log.note(f'Missing sub-variables for {aggregate}: '
                         f'{missing_sub_vars}. Aggregating only the '
                         f'variables present: {present_sub_vars}.')
            continue
        # Every part: the aggregate is their sum, added one by one, in the
        # mapping's order (as the files' own aggregates are), then put
        # after the other variables
        total = sum(erf.sel(forcing_variable=var, drop=True)
                    for var in present_sub_vars)
        erf = xr.concat(
            [erf, total.expand_dims(forcing_variable=[aggregate])],
            dim='forcing_variable')
    return erf


def load_ERF_IGCC(metadata, include_sub_vars=False):
    """Load IGCC historical ERFs (supplied by Chris Smith), 1000 members,
    labelled 'ens<configuration number>'.

    metadata: e.g. the DatasetMetadata for 'IGCC-2025'
    include_sub_vars: read the _full file, which also holds the
        sub-variables (only some years have one)
    Returns a DataArray (year, erf_member, forcing_variable).
    """
    # Find the file: the _full file also holds the sub-variables
    full = '_full' if include_sub_vars else ''
    file_ERF = find_dataset_file(
        metadata, f'ERF_DAMIP_1000_1750-{metadata.year}{full}.nc')

    # The file's variables, each over (time, ensemble), as one DataArray
    # with the variables along a dimension of their own
    ds = xr.load_dataset(file_ERF)
    # ... without the total, which is made from its parts where needed
    ds = ds.drop_vars('total', errors='ignore')
    erf = ds.to_array(dim='forcing_variable')

    # The engine's names: the years, the members labelled ens<number>, in
    # order of their labels (as text), and the variables' internal names
    erf = erf.rename({'time': 'year', 'ensemble': 'erf_member'})
    erf = erf.assign_coords(
        erf_member=['ens' + str(number)
                    for number in erf.erf_member.to_numpy()])
    erf = erf.sortby('erf_member')
    rename_map = {'wmghg': 'GHG', 'other_ant': 'OHF', 'natural': 'Nat'}
    return erf.assign_coords(forcing_variable=[
        rename_map.get(var, var)
        for var in erf['forcing_variable'].to_numpy()])


def load_ERF_simulated_table(file_ERF):
    """A simulated product's ERF table, read from its CSV file, with the
    engine's variable names.

    file_ERF: the CSV file, a column 'year' and a column per forcing
    Returns a DataFrame indexed by year, one column per variable, without
    the CO2 concentrations (the column 'CO2'), which are not forcing.
    """
    # The file's columns, renamed to the engine's names
    df_ERF = pd.read_csv(file_ERF).rename(columns={
        'ERF_anthro': 'Ant', 'ERF_natural': 'Nat', 'ERF_other_human': 'OHF',
        'ERF_wmghg': 'GHG', 'ERF_CO2': 'co2'})
    # ... indexed by year, without the CO2 concentrations
    return df_ERF.set_index('year').drop(columns='CO2', errors='ignore')


def load_ERF_SMILE_ESM(metadata):
    """Load SMILE_ESM ERFs: the MPI-ESM1.2-LR large ensemble (supplied by
    John Nicklas, for Thorne et al.); a single timeseries per SSP, labelled
    'single'.

    metadata: e.g. the DatasetMetadata for 'SMILE_ESM-SSP245'
    Returns a DataArray (year, erf_member, forcing_variable).
    """
    # Find the file for this SSP (named in lower case), in the model's folder
    file_ERF = find_dataset_file(
        metadata,
        f'MPI-ESM1.2-LR/ERF_ESM1-2-LR_{metadata.variant.lower()}.csv')
    # Its one timeseries, as the member 'single'
    return forcing_from_table(load_ERF_simulated_table(file_ERF))


def load_ERF_FutureVolc_ESM(metadata):
    """Load FutureVolc_ESM ERFs: the NorESM1-M RCP4.5 large ensemble with
    future volcanic eruptions (Bethke et al., 2017; supplied by John Nicklas,
    for Thorne et al.).

    metadata: the DatasetMetadata for 'FutureVolc_ESM-rcp45-Volc' (a natural
        ERF member for each temperature member, labelled '1' to '60') or
        'FutureVolc_ESM-rcp45-VolcConst' (a single timeseries, labelled
        'single')
    Returns a DataArray (year, erf_member, forcing_variable).
    """
    if metadata.variant == 'rcp45-VolcConst':
        # Every forcing in one file, one timeseries: the member 'single'
        file_ERF = find_dataset_file(
            metadata, f'NorESM1-M/ERF_NorESM_{metadata.variant}.csv')
        return forcing_from_table(load_ERF_simulated_table(file_ERF))

    elif metadata.variant == 'rcp45-Volc':
        # The human forcings: one timeseries, shared by every member
        file_ERF_anthro = find_dataset_file(
            metadata, f'NorESM1-M/ERF_anthro_NorESM_{metadata.variant}.csv')
        anthro = forcing_from_table(load_ERF_simulated_table(file_ERF_anthro))

        # The natural forcing: one timeseries per member, the members as the
        # file's columns: (year, erf_member), then with Nat as the one
        # variable: (year, erf_member, forcing_variable)
        file_ERF_natural = find_dataset_file(
            metadata, f'NorESM1-M/ERF_natural_NorESM_{metadata.variant}.csv')
        df_natural = pd.read_csv(file_ERF_natural, index_col=0)
        natural = xr.DataArray(
            df_natural.to_numpy()[:, :, np.newaxis],
            dims=('year', 'erf_member', 'forcing_variable'),
            coords={'year': df_natural.index.to_numpy(),
                    'erf_member': df_natural.columns.to_list(),
                    'forcing_variable': ['Nat']})

        # Every member: the human forcings repeated for each, beside its own
        # natural forcing, the members in order of their labels (as text)
        anthro = anthro.isel(erf_member=0, drop=True).expand_dims(
            erf_member=natural.erf_member.to_numpy()).transpose(
                'year', 'erf_member', 'forcing_variable')
        return xr.concat([anthro, natural], dim='forcing_variable').sortby(
            'erf_member')

    # Any other variant
    else:
        raise ValueError(
            f'Unknown FutureVolc_ESM variant {metadata.variant!r}: use '
            'rcp45-Volc or rcp45-VolcConst.')


def load_ERF_SSP_extension(metadata):
    """Load observed ERFs extended along an SSP (supplied by Chris Smith),
    841 members, labelled by their configuration numbers.

    metadata: e.g. the DatasetMetadata for 'SSP_extension-2022-SSP245'.
        The SSPs share their history to 2022 and diverge from 2023, so the
        data is filed with the 2022 data.
    Returns a DataArray (year, erf_member, forcing_variable).
    """
    # Find the file, whose name holds the FaIR and calibration versions
    file_ERF = find_dataset_file(metadata, 'ssp_forcing_*.nc')
    ds = xr.load_dataset(file_ERF)

    # Select the SSP (named in lower case in the file), stopping if it isn't
    # there
    ssp = metadata.variant.lower()
    available = ds.scenario.to_numpy().tolist()
    if ssp not in available:
        raise ValueError(
            f'{metadata.variant!r} is not in {file_ERF.name}, which has '
            f"{[a.replace('ssp', 'SSP') for a in available]}.")
    ds = ds.sel(scenario=ssp, drop=True)

    # OHF, the human forcings other than the greenhouse gases: aerosol and
    # other
    ds['OHF'] = ds['aerosol'] + ds['other']
    # The variables along a dimension of their own, with the engine's names:
    # the years (mid-year in the file, e.g. 1750.5, so rounded down), the
    # members in order of their configuration numbers, and the variables'
    # internal names
    erf = ds.to_array(dim='forcing_variable').rename(
        {'time': 'year', 'config': 'erf_member'})
    erf = erf.assign_coords(year=erf.year.to_numpy().astype(int))
    erf = erf.sortby('erf_member')
    rename_map = {'ghg': 'GHG', 'natural': 'Nat', 'anthro': 'Ant',
                  'total': 'Tot'}
    return erf.assign_coords(forcing_variable=[
        rename_map.get(var, var)
        for var in erf['forcing_variable'].to_numpy()])


def load_GMT(gmt_id, member, start_pi, end_pi):
    """Load a GMT dataset, keep the member asked for, and remove the
    pre-industrial baseline.

    gmt_id: e.g. 'HadCRUT-2025' or 'SMILE_ESM-SSP245' (see INPUT DATASETS)
    member: 'all', or one member's label, e.g. '7'
    start_pi, end_pi: the pre-industrial period, e.g. 1850, 1900; None for
        no offset

    Returns the observed temperatures, a DataArray (year, gmt_member): year
    first, then members, as every input is (see gwi.py's NAMING), the
    members in the order they appear in the file (the order of the o
    axis).
    """
    metadata = dataset_metadata(gmt_id, 'GMT')

    # Load the whole dataset with its product's loader
    if metadata.product == 'HadCRUT':
        gmt = load_GMT_HadCRUT(metadata)
    elif metadata.product == 'JK_multi':
        gmt = load_GMT_JK_multi(metadata)
    elif metadata.product == 'SMILE_ESM':
        gmt = load_GMT_SMILE_ESM(metadata)
    elif metadata.product == 'FutureVolc_ESM':
        gmt = load_GMT_FutureVolc_ESM(metadata)
    else:
        raise ValueError(f'No GMT loader for {gmt_id!r}.')

    # Keep just the member(s) asked for
    gmt = extract_GMT_members(gmt, member, gmt_id)

    # Remove the pre-industrial baseline
    return preindustrial_baseline(gmt, start_pi, end_pi).rename('gmt')


def temperatures_from_table(df_temp):
    """Temperatures read from a CSV file, as a DataArray.

    df_temp: a DataFrame indexed by year, one column per member, labelled
        as text
    Returns a DataArray (year, gmt_member).
    """
    return xr.DataArray(
        df_temp.to_numpy(), dims=('year', 'gmt_member'),
        coords={'year': df_temp.index.to_numpy(),
                'gmt_member': df_temp.columns.to_list()})


def extract_GMT_members(gmt, member, gmt_id):
    """Keep all members of the temperatures, or just one.
    (For ERF members, see extract_ERF_members.)

    gmt: the temperatures, a DataArray (year, gmt_member), as from the
        load_GMT_* loaders
    member: 'all', or one member's label, e.g. '7'
    gmt_id: e.g. 'HadCRUT-2025', for error messages
    """
    # Keep everything
    if member == 'all':
        return gmt

    # Stop if the member isn't there
    available = gmt.gmt_member.to_numpy().tolist()
    if member not in available:
        raise ValueError(
            f'GMT member {member!r} is not in {gmt_id}, which has '
            f'{describe_members(available)}.')

    # Keep that member
    return gmt.sel(gmt_member=[member])


def load_GMT_HadCRUT(metadata):
    """Load the HadCRUT5 ensemble, members labelled '1' to '200'.

    metadata: e.g. the DatasetMetadata for 'HadCRUT-2025'
    Returns a DataArray (year, gmt_member).
    """
    # Find the ensemble file, whose name holds the HadCRUT version
    temp_ens_Path = find_dataset_file(
        metadata, 'HadCRUT.*.analysis.ensemble_series.global.annual.csv')
    # The file's ensemble members: its columns named 'Realization x',
    # indexed by its 'Time' column (the years) ...
    df_temp_Obs = pd.read_csv(temp_ens_Path).set_index('Time').filter(
        regex='Realization')
    # ... each labelled just x
    df_temp_Obs.columns = [col.split(' ')[-1] for col in df_temp_Obs.columns]

    # Truncate to the final year in the dataset ID (the "vintage" of the
    # dataset, eg 'HadCRUT-2025' means observations through 2025).
    #
    # HadCRUT ships a row for the year currently in progress, which is a mean
    # over an incomplete set of months and is therefore not a meaningful annual
    # value for us (eg HadCRUT 5.1.0.0 in data/observed-2025/ carries a 2026 row
    # whose coverage uncertainty is ~30x that of the complete years).
    return truncate_to_end_year(temperatures_from_table(df_temp_Obs),
                                metadata, temp_ens_Path.name)


def load_GMT_JK_multi(metadata):
    """Load John Kennedy's multi-dataset observational ensemble, members
    labelled '1' to '100'.

    metadata: e.g. the DatasetMetadata for 'JK_multi-2025'
    Returns a DataArray (year, gmt_member).
    """
    # Find the file
    temp_ens_Path = find_dataset_file(metadata, 'sst_pseudo.csv')

    # The file has no header: its first column is the years, and each other
    # column a member, labelled 1 to N in order
    df_temp_Obs = pd.read_csv(temp_ens_Path, header=None, index_col=0)
    df_temp_Obs.columns = [str(realisation + 1)
                           for realisation in range(df_temp_Obs.shape[1])]

    # Same cut-off as HadCRUT (see load_GMT_HadCRUT)
    return truncate_to_end_year(temperatures_from_table(df_temp_Obs),
                                metadata, temp_ens_Path.name)


def load_GMT_SMILE_ESM(metadata):
    """Load SMILE_ESM temperatures: the MPI-ESM1.2-LR large ensemble
    (supplied by John Nicklas, for Thorne et al.), members labelled '0' to
    '49'.

    metadata: e.g. the DatasetMetadata for 'SMILE_ESM-SSP245'
    Returns a DataArray (year, gmt_member).
    """
    # Find the file for this SSP (named in lower case), in the model's folder
    file_temp = find_dataset_file(
        metadata, f'MPI-ESM1.2-LR/ts_ESM1-2-LR_{metadata.variant.lower()}.csv')
    # Its members as columns, indexed by year (the first column)
    return temperatures_from_table(pd.read_csv(file_temp, index_col=0))


def load_GMT_FutureVolc_ESM(metadata):
    """Load FutureVolc_ESM temperatures: the NorESM1-M RCP4.5 large ensemble
    with future volcanic eruptions (Bethke et al., 2017; supplied by John
    Nicklas, for Thorne et al.), members labelled '1' to '60'.

    metadata: e.g. the DatasetMetadata for 'FutureVolc_ESM-rcp45-Volc'
    Returns a DataArray (year, gmt_member).
    """
    # Find the file for this variant, in the model's folder
    file_temp = find_dataset_file(
        metadata, f'NorESM1-M/ts_NorESM_{metadata.variant}.csv')
    # Its members as columns, indexed by year (the first column)
    return temperatures_from_table(pd.read_csv(file_temp, index_col=0))


def truncate_to_end_year(gmt, metadata, file_name):
    """Cut observations off at the year in their dataset ID, e.g. at 2025
    for 'HadCRUT-2025'.

    gmt: the temperatures, a DataArray (year, gmt_member)
    metadata: the DatasetMetadata of an observed dataset
    file_name: its file, for error messages
    """
    # Stop if the file ends before the year in the ID
    end_obs = metadata.year
    years = gmt.year.to_numpy()
    if end_obs not in years:
        # The file is short of what the dataset ID claims: fail loudly
        # rather than silently analysing a shorter record than intended.
        raise ValueError(
            f"Dataset '{metadata.id}' declares observations through "
            f"{end_obs}, but {file_name} only covers "
            f"{years.min()}-{years.max()}.")

    # Drop any later rows
    return gmt.sel(year=slice(None, end_obs))


def preindustrial_baseline(gmt, start_pi, end_pi):
    """Remove the pre-industrial baseline from temperatures.

    gmt: the temperatures, a DataArray (year, gmt_member)
    start_pi, end_pi: e.g. 1850, 1900; None to leave the data unchanged
    """
    # Check if offsets are explicitly bypassed.
    if start_pi is None or end_pi is None:
        return gmt

    # Stop if the pre-industrial period isn't within the data
    years = gmt.year.to_numpy()
    if start_pi not in years or end_pi not in years:
        raise ValueError(
            f'Pre-industrial offsetting period {start_pi}-{end_pi} is not '
            f'within the temperature data, which covers '
            f'{years.min()}-{years.max()}.')

    # The offset is the pre-industrial mean of the ensemble's median (for
    # HadCRUT, its best estimate), and the same offset is taken from every
    # member, which keeps the ensemble's spread (it would be wrong to take
    # each member's own pre-industrial mean from it)
    offset = gmt.median('gmt_member').sel(
        year=slice(start_pi, end_pi)).mean('year')
    return gmt - offset


def load_PiC(picontrol_id, years, start_pi, end_pi):
    """Load piControl temperatures, in segments as long as the GMT record.

    picontrol_id: e.g. 'CMIP6' (the only one for now)
    years: the years of the GMT data, e.g. 1850 to 2025. Each segment is
        this long, and is indexed by these years.
    start_pi, end_pi: e.g. 1850, 1900, both included; None for no
        pre-industrial offset (each segment is then offset by its own mean)

    Returns the segments, a DataArray (year, pic_member): year first, then
    members (each member a segment), as every input is (see gwi.py's
    NAMING), the segments in the order the loader makes them (the order of
    the c axis).
    """
    # Load with the dataset's loader
    if picontrol_id == 'CMIP6':
        return load_PiC_CMIP6(years, start_pi, end_pi).rename('pic')
    else:
        raise ValueError(
            f'No piControl loader for {picontrol_id!r}.')


def load_PiC_CMIP6(years, start_pi, end_pi):
    """Slice the CMIP6 piControl simulations (.MAG files) into segments as
    long as `years`, overlapping by half: a DataArray (year, pic_member),
    each member a segment.

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

        # The simulation's global mean temperature, every year it has: read
        # with pymagicc, which gives each region's variable over time
        simulation = pymagicc.io.MAGICCData(file).to_xarray()
        temp = simulation[var].sel(region='World').dropna('time').to_numpy()

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

    # The segments, one per column, indexed by the years given: (t, c)
    return xr.DataArray(
        np.stack(list(dict_temp.values()), axis=1),
        dims=('year', 'pic_member'),
        coords={'year': years, 'pic_member': list(dict_temp.keys())})


def filter_PiControl(pic, timeframes):
    """Remove piControl segments whose variability is implausible next to
    the observations.

    pic: the segments, a DataArray (year, pic_member), as from load_PiC
    timeframes: e.g. [1, 3, 30] (not used at present)
    Returns the segments kept, a DataArray (year, pic_member), in their
    order.
    """
    kept_segments = []
    for segment in pic.pic_member.to_numpy():
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
        temp = pic.sel(pic_member=segment).to_numpy()
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
            kept_segments.append(segment)

    # The segments approved, in their order
    return pic.sel(pic_member=kept_segments)
