import os
import multiprocessing as mp
import numpy as np
import pandas as pd
# import functools
import xarray as xr
import glob
from pathlib import Path
import pymagicc
import sys


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def load_ERF(scenario, regress_vars, ensemble_members):
    if 'observed-20' in scenario:
        df_ERF = load_ERF_CMIP6(scenario, regress_vars)
    elif 'observed-SSP' in scenario:
        df_ERF = load_ERF_SSP(scenario, regress_vars)
    elif 'SMILE_ESM' in scenario:
        df_ERF = load_ERF_SMILE(scenario, regress_vars)
    elif 'NorESM' in scenario:
        df_ERF = load_ERF_NorESM(scenario, regress_vars)
    else:
        raise ValueError('Invalid scenario for ERF data.')

    # Select vars:
    df_ERF = df_ERF.loc[:, (regress_vars, slice(None))]

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
        print(f'Invalid ensemble members {ensemble_members} for ensemble:'
              + f'{df_ERF.columns.get_level_values("ensemble").unique()}')
        raise ValueError('Invalid ensemble member {ensemble_member} for data.')

    return df_ERF.loc[:, (slice(None), ens_mems)]


def load_ERF_CMIP6(scenario, regress_vars=['GHG', 'OHF', 'Nat']):
    """Load the ERFs from Chris."""
    # ERF location
    here = Path(__file__).parent
    end = scenario.split('-')[-1]
    file_ERF = here / f'../data/{scenario}/ERF/Chris/ERF_DAMIP_1000_1750-{end}.nc'
    # import ERF_file to xarray dataset and convert to pandas dataframe
    df_ERF = xr.open_dataset(file_ERF).to_dataframe()
    # assign the columns the name 'variable'
    df_ERF.columns.names = ['variable']
    # remove the column called 'total' from df_ERF
    df_ERF = df_ERF.drop(columns='total')
    # rename the variable columns
    df_ERF = df_ERF.rename(columns={'wmghg': 'GHG',
                                    'other_ant': 'OHF',
                                    'natural': 'Nat'})
    # move the multi-index 'ensemble' level to a column,
    # and then set the 'ensemble' column to second column level
    df_ERF = df_ERF.reset_index(level='ensemble')
    df_ERF['ensemble'] = 'ens' + df_ERF['ensemble'].astype(str)
    df_ERF = df_ERF.pivot(columns='ensemble')

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

    # Check whether the regress_vars is the same as the columns of df_ERF
    check_vars = sorted(regress_vars) == sorted(forc_var_names)

    # If regress_vars and forc_vars are the same, no need to do anything.
    # If they are not the same, aggregate into requried variables:
    if check_vars:
        pass

    elif not check_vars and regress_vars == ['Tot']:
        # If 'Tot' is the only variable to regress, combine all variables:
        df_ERF_Tot = df_ERF.loc[:, ('GHG', slice(None))
                                ].copy().rename(columns={'GHG': 'Tot'})
        # Group df_ERF by ensemble name, and sum across variable names
        df_ERF_Tot[:] = df_ERF[['GHG', 'OHF', 'Nat']
                               ].groupby(level='ensemble', axis=1
                                         ).sum()
        df_ERF = df_ERF_Tot

    elif not check_vars and regress_vars == ['Ant', 'Nat']:
        # If 'Ant' and 'Nat' are the only variables to regress, combine
        # 'GHG' and 'OHF' into 'Ant':
        df_ERF_Ant = df_ERF.loc[:, ('GHG', slice(None))
                                ].copy().rename(columns={'GHG': 'Ant'})
        # Group df_ERF by ensemble name, and sum across variable names
        df_ERF_Ant[:] = df_ERF[['GHG', 'OHF']
                               ].groupby(level='ensemble', axis=1
                                         ).sum()
        # Concatenate 'Ant' with 'Nat':
        df_ERF_Nat = df_ERF.loc[:, ('Nat', slice(None))]
        df_ERF = pd.concat([df_ERF_Ant, df_ERF_Nat], axis=1)

    else:
        raise ValueError('Invalid combination of variables for regression.')

    return df_ERF


def load_ERF_SMILE(scenario, regress_vars=['GHG', 'OHF', 'Nat']):
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
                                           'ERF_wmghg': 'GHG'
                                           }
                                  ).set_index('Year')

    # Add a second level to the column names ,and set the name of the second
    # level to 'ensemble'. Make the value of this 'single' for all of the
    # columns. This keeps the data structure the same as the multi-ensemble
    # data.

    if sorted(regress_vars) == sorted(['Ant', 'Nat']):
        # Remove all columns not named 'Ant' or 'Nat':
        df_ERF = df_ERF[['Ant', 'Nat']]
    elif sorted(regress_vars) == sorted(['GHG', 'OHF', 'Nat']):
        df_ERF = df_ERF[['GHG', 'OHF', 'Nat']]

    df_ERF.columns = pd.MultiIndex.from_tuples(
        [(col, 'single') for col in df_ERF.columns],
        names=['variable', 'ensemble'])

    forc_var_names = sorted(df_ERF.columns.get_level_values(
        'variable').unique().to_list())

    # Check whether the regress_vars is the same as the columns of df_ERF
    check_vars = sorted(regress_vars) == sorted(forc_var_names)

    # If regress_vars and forc_vars are the same, no need to do anything.
    # If they are not the same, aggregate into requried variables:
    if check_vars:
        pass

    elif not check_vars and regress_vars == ['Tot']:
        # If 'Tot' is the only variable to regress, combine all variables:
        df_ERF_Tot = df_ERF.loc[:, ('Ant', slice(None))
                                ].copy().rename(columns={'Ant': 'Tot'})
        # Group df_ERF by ensemble name, and sum across variable names
        df_ERF_Tot[:] = df_ERF[['Ant', 'Nat']
                               ].groupby(level='ensemble', axis=1
                                         ).sum()
        df_ERF = df_ERF_Tot

    else:
        raise ValueError('Invalid combination of variables for this scenario.')

    return df_ERF


def load_ERF_NorESM(scenario, regress_vars=['GHG', 'OHF', 'Nat']):
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
                                               'ERF_wmghg': 'GHG'
                                               }
                                      ).set_index('Year')
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
                                               'ERF_wmghg': 'GHG'
                                               }
                                      ).set_index('Year')
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
            [df_ERF_anthro['GHG', str(ens)].equals(df_ERF_anthro['GHG', '0'])
             for ens in ens_num])
        if not check_ens:
            raise ValueError(
                'Ensemble values are not the same for all variables.')

        # Combine the two dataframes
        df_ERF = pd.concat([df_ERF_anthro, df_ERF_natural], axis=1)

    if sorted(regress_vars) == sorted(['Ant', 'Nat']):
        # Remove all columns not named 'Ant' or 'Nat':
        df_ERF = df_ERF.loc[:, (['Ant', 'Nat'], slice(None))]
    elif sorted(regress_vars) == sorted(['GHG', 'OHF', 'Nat']):
        df_ERF = df_ERF.loc[:, (['GHG', 'OHF', 'Nat'], slice(None))]

    # Check whether the regress_vars is the same as the columns of df_ERF
    forc_var_names = sorted(df_ERF.columns.get_level_values(
        'variable').unique().to_list())
    check_vars = sorted(regress_vars) == sorted(forc_var_names)

    # If regress_vars and forc_vars are the same, no need to do anything.
    # If they are not the same, aggregate into requried variables:
    if check_vars:
        pass

    elif not check_vars and regress_vars == ['Tot']:
        # If 'Tot' is the only variable to regress, combine all variables:
        df_ERF_Tot = df_ERF.loc[:, ('GHG', slice(None))
                                ].copy().rename(columns={'GHG': 'Tot'})
        # Group df_ERF by ensemble name, and sum across variable names
        df_ERF_Tot[:] = df_ERF[['GHG', 'OHF', 'Nat']
                               ].groupby(level='ensemble', axis=1
                                         ).sum()
        df_ERF = df_ERF_Tot

    else:
        raise ValueError('Invalid combination of variables for regression.')

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

    forc_var_names = sorted(df_ERF.columns.get_level_values(
        'variable').unique().to_list())

    # Check whether the regress_vars are within the columns of df_ERF
    check_vars = set(regress_vars).issubset(forc_var_names)
    if check_vars:
        df_ERF = df_ERF.loc[:, (regress_vars, slice(None))]
    else:
        raise ValueError('Invalid combination of variables for regression.')

    # Check that the ensemble names are the same for all variables.
    dict_ensemble_names = {}
    for var in regress_vars:
        forc_subset = df_ERF.loc[:, (var, slice(None))]
        forc_ens_names = sorted(
            list(forc_subset.columns.get_level_values("ensemble").unique()))
        dict_ensemble_names[var] = forc_ens_names

    check_ens = all(
        [dict_ensemble_names[var] == dict_ensemble_names[regress_vars[0]]
         for var in regress_vars]
         )

    if not check_ens:
        raise ValueError('Ensemble names are not the same for all variables.')

    return df_ERF


def load_Temp(scenario, ensemble_members, start_pi, end_pi):
    if 'observed-20' in scenario:
        df_temp = load_HadCRUT(scenario, start_pi, end_pi)
    elif 'observed-SSP' in scenario:
        df_temp = load_HadCRUT('observed-2024', start_pi, end_pi)
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
        raise ValueError('Invalid ensemble member {ensemble_member} for data.')

    # Remove pre-industrial baseline from temperature data
    df_temp = preindustrial_baseline(df_temp, start_pi, end_pi)

    return df_temp


def load_HadCRUT(scenario, start_pi, end_pi):
    """Load HadCRUT5 observations and remove PI baseline."""
    here = Path(__file__).parent
    temp_ens_Path = (
        f'../data/{scenario}/Temp/HadCRUT/' +
        'HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.csv')
    temp_ens_Path = here / temp_ens_Path
    # read temp_Path into pandas dataframe, rename column 'Time' to 'Year'
    # and set the index to 'Year', keeping only columns with 'Realization' in
    # the column name, since these are the ensembles
    df_temp_Obs = pd.read_csv(temp_ens_Path,
                              ).rename(columns={'Time': 'Year'}
                                       ).set_index('Year'
                                                   ).filter(regex='Realization'
                                                            )

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
    df_temp_Obs = load_HadCRUT(start_pi, end_pi, start_yr, end_yr)
    temp_Yrs = df_temp_Obs.index.values
    arr_temp_Obs = df_temp_Obs.values
    # Apply the function defs.rate_calc to each column of this dataframe

    dfs_rates = []
    for year in np.arange(1950, end_yr+1):
        print(year, end='\r')
        recent_years = ((year-9 <= temp_Yrs) * (temp_Yrs <= year))
        ten_slice = arr_temp_Obs[recent_years, :]

        with mp.Pool(os.cpu_count()) as p:
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
            with mp.Pool(os.cpu_count()) as p:
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


def check_steps(all_reg_ranges):
    """Check that the years are in steps of 1."""
    end_yrs = sorted([
        int(regressed_years.split('-')[1])
        for regressed_years in all_reg_ranges
    ])
    all_year_steps = all(np.diff(end_yrs) == 1)

    out_dict = {
        'check_bool': all_year_steps,
        'range': f'{min(all_reg_ranges)} to {max(all_reg_ranges)}' ,
    }

    return out_dict


def check_headlines(hy):
    """Check that the headline years are in the correct format."""
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
