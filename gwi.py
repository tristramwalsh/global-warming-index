"""Script to generate global warming index."""

import os
import sys

import datetime as dt
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats as ss

import src.graphing as gr
import src.definitions as defs

import models.AR5_IR as AR5_IR
import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair


###############################################################################
# DEFINE FUNCTIONS ############################################################
###############################################################################
def GWI_faster(
        model_choice, inc_reg_const, inc_pi_offset,
        df_forc, df_params, df_temp_PiC, df_temp_Obs,
        start_trunc, end_trunc, start_pi, end_pi,
        start_regress, end_regress):
    """Calculate the global warming index (GWI)."""
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
    # NOTE: we get passed a list of variables.

    # Prepare results #########################################################
    # Total sub-ensemble size: multiple number of ensemble members for each of:
    # HadCRUT sub-ensemble * piControl sub-ensemble * ERF sub-ensemble
    # Specify only 1 for the number of FaIR paramaterisations, as we are
    # parallelising across the parameterisations, and therefore only have one
    # for each instance of the function call.
    n = (len(ens_list_Obs) * len(ens_list_PiC) * len(ens_list_ERF) * 1)

    # Include residuals and totals for sum total and anthropogenic warming in
    # the same array as attributed results. +1 each for Ant, TOT, Res,
    # InternalVariability, ObservedTemperatures
    # NOTE: the order in dimension is:
    # vars_list = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
    # TODO: rewrite the above.
    # Add the list of variables that we diagnose after the multi-variable 
    # regression. If we aren't directly regressing 'Ant', then we need to
    # include it in the extra_vars list. If we are not, then we need to include
    # 'Ant' in the variables list.
    extra_vars = defs.extra_vars(var_list_ERF)

    # Create empty output array, with dimensions: (years, variables, samples)
    temp_Att_Results = np.empty(
      (end_trunc - start_trunc + 1,  # number of years (after truncation)
       len(var_list_ERF) + len(extra_vars),  # variables dimension
       n),  # samples
      dtype=np.float32  # make the array smaller in memory
      )
    # coef_Reg_Results = np.zeros((len(variables) + int(inc_reg_const), n))

    # slice df_temp_obs dataframe to include years *inclusively* between
    # start_trunc and end_trunc
    df_temp_Obs_trunc = df_temp_Obs.loc[start_trunc:end_trunc]

    # slice df_temp_PiC dataframe to include years *inclusively* between
    # start_trunc and end_trunc
    PiC_Yrs = df_temp_PiC.index.to_numpy()
    df_temp_PiC_trunc = df_temp_PiC.loc[start_trunc:end_trunc]
    # Get full forcings, and truncated years timeseries
    temp_Yrs = df_temp_Obs.index.to_numpy()  # Full temperature years

    trunc_Yrs = df_temp_Obs_trunc.index.to_numpy()  # Truncation years
    forc_Yrs = df_forc.index.to_numpy()  # Full forcing years

    # Prepare FaIR parameters for this particular model.
    params_FaIR = df_params[model_choice]
    params_FaIR.columns = pd.MultiIndex.from_product(
        [[model_choice], params_FaIR.columns])

    # Prepare results array for temperatures. Note that temp_Mod naming refers
    # to the fact that these temperatures are outputs from the model.
    temp_Mod_array_all_years = np.empty(shape=(forc_Yrs.shape[0],
                                               len(var_list_ERF),
                                               len(ens_list_ERF)))

    # Calculate temperatures from forcings for all ensembles at once,
    # leveraging FaIR's vectorisation
    for var in var_list_ERF:
        # Select forcings for the specific variable. This selects all ensemble
        # members available from the random subsample.
        # TODO: check the date range when specifying separate truncation years
        # in future versions.
        forc_var_All = df_forc.loc[:, (var, slice(None))]

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
            forc_FaIR[ens] = forc_var_All[(var, ens)].to_numpy()

        # Run FaIR. Convert output to numpy array for later regression.
        temp_All = fair.run_FaIR(emissions_in=emis_FAIR,
                                 forcing_in=forc_FaIR,
                                 thermal_parameters=params_FaIR,
                                 show_run_info=False)['T'].to_numpy()
        temp_Mod_array_all_years[:, var_list_ERF.index(var), :] = temp_All

    i = 0
    for ens in range(len(ens_list_ERF)):
        # Select just the one ensemble member:
        temp_Mod_array_all_years_ens = temp_Mod_array_all_years[:, :, ens]

        # Remove pre-industrial offset before regression if specified.
        if inc_pi_offset:
            ens_offset = temp_Mod_array_all_years_ens[(forc_Yrs >= start_pi) &
                                                      (forc_Yrs <= end_pi), :
                                                      ].mean(axis=0)
            temp_Mod_array_all_years_ens = (temp_Mod_array_all_years_ens -
                                            ens_offset)

        # Add a constant offset term to the regression if specified.
        if inc_reg_const:
            temp_Mod_array_all_years_ens = np.append(
                temp_Mod_array_all_years_ens,
                np.ones((temp_Mod_array_all_years_ens.shape[0], 1)),
                axis=1)
        n_reg_vars = temp_Mod_array_all_years_ens.shape[1]

        # Prepare arrays to store regression coefficients.
        # Dimensions correspond to:
        # (1st dimension) number of variables and
        # (2nd dimension) the number of samples available for each of
        # reference Obs temps piControl internal variability temps. This is
        # because are regressing all sub-ensemble members for Obs and PiC
        # separately, and then adding the regression coefficients together
        # afterwards.
        coef_Obs_Results = np.empty((n_reg_vars, len(ens_list_Obs)))
        coef_PiC_Results = np.empty((n_reg_vars, len(ens_list_PiC)))

        # Regress against observations
        c_i = 0
        # Iterate over the samples of the observed temperature ensemble
        for temp_Obs_Ens in ens_list_Obs:
            # Select the required reference temperature ensemble timeseries
            temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
            # Select only the year range for temp_Mod_array_all_years_ens and
            # temp_Obs_i that corresponds to the regression range.
            temp_Mod_regress_yrs = temp_Mod_array_all_years_ens[
                                        (forc_Yrs >= start_regress) &
                                        (forc_Yrs <= end_regress)]
            temp_Obs_i_regress = temp_Obs_i[(temp_Yrs >= start_regress) &
                                            (temp_Yrs <= end_regress)]
            coef_Obs_i = np.linalg.lstsq(
                temp_Mod_regress_yrs, temp_Obs_i_regress, rcond=None)[0]
            coef_Obs_Results[:, c_i] = coef_Obs_i
            c_i += 1

        # Regress against piControl
        c_j = 0
        # Iterate over the samples of the piControl temperature ensemble
        for temp_PiC_Ens in ens_list_PiC:
            # Select the required piControl temperature ensemble timeseries
            temp_PiC_j = df_temp_PiC[temp_PiC_Ens].to_numpy()
            # Select only the year range for temp_Mod_array_all_years_ens and
            # temp_PiC_j that corresponds to the regression range.
            temp_Mod_regress_yrs = temp_Mod_array_all_years_ens[
                                        (forc_Yrs >= start_regress) &
                                        (forc_Yrs <= end_regress)]
            temp_PiC_j_regress = temp_PiC_j[(PiC_Yrs >= start_regress) &
                                            (PiC_Yrs <= end_regress)]
            coef_PiC_j = np.linalg.lstsq(
                temp_Mod_regress_yrs, temp_PiC_j_regress, rcond=None)[0]
            coef_PiC_Results[:, c_j] = coef_PiC_j
            c_j += 1

        # Combine regression coefficients from observations and piControl

        # Cut the full-length temperatures (that were calculated from ERF) down
        # to the truncation length.
        yr_mask = ((forc_Yrs >= start_trunc) & (forc_Yrs <= end_trunc))
        temp_Mod_trunc_yrs_ens = temp_Mod_array_all_years_ens[yr_mask, :]

        # Now combine the coeffieicnts with the truncated model outputs:
        for c_k in range(coef_Obs_Results.shape[1]):
            for c_l in range(coef_PiC_Results.shape[1]):
                # Regression coefficients
                coef_Reg = (coef_Obs_Results[:, c_k] +
                            coef_PiC_Results[:, c_l])

                # Attributed warming for each component
                temp_Att = temp_Mod_trunc_yrs_ens * coef_Reg

                # Extract T_Obs and T_PiC data for this c_i, c_j combo.
                temp_Obs_kl = df_temp_Obs_trunc[ens_list_Obs[c_k]
                                                ].to_numpy()
                # temp_PiC_kl = df_temp_PiC[ens_list_PiC[c_l]
                #                           ].to_numpy()

                # Save outputs from the calculation:
                # Regression coefficients
                # coef_Reg_Results[:, i] = coef_Reg

                # Attributed warming for each component.
                # NOTE: the constant term in the regression is not included in
                # this array, to save memory space. This explains the slicing
                # on the next two lines:
                # (the -1*inc_reg_const in temp_Att_Results,
                # and the :-1 in temp_Att).
                temp_Att_Results[:, :(n_reg_vars-(1*inc_reg_const)), i] = \
                    temp_Att[:, :-1]

                # # Actual piControl IV sample that used for this c_k, c_l
                # temp_Att_Results[:, -2, i] = temp_PiC_kl
                # # The temp_Obs (dependent var) for this c_k, c_l
                # temp_Att_Results[:, -1, i] = temp_Obs_kl

                # TOTAL WARMING
                # NOTE: no conditional required: 'Tot' is in position -2
                # regardless of the number of regression variables:
                # e.g. [Ant, Nat, OHF, Tot, Res] for 3-way
                # e.g. [Tot, Res] for 1-way
                # Even in a 1-way regression, you still want to add the
                # constant regression offset in the Tot warming output, hence
                # why we always sum over the temp_Att variables dimension.
                temp_Tot = temp_Att.sum(axis=1)
                temp_Att_Results[:, -2, i] = temp_Tot

                # RESIDUAL WARMING
                if 'Res' in extra_vars:
                    temp_Att_Results[:, -1, i] = (temp_Obs_kl - temp_Tot)

                # ANTROPOGENIC WARMING
                if 'Ant' in extra_vars:
                    temp_Ant = (temp_Att[:, var_list_ERF.index('GHG')] +
                                temp_Att[:, var_list_ERF.index('OHF')])
                    temp_Att_Results[:, -3, i] = temp_Ant

                # Visual display of progress through calculation ##############
                # Turned off for now to avoid cluttering the slurm output,
                # which seems not to be able to do carriage return.
                # if i % 1000 == 0:
                #     percentage = int((i+1)/n*100)
                #     loading_bar = (percentage // 5*'.' +
                #                    (20 - percentage // 5)*' ')
                #     print(f'calculating {loading_bar} {percentage}%',
                #           end='\r')
                i += 1
                # #############################################################
    return temp_Att_Results


# def GWI(
#         variables, inc_reg_const,
#         df_forc, df_params, df_temp_PiC, df_temp_Obs,
#         start_trunc, end_trunc):
#     """Calculate the global warming index (GWI)."""
#     # - BRING start_pi AND end_pi INSIDE THE FUNCTION

#     # Prepare results #########################################################
#     n = (df_temp_Obs.shape[1] * df_temp_PiC.shape[1] *
#          len(forc_subset.columns.get_level_values("ensemble").unique()) *
#          len(df_params.columns.levels[0]))
#     # Include residuals and totals for sum total and anthropogenic warming in
#     # the same array as attributed results. +1 each for Ant, Tot, Res
#     # NOTE: the order in dimension is:
#     # vars_list = ['GHG', 'Nat', 'OHF', 'Ant', 'Tot', 'Res']
#     temp_Att_Results = np.zeros(
#       (end_trunc - start_trunc + 1,  # years
#        len(variables) + 3,  # variables
#        n),  # samples
#       dtype=np.float32  # make the array smaller in memory
#       )
#     coef_Reg_Results = np.zeros((len(variables) + int(inc_reg_const), n))

#     forc_Yrs = df_forc.index.to_numpy()
#     # slice df_temp_obs dataframe to include years between start_trunc and end_trunc
#     df_temp_Obs = df_temp_Obs.loc[start_trunc:end_trunc]
#     # slice df_temp_PiC dataframe to include years between start_trunc and end_trunc
#     df_temp_PiC = df_temp_PiC.loc[start_trunc:end_trunc]

#     # Loop over all sampling combinations #####################################
#     i = 0
#     for CMIP6_model in df_params.columns.levels[0].unique():
#         # Select the specific model's parameters
#         params_FaIR = df_params[CMIP6_model]
#         # Since the above line seems to get rid of the top column level (the
#         # model name), and therefore reduce the level to 1, we need to re-add
#         # the level=0 column name (the model name) in order for this to be
#         # compatible with the required FaIR format...
#         params_FaIR.columns = pd.MultiIndex.from_product(
#             [[CMIP6_model], params_FaIR.columns])

#         for forc_Ens in df_forc.columns.get_level_values("ensemble").unique():
#             # Select forcings for the specific ensemble member
#             forc_Ens_All = df_forc.loc[:end_trunc, (slice(None), forc_Ens)]

#             # FaIR won't run without emissions or concentrations, so specify
#             # no zero emissions for input.
#             emis_FAIR = fair.return_empty_emissions(
#                 df_to_copy=False,
#                 start_year=min(forc_Yrs), end_year=end_trunc, timestep=1,
#                 scen_names=variables)
#             # Prepare a FaIR-compatible forcing dataframe
#             forc_FaIR = fair.return_empty_forcing(
#                 df_to_copy=False,
#                 start_year=min(forc_Yrs), end_year=end_trunc, timestep=1,
#                 scen_names=variables)
#             for var in variables:
#                 forc_FaIR[var] = forc_Ens_All[var].to_numpy()

#             # Run FaIR
#             # Convert back into numpy array for comapbililty with the pre-FaIR
#             # code below.
#             temp_All = fair.run_FaIR(emissions_in=emis_FAIR,
#                                      forcing_in=forc_FaIR,
#                                      thermal_parameters=params_FaIR,
#                                      show_run_info=False)['T'].to_numpy()

#             # Remove pre-industrial offset before regression
#             if inc_pi_offset:
#                 _ofst = temp_All[(forc_Yrs >= start_pi) &
#                                  (forc_Yrs <= end_pi), :
#                                  ].mean(axis=0)
#             else:
#                 _ofst = 0
#             temp_Mod = temp_All[(forc_Yrs >= start_trunc) &
#                                 (forc_Yrs <= end_trunc)] - _ofst

#             # Decide whether to include a Constant offset term in regression
#             if inc_reg_const:
#                 temp_Mod = np.append(temp_Mod,
#                                      np.ones((temp_Mod.shape[0], 1)),
#                                      axis=1)
#             n_reg_vars = temp_Mod.shape[1]

#             coef_Obs_Results = np.empty((temp_Mod.shape[1],
#                                          df_temp_Obs.shape[1]))
#             coef_PiC_Results = np.empty((temp_Mod.shape[1],
#                                          df_temp_PiC.shape[1]))

#             # Regree against observations
#             c_i = 0
#             for temp_Obs_Ens in df_temp_Obs.columns:
#                 temp_Obs_i = df_temp_Obs[temp_Obs_Ens].to_numpy()
#                 coef_Obs_i = np.linalg.lstsq(temp_Mod, temp_Obs_i,
#                                              rcond=None)[0]
#                 coef_Obs_Results[:, c_i] = coef_Obs_i
#                 c_i += 1

#             # Regress against piControl
#             c_j = 0
#             for temp_PiC_Ens in df_temp_PiC.columns:
#                 temp_PiC_j = df_temp_PiC[temp_PiC_Ens].to_numpy()
#                 coef_PiC_j = np.linalg.lstsq(temp_Mod, temp_PiC_j,
#                                              rcond=None)[0]
#                 coef_PiC_Results[:, c_j] = coef_PiC_j
#                 c_j += 1

#             for c_k in range(coef_Obs_Results.shape[1]):
#                 for c_l in range(coef_PiC_Results.shape[1]):
#                     # Regression coefficients
#                     coef_Reg = (coef_Obs_Results[:, c_k] +
#                                 coef_PiC_Results[:, c_l])
#                     # Attributed warming for each component
#                     temp_Att = temp_Mod * coef_Reg

#                     # Extract T_Obs and T_PiC data for this c_i, c_j combo.
#                     temp_Obs_kl = df_temp_Obs[df_temp_Obs.columns[c_k]
#                                               ].to_numpy()
#                     # temp_PiC_kl = df_temp_PiC[df_temp_PiC.columns[c_l]
#                     #                           ].to_numpy()

#                     # Save outputs from the calculation:
#                     # Regression coefficients
#                     coef_Reg_Results[:, i] = coef_Reg

#                     # Attributed warming for each component
#                     temp_Att_Results[:, :(n_reg_vars-(1*inc_reg_const)), i] = \
#                         temp_Att[:, :-1]

#                     # # Actual piControl IV sample that used for this c_k, c_l
#                     # temp_Att_Results[:, -2, i] = temp_PiC_kl
#                     # # The temp_Obs (dependent var) for this c_k, c_l
#                     # temp_Att_Results[:, -1, i] = temp_Obs_kl
#                     # TOTAL
#                     temp_Tot = temp_Att.sum(axis=1)
#                     temp_Att_Results[:, -2, i] = temp_Tot
#                     # RESIDUAL
#                     temp_Att_Results[:, -1, i] = (temp_Obs_kl - temp_Tot)
#                     # ANTROPOGENIC
#                     temp_Ant = (temp_Att[:, variables.index('GHG')] +
#                                 temp_Att[:, variables.index('OHF')])
#                     temp_Att_Results[:, -3, i] = temp_Ant

#                     # Visual display of pregress through calculation
#                     if i % 1000 == 0:
#                         percentage = int((i+1)/n*100)
#                         loading_bar = (percentage // 5*'.' +
#                                        (20 - percentage // 5)*' ')
#                         print(f'calculating {loading_bar} {percentage}%',
#                               end='\r')
#                     i += 1

#     print(f"calculating {20*'.'} {100}%", end='\r')
#     return temp_Att_Results, coef_Reg_Results


###############################################################################
# MAIN CODE BODY ##############################################################
###############################################################################

if __name__ == "__main__":

    # Request whether to include pre-industrial offset and constant term in
    # regression. Following discussion with Myles, fix the regression options
    # as the following (there is no user selection any more):
    inc_pi_offset = True
    inc_reg_const = True

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

    # Determine whether to plot the filtered piControl ensemble members.
    # Defaults to 'no', as it useful only for diagnosing the method, and
    # doesn't need re-plotting every time.
    plot_piControl = False

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

    if len(sys.argv) > 1:
        # Separate out the names and values for each argv, and place them in
        # a dictionary for later use.
        argvs = sys.argv
        argv_dict = {argv.split('=')[0]: argv.split('=')[1]
                     for argv in argvs
                     if '=' in argv}
    else:
        # Adding this simplifies logic later on, as we can always assume that
        # the dictionary exists, and just check for the presence.
        argv_dict = {}

    # Define pre-industrial period for temperature offset.
    # The average of this period is used as the offset, as standard in IPCC.
    if '--preindustrial-era' in argv_dict:
        start_pi = int(argv_dict['--preindustrial-era'].split('-')[0])
        end_pi = int(argv_dict['--preindustrial-era'].split('-')[1])
    else:
        start_pi, end_pi = 1850, 1900  # As in IPCC AR6 Ch.3

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

    # Determine whether to calculate rates (computaitonally expensive).
    if '--include-rate' in argv_dict:
        rate_toggle = argv_dict['--include-rate']
        rate_toggle = True if rate_toggle == 'y' else False
    else:
        # If no command line argument passed, use interactivity as a backup.
        rate_toggle = input('Calculate rates? (y/n): ')
        rate_toggle = True if rate_toggle == 'y' else False

    # Determine whether to include headline calculations in the output.
    # If no, then only the timeseries are calculated, and not the SR1.5 and
    # AR6 definitions that are used for the IGCC assessments.
    if '--include-headlines' in argv_dict:
        headline_toggle = argv_dict['--include-headlines']
        headline_toggle = True if headline_toggle == 'y' else False
    else:
        headline_toggle = input('Include headlines? (y/n): ')
        headline_toggle = True if headline_toggle == 'y' else False

    # Specify regression variables:
    if '--regress-variables' in argv_dict:
        regress_vars = sorted(argv_dict['--regress-variables'].split(','))
    else:
        regress_vars = sorted(['GHG', 'Nat', 'OHF'])

    # Specify the scenario
    if '--scenario' in argv_dict:
        scenario = argv_dict['--scenario']
    else:
        # available scenarios:
        available = os.listdir('data/')
        scenario = input(f'Scenario (e.g. {available}: ')

    # Specify the subset of the scenario's ensemble.
    # By default, use the entire ensemble of forcings and temperatures.
    # Optionally specify a specific member, e.g. as required for NorESM Volc
    # scenarios for Thorne et al., 2025
    if '--specify-ensemble-member' in argv_dict:
        ensemble_members = argv_dict['--specify-ensemble-member']
    else:
        ensemble_members = 'all'

    # Create directory structure based on the input parameters.
    output_path = (
        f'SCENARIO--{scenario}/' +
        f'ENSEMBLE-MEMBER--{ensemble_members}/' +
        f'VARIABLES--{"-".join(regress_vars)}/' +
        f'REGRESSED-YEARS--{start_regress}-{end_regress}/'
    )

    # Create a folder to store the plots
    results_folder = 'results/iterations/'
    if not os.path.exists(f'{results_folder}{output_path}'):
        os.makedirs(f'{results_folder}{output_path}')
    plot_folder = 'plots/iterations/'
    if not os.path.exists(f'{plot_folder}{output_path}'):
        os.makedirs(f'{plot_folder}{output_path}')

    ###########################################################################
    # READ IN THE DATA ########################################################
    ###########################################################################

    # Effective Radiative Forcing
    df_forc = defs.load_ERF(scenario, regress_vars, ensemble_members)
    forc_var_names = sorted(
        df_forc.columns.get_level_values('variable').unique())
    # Obtain the ERF_start and ERF_end from the dataframe.
    forc_Yrs = np.array(df_forc.index)
    forc_Yrs_min = forc_Yrs.min()
    forc_Yrs_max = forc_Yrs.max()

    print(df_forc)

    # Check that the truncation years are within the ERF data range.
    # TODO: write down how and when the truncation happens (i.e. after the
    # regression).
    if start_trunc < forc_Yrs_min:
        start_trunc = forc_Yrs_min
        print('Truncation start year is before ERF data range, '
              f'setting start year to ERF data minimum: {start_trunc}')
    if end_trunc > forc_Yrs_max:
        end_trunc = forc_Yrs_max
        print('Truncation end year is after ERF data range, '
              f'setting end year to ERF data maximum: {end_trunc}')

    trunc_Yrs = np.arange(start_trunc, end_trunc+1)

    # TEMPERATURE
    df_temp_Obs = defs.load_Temp(scenario, ensemble_members, start_pi, end_pi)
    n_yrs = df_temp_Obs.shape[0]
    # Obtain the maximum regressable years from the dataframe.
    temp_Yrs = np.array(df_temp_Obs.index)

    # Check that the regression years are within the temperature data range.
    if start_regress < temp_Yrs.min():
        start_regress = temp_Yrs.min()
        print(
            'Regression start year is before reference temperature data range, '
            f'setting start year to temperature data minimum: {start_regress}')
    if end_regress > temp_Yrs.max():
        end_regress = temp_Yrs.max()
        print('Regression end year is after reference temperature data range, '
              f'setting end year to temperature data maximum: {end_regress}')

    # Check that the regression years are within the forcing data range
    if start_regress < forc_Yrs_min:
        start_regress = forc_Yrs_min
        print('Regression start year is before forcing data range, '
              f'setting start year to forcing data minimum: {start_regress}')
    if end_regress > forc_Yrs_max:
        end_regress = forc_Yrs_max
        print('Regression end year is after forcing data range, '
              f'setting end year to forcing data maximum: {end_regress}')

    print('Calculating GWI with the following parameters:')
    print(f'Regressed variables: {regress_vars}')
    print(f'Forcing range: {forc_Yrs_min}-{forc_Yrs_max}')
    print(f'Reference temperature range: {temp_Yrs.min()}-{temp_Yrs.max()}')
    print(f'Pre-industrial era: {start_pi}-{end_pi}')
    print(f'Truncation range: {start_trunc}-{end_trunc}')
    print(f'Regression range: {start_regress}-{end_regress}')
    print(f'Number of samples: {samples}')
    print(f'Include rates: {rate_toggle}')
    print(f'Include pre-industrial offset: {inc_pi_offset}')
    print(f'Include constant term in regression: {inc_reg_const}')
    print(f'Using model: {model_choice}')
    print(f'Plotting pruned piControl: {plot_piControl}')

    # CMIP6 PI-CONTROL
    timeframes = [1, 3, 30]
    df_temp_PiC = defs.load_PiC(scenario, n_yrs, start_pi, end_pi)
    df_temp_PiC = defs.filter_PiControl(df_temp_PiC, timeframes)
    df_temp_PiC.set_index(np.arange(n_yrs)+start_trunc, inplace=True)
    # NOTE: For 1850-2022, we get 183 realisations for piControl
    # NOTE: For 1850-2023, we get 181 realisations for piControl

    # Create a very rough estimate of the internal variability for the HadCRUT5
    # best estimate. This is used to roughly compare the piControl internal
    # variability to the observed internal variability to roughly remove
    # ensemble members that are too variable in piControl.
    # TODO: regress natural forcings out of this as well...
    # TODO: check whether picontrol simulations have natural forcings such as
    # volcanic and solar activity.
    temp_Obs_signal = defs.temp_signal(
        df_temp_Obs.quantile(q=0.5, axis=1).to_numpy(), 30, 'extrapolate')
    temp_Obs_IV = df_temp_Obs.quantile(q=0.5, axis=1) - temp_Obs_signal

    if plot_piControl:
        # PLOT THE INTERNAL VARIABILITY #######################################
        fig = plt.figure(figsize=(15, 10))
        gr.running_mean_internal_variability(
            timeframes, df_temp_PiC, temp_Obs_IV)
        gr.overall_legend(fig, loc='lower center', ncol=2, nrow=False)
        fig.suptitle(
            'Selected Sample of Internal Variability from CMIP6 PiControl')
        fig.savefig(f'{plot_folder}{output_path}' +
                    '0_Selected_CMIP6_Ensembles.png')

        # PLOT THE ENSEMBLE ###################################################
        fig = plt.figure(figsize=(15, 10))
        ax1 = plt.subplot2grid(shape=(1, 4), loc=(0, 0), rowspan=1, colspan=3)
        ax2 = plt.subplot2grid(shape=(1, 4), loc=(0, 3), rowspan=1, colspan=1)
        gr.plot_internal_variability_sample(
            ax1, ax2, df_temp_PiC, df_temp_Obs, temp_Obs_IV,
            sigmas, sigmas_all)
        gr.overall_legend(fig, loc='lower center', ncol=3, nrow=False)
        fig.suptitle(
            'Selected Sample of Internal Variability from CMIP6 pi-control')
        fig.savefig(f'{plot_folder}{output_path}' +
                    '1_Distribution_Internal_Variability.png')


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
        CMIP6_param_csv = ('models/FaIR_CMIP6_climresp.csv')
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

    # 1. Select random samples of the forcing data
    print(f'Forcing ensemble all: {len(df_forc.columns.get_level_values("ensemble").unique())}')
    print(f'Forcing ensemble all: {df_forc.columns.get_level_values("ensemble").unique()}')

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
    print(f'Max available ensemble: {_n_all}')

    # Print the randomly subsampled ensemble size
    _n_sub = (
        _nf *  # number of forcing ensembles
        len(CMIP6_param_df.columns.levels[0].unique()) *
        df_temp_Obs_subset.shape[1] *
        df_temp_PiC_subset.shape[1]
    )
    print(f'Sub-sampled ensemble size: {_n_sub}')

    # Parallelise GWI calculation, with each thread corresponding to a
    # single (model) parameterisation for FaIR.
    T1 = dt.datetime.now()
    with mp.Pool(os.cpu_count()) as p:
        print('Partialising Function')
        partial_GWI = functools.partial(
            GWI_faster,
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
            end_regress=end_regress
        )
        print('Calculating GWI (parallelised)', end=' ')
        results = p.map(partial_GWI, models)

    # Create a list of the names of the attributed warming variables
    # TODO: rename this to vars_Att or something, since Python syntax makes
    # vars_list red, so probably a bad idea.
    vars_list = forc_var_names
    # print(vars_list)
    vars_list.extend(defs.extra_vars(forc_var_names))
    # print(vars_list)

    T1_1 = dt.datetime.now()
    print(f'... took {T1_1 - T1}')

    print('Concatenating Results', end=' ')
    # Combine results from temperature attributions from all parallel model
    # emulations ('results' above is a list of arrays, one for each emulation).
    temp_Att_Results = np.concatenate(results, axis=2)
    # print(temp_Att_Results.shape)
    T2 = dt.datetime.now()
    print(f'... took {T2 - T1_1}')

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

    # PRODUCE FINAL RESULTS DATASETS ######################################

    # For multiple runs, we want to save the results with a unique identifier.
    # Use the current date and time of calculation for simplicity.
    current_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    iteration_id = current_time

    variation = (
        f'SCENARIO--{scenario}_' +
        f'VARIABLES--{"-".join(regress_vars)}_' +
        f'ENSEMBLE-SIZE--{n}_' +
        f'REGRESSED-YEARS--{start_regress}-{end_regress}_' +
        f'DATE-CALCULATED--{iteration_id}'
    )

    # NOTE TO SELF: multidimensional np.percentile() changes the order of
    # the axes, so that the axis along which you took the percentiles is
    # now the first axis, and the other axes are the remaining axes...

    # TIMESERIES RESULTS ######################################################
    print('Calculating percentiles', end=' ')
    gwi_timeseries_array = np.percentile(temp_Att_Results, sigmas_all, axis=2)
    dict_Results = {
        (var, sigma):
        gwi_timeseries_array[sigmas_all.index(sigma), :, vars_list.index(var)]
        for var in vars_list for sigma in sigmas_all
    }
    df_Results = pd.DataFrame(dict_Results, index=trunc_Yrs)
    df_Results.columns.names = ['variable', 'percentile']
    df_Results.index.name = 'Year'
    df_Results.to_csv(f'{results_folder}{output_path}' +
                      f'GWI_results_timeseries_{variation}.csv')

    T3 = dt.datetime.now()
    print(f'... took {T3 - T2}')

    # HEADLINE RESULTS ########################################################
    # NOTE: Currently, the headline results are calculated for two periods:
    # 1. The same years as the most recent IPCC reports (2017, and 2010-2019)
    # 2. The end of the truncation period, NOT the end of the regressed range.
    # The preferred range dependes on usage context, and can be changed later.

    if headline_toggle:
        print('Calculating headlines')

        # Define the years for the headline results (including SR1.5 repeats)
        if ((2017 in trunc_Yrs) and (end_regress != 2017)):
            # The final condition is to avoid duplicate calculations when
            # the end_regress is 2017.
            years_headlines = [2017, end_regress]
        else:
            years_headlines = [end_regress]

        # GWI-ANNUAL DEFINITION (SIMPLE VALUE IN A GIVEN YEAR) ################
        # if 2017 in trunc_Yrs:
        #     dfs = [df_Results.loc[], df_Results.loc[[end_regress]]]
        # else:
        #     dfs = [df_Results.loc[[end_regress]]]
        dfs = [df_Results.loc[[y]] for y in years_headlines]

        # SR15 DEFINITION (CENTRE OF 30-YEAR TREND) ###########################
        # Calculate the linear trend of the final 15 years of the timeseries
        # and use this to calculate the present-day warming
        print('Calculating SR15-definition temps', end=' ')
        for year in years_headlines:
            years_SR15 = ((year-15 <= trunc_Yrs) * (trunc_Yrs <= year))
            temp_Att_Results_SR15_recent = temp_Att_Results[years_SR15, :, :]

            # Calculate SR15-definition warming for each var-ens combination
            # See SR15 Ch1 1.2.1
            # temp_Att_Results_SR15 = np.apply_along_axis(
            #     final_value_of_trend, 0, temp_Att_Results_SR15_recent)
            temp_Att_Results_SR15 = np.empty(
                temp_Att_Results_SR15_recent.shape[1:])
            for vv in range(temp_Att_Results_SR15_recent.shape[1]):
                # print(vv)
                with mp.Pool(os.cpu_count()) as p:
                    times = [temp_Att_Results_SR15_recent[:, vv, ii]
                             for ii
                             in range(temp_Att_Results_SR15_recent.shape[2])]
                    # final_value_of_trend is from src/definitions.py
                    results = p.map(defs.final_value_of_trend, times)
                temp_Att_Results_SR15[vv, :] = np.array(results)

            # Obtain statistics
            gwi_headline_array = np.percentile(
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

        T4 = dt.datetime.now()
        print(f'... took {T4 - T3}')

        # AR6 DEFINITION (DECADE MEAN) ########################################
        print('Calculating AR6-definition temps', end=' ')
        if ((2010 in trunc_Yrs) and (2019 in trunc_Yrs) and (end_regress != 2019)):
            # The final condition is to avoid duplicate calculations when
            # the end_regress is 2019.
            years_headlines = [[2010, 2019], [end_regress-9, end_regress]]
        else:
            years_headlines = [[end_regress-9, end_regress]]

        for years in years_headlines:
            recent_years = ((years[0] <= trunc_Yrs) * (trunc_Yrs <= years[1]))
            temp_Att_Results_AR6 = \
                temp_Att_Results[recent_years, :, :].mean(axis=0)

            # Obtain statistics
            gwi_headline_array = np.percentile(
                temp_Att_Results_AR6, sigmas_all, axis=1)
            dict_Results = {
                (var, sigma): gwi_headline_array[sigmas_all.index(sigma),
                                                 vars_list.index(var)]
                for var in vars_list for sigma in sigmas_all
            }
            df_headlines_i = pd.DataFrame(
                dict_Results, index=['-'.join([str(y) for y in years])])
            df_headlines_i.columns.names = ['variable', 'percentile']
            df_headlines_i.index.name = 'Year'
            dfs.append(df_headlines_i)

        T5 = dt.datetime.now()
        print(f'... took {T5 - T4}')

        # CGWL DEFINITION (20YR MEAN CENTERED WITH PROJECTIONS) ###############
        # This definition was propsoed in Betts et al., 2023, and is not
        # currently used in the IPCC/IGCC assessments.

        # Check that we have the years needed for the CGWL definition
        if ((end_regress-9 in trunc_Yrs) and (end_regress+10 in trunc_Yrs)):
            print('Calculating CGWL definition temps', end=' ')
            for years in [[end_regress-9, end_regress+10]]:
                recent_years = ((years[0] <= trunc_Yrs) *
                                (trunc_Yrs <= years[1]))
                temp_Att_Results_CGWL = \
                    temp_Att_Results[recent_years, :, :].mean(axis=0)
                # Obtain statistics
                gwi_headline_array = np.percentile(
                    temp_Att_Results_CGWL, sigmas_all, axis=1)
                dict_Results = {
                    (var, sigma): gwi_headline_array[sigmas_all.index(sigma),
                                                     vars_list.index(var)]
                    for var in vars_list for sigma in sigmas_all
                }
                df_headlines_i = pd.DataFrame(
                    dict_Results, index=[
                        f"{'-'.join([str(y) for y in years])} (CGWL definition)"
                    ])
                df_headlines_i.columns.names = ['variable', 'percentile']
                df_headlines_i.index.name = 'Year'
                dfs.append(df_headlines_i)
        else:
            print('CGWL definition skipped; required number of projected' +
                  'years not available.')

        T6 = dt.datetime.now()
        print(f'... took {T6 - T5}')

        df_headlines = pd.concat(dfs, axis=0)
        df_headlines.to_csv(f'{results_folder}{output_path}' +
                            f'GWI_results_headlines_{variation}.csv')

    # RATE: AR6 DEFINITION
    if rate_toggle:
        T7 = dt.datetime.now()
        dfs_rates = []
        for year in np.arange(1950, end_trunc+1):
            print(f'Calculating AR6-definition warming rate: {year}', end='\r')
            recent_years = ((year-9 <= trunc_Yrs) * (trunc_Yrs <= year))
            ten_slice = temp_Att_Results[recent_years, :, :]

            # Calculate AR6-definition warming rate for each var-ens
            # combination. See AR6 WGI Chapter 3 Table 3.1.
            temp_Rate_Results = np.empty(
                ten_slice.shape[1:])
            # Only include 'Ant'
            for vv in range(ten_slice.shape[1]):
                # Parallelise over ensemble members
                with mp.Pool(os.cpu_count()) as p:
                    single_series = [ten_slice[:, vv, ii]
                                     for ii in range(ten_slice.shape[2])]
                    # final_value_of_trend is from src/definitions.py
                    results = p.map(defs.rate_func, single_series)
                temp_Rate_Results[vv, :] = np.array(results)

            # Obtain statistics
            gwi_rate_array = np.percentile(
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
        print('')
        print(f'... took {T8 - T7}')
