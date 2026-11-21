"""Pieces of the GWI attribution calculation in gwi.py: the shape of the
results array, and the prior (unconstrained) warming from FaIR.
"""
import numpy as np
import pandas as pd

import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair
import src.method.variables as variables


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
    diagnosed = variables.extra_vars(regress_vars)
    return forc_vars + [v for v in diagnosed if v not in forc_vars]


def ensemble_members_per_model(df_forc, df_temp_Obs, df_temp_PiC):
    """Return how many ensemble members one FaIR parameterisation produces.

    Each emulation crosses every sampled ERF ensemble member with every
    sampled observational temperature member and every sampled piControl
    member, so the count is simply the product of the three.
    """
    n_erf = len(df_forc.columns.get_level_values('ensemble').unique())
    return n_erf * df_temp_Obs.shape[1] * df_temp_PiC.shape[1]


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
