"""Pieces of the GWI attribution calculation in gwi.py: the shape of the
results array, and the prior (unconstrained) warming from FaIR.
"""

import src.method.emulators as emulators
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
    """Calculate prior (pre-constrained) warming: the modelled warming from
    each forcing variable, before any regression onto observations.

    model_choice: the FaIR parameter set to use, a name in df_params
    df_params: FaIR thermal parameter sets, columns (set name, parameter)
    df_forc: forcing, columns (variable, ensemble), for every ERF member
    Returns an array (years, variables, ERF members), variables sorted.
    """
    var_list_ERF = sorted(df_forc.columns.get_level_values(
        "variable").unique().to_list())
    ens_list_ERF = df_forc.columns.get_level_values(
        "ensemble").unique().to_list()
    # Stage A, for the one parameter set model_choice. The adapter runs every
    # parameter set in the table it is given, in the table's order, and
    # returns them along the first axis (p) of T_mod, shape (p, e, t, v).
    # Given only model_choice's columns, it runs just that set: p has length
    # 1, and T_mod[0] is the warming with parameter set model_choice.
    params_this_set = df_params[[model_choice]]
    T_mod = emulators.run_fair_v2_alpha(
        emulators.forcing_array(df_forc, var_list_ERF, ens_list_ERF),
        df_forc.index.to_numpy(), params_this_set)
    # Take that one parameter set, and reorder the axes from (e, t, v) to
    # the (t, v, e) of the priors array: (years, variables, ERF members).
    temp_Mod_array = T_mod[0].transpose(1, 2, 0)
    return temp_Mod_array
