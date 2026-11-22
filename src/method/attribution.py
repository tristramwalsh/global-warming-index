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

    Both the attribution and the reference test (tests/reference_gwi.py)
    need this list, so it lives here rather than being worked out twice.
    """
    forc_vars = sorted(
        df_forc.columns.get_level_values('variable').unique().to_list())
    diagnosed = variables.extra_vars(regress_vars)
    return forc_vars + [v for v in diagnosed if v not in forc_vars]


def prior_warming(df_params, df_forc):
    """Calculate prior (pre-constrained) warming: the modelled warming from
    each forcing variable, before any regression onto observations.

    df_params: FaIR thermal parameter sets, columns (set name, parameter);
        every set is run, in the order they appear
    df_forc: forcing, columns (variable, ensemble), for every ERF member
    Returns an array (t, v, p · e): (years, variables, members), variables
    sorted, members ordered with the parameter set varying slowest (one
    block of ERF members per parameter set).
    """
    # The forcing variables, sorted, and the ERF members, in order
    var_list_ERF = sorted(df_forc.columns.get_level_values(
        "variable").unique().to_list())
    ens_list_ERF = df_forc.columns.get_level_values(
        "ensemble").unique().to_list()
    # Stage A for every parameter set and ERF member: (p, e, t, v)
    T_mod = emulators.run_fair_v2_alpha(
        emulators.forcing_array(df_forc, var_list_ERF, ens_list_ERF),
        df_forc.index.to_numpy(), df_params)
    # Reorder the axes to (t, v, p, e)
    T_mod = T_mod.transpose(2, 3, 0, 1)
    # Merge the parameter sets and ERF members into one member axis, p
    # varying slowest: (t, v, p · e)
    n_years, n_vars = T_mod.shape[:2]
    temp_Priors = T_mod.reshape(n_years, n_vars, -1)
    return temp_Priors
