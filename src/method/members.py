"""Rebuilding ensemble members from the pieces (stage C).

The axes are named as everywhere in the engine:

    p  FaIR parameter set
    e  ERF ensemble member
    o  observed GMT member
    c  piControl segment
    t  year (here, the output years)
    v  forcing variable (e.g. GHG, Nat, OHF, and any sub-variables, e.g. co2)
    k  regressor: one per regressed variable, plus one for the constant term
       if it is used

A member is one choice of each source of uncertainty, (p, e, o, c). It is
rebuilt from the pieces made by stages A and B:

    β          = β_obs[p, e, o] + β_pic[p, e, c]      its k scaling factors
    T_att(t, v) = β[k(v)] · T_mod[p, e, t, v]          for every forcing
                                                       variable v

where k(v) is the regressor that scales v: v itself if it is regressed, or
the regressed variable it is part of (e.g. co2 is scaled by GHG's factor).
Then the diagnosed variables:

    Tot(t) = Σ over the regressed variables of T_att(t, v)  (+ the constant)
    Res(t) = T_obs[o](t) − Tot(t)       (NaN in years without observations)
    Ant(t) = T_att(t, GHG) + T_att(t, OHF)

Tot and Ant are written only when they are diagnosed, i.e. above the
regressed variables (see variables.extra_vars): when Tot is itself
regressed, its column is its own T_att, without the constant.
"""

import numpy as np

import src.method.variables as variables


def rebuild(T_mod, coefs, T_obs, member_index, forcing_vars, regress_vars,
            inc_reg_const, output_vars, years=slice(None)):
    """Rebuild members from the pieces: attributed warming for each one.

    T_mod: array (p, e, t, v), the modelled warming from stage A, relative to
        the pre-industrial period if used, over the output years
    coefs: regression.Coefficients from stage B: obs (p, e, o, k) and pic
        (p, e, c, k)
    T_obs: array (t, o), the observed GMT members over the output years, NaN
        in years without observations
    member_index: (p, e, o, c), four integer arrays of length N: member n is
        (p[n], e[n], o[n], c[n])
    forcing_vars: the names along T_mod's v axis
    regress_vars: the names of the regressed variables, in the order of the
        k axis (the constant, if used, is the last regressor)
    inc_reg_const: whether the regression has a constant term
    output_vars: the names along the output's variable axis, in order: the
        forcing variables, then the diagnosed ones (Ant, Tot, Res) that are
        not forcing variables
    years: which of the t axis to rebuild, e.g. a slice for a block of years
    Returns an array (t, variables, N), float32: the layout of the
    attribution results array.
    """
    # The four indices of every member
    p, e, o, c = member_index
    # Each member's scaling factors, summed from its two fits: (N, k)
    beta = coefs.obs[p, e, o] + coefs.pic[p, e, c]
    # Each member's modelled warming over the chosen years: (N, t, v)
    T_member = T_mod[p, e][:, years, :]

    # The regressor that scales each forcing variable: its own name if it is
    # regressed, or the regressed variable it is part of
    regressed_ancestor = variables.map_var_to_regression_aggregate(
        forcing_vars, regress_vars)
    # ... as positions on the k axis, one per forcing variable: (v,)
    k_of_v = [regress_vars.index(regressed_ancestor[v]) for v in forcing_vars]
    # Every forcing variable's attributed warming: (N, t, v) * (N, 1, v)
    T_att = T_member * beta[:, np.newaxis, k_of_v]

    # Where the regressed variables sit on the v axis
    v_regressed = [forcing_vars.index(v) for v in regress_vars]
    # The total fitted warming, the sum of the regressed variables: (N, t)
    total = T_att[:, :, v_regressed].sum(axis=-1)
    if inc_reg_const:
        # Plus each member's constant term, the last regressor: (N, 1)
        total += beta[:, -1:]

    # The output: (t, variables, N), filled one variable at a time
    n_years = T_member.shape[1]
    out = np.empty((n_years, len(output_vars), len(p)), dtype=np.float32)
    for j, var in enumerate(forcing_vars):
        # Each forcing variable's attributed warming, members last: (t, N)
        out[:, output_vars.index(var), :] = T_att[:, :, j].T
    # The variables diagnosed after the regression
    diagnosed = variables.extra_vars(regress_vars)
    if 'Tot' in diagnosed:
        # The total, when it is above the regressed variables
        out[:, output_vars.index('Tot'), :] = total.T
    if 'Ant' in diagnosed:
        # Anthropogenic warming: greenhouse gases plus other human forcings
        out[:, output_vars.index('Ant'), :] = (
            T_att[:, :, forcing_vars.index('GHG')] +
            T_att[:, :, forcing_vars.index('OHF')]).T
    # The residual, observations minus the total, for each member's observed
    # member o: (t, N), NaN where there are no observations
    out[:, output_vars.index('Res'), :] = T_obs[years][:, o] - total.T
    return out
