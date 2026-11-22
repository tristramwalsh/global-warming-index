"""Which variables the GWI outputs hold, for gwi.py: the variables along the
attribution's results array, and the aggregate variables (Ant, Tot), added
as sums of their parts to the priors and to the forcing.
"""

import numpy as np

import src.method.variables as variables


def attribution_output_vars(forcing_vars, regress_vars):
    """Return the variable names along axis 1 of the attribution array.

    forcing_vars: the forcing variables, sorted (the v axis of T_mod)
    regress_vars: the regressed variables

    These are the forcing variables, followed by the aggregates diagnosed
    after the regression (see extra_vars) that are not already among them.
    The order matters: it is the order of the variable axis of
    temp_Att_Results, and of the output CSV columns.
    """
    diagnosed = variables.extra_vars(regress_vars)
    return list(forcing_vars) + [v for v in diagnosed if v not in forcing_vars]


def add_aggregate_variables(per_variable, forcing_vars, regress_vars):
    """Add the aggregate variables that the attribution diagnoses (Ant, Tot),
    as sums of their parts, to an array of forcing variables.

    Stage A runs each forcing variable on its own, so its modelled warming
    (the prior warming) holds no aggregates; nor does the forcing, unless
    they are regressed. The attribution diagnoses them after the
    regression (members.rebuild); this adds the same aggregates to the
    priors and to the ERF output, so all three hold the same variables.

    per_variable: array (..., v), the last axis along forcing_vars: e.g. the
        forcing (t, e, v), or stage A's modelled warming T_mod (p, e, t, v)
    forcing_vars: the names along its last axis
    regress_vars: the regressed variables, which decide the aggregates
    Returns (array (..., v'), names): every forcing variable and every
    aggregate that is not among them, sorted by name.

    The aggregates are those of variables.extra_vars, less the residual
    (Res), which needs observations, and the same sums as members.rebuild
    uses after the regression, with every scaling factor 1:

        Tot = Σ over the regressed variables     Ant = GHG + OHF

    Summing modelled warming like this needs a linear emulator: the warming
    from a sum of forcings is then the sum of their warmings. FaIR's thermal
    response is linear. A nonlinear emulator would need each aggregate's
    forcing run through it on its own instead. Every input dataset's own
    Ant and Tot forcings equal the sums of their parts to within 1e-9 W/m²
    (rounding in its files), so for forcing this changes nothing either.
    """
    # The aggregates to add: those diagnosed, other than the residual, that
    # are not already forcing variables
    aggregates = [v for v in variables.extra_vars(regress_vars)
                  if v != 'Res' and v not in forcing_vars]
    # The parts each aggregate sums
    parts = {'Tot': list(regress_vars), 'Ant': ['GHG', 'OHF']}
    # Each aggregate, summed over its parts: one (...) array each
    summed = [sum(per_variable[..., forcing_vars.index(part)]
                  for part in parts[aggregate])
              for aggregate in aggregates]
    # Every variable's name, forcing variables then aggregates
    names = list(forcing_vars) + aggregates
    if summed:
        # The aggregates, along a new last axis: (..., aggregates)
        summed = np.stack(summed, axis=-1)
        # Appended after the forcing variables: (..., v')
        per_variable = np.concatenate([per_variable, summed], axis=-1)
    # The order that sorts the names
    order = sorted(range(len(names)), key=lambda i: names[i])
    # Every variable, sorted by name: (..., v')
    per_variable = per_variable[..., order]
    return per_variable, [names[i] for i in order]
