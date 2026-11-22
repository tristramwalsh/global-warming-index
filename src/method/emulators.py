"""Emulator adapters: the climate response to each forcing variable.

Stage A of the GWI:

    T_mod[p, e, t, v] = the warming in year t from forcing variable v of ERF
                        member e on its own, emulated with parameter set p

Each adapter feeds plain arrays to an emulator's own code, in the input
format that emulator expects, and returns its output as plain arrays. The
emulators themselves live, unchanged, in models/. This is the only module
that imports an emulator, so the rest of the code never sees an emulator's
own formats.
"""

import numpy as np
import pandas as pd

import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair


def forcing_array(df_forc, variables, members):
    """Forcing as an array (t, e, v), from a DataFrame with columns
    (variable, ensemble), in the order of `members` and `variables`."""
    return np.stack(
        [np.stack([df_forc[(var, member)].to_numpy() for var in variables],
                  axis=-1)
         for member in members],
        axis=1)


def run_fair_v2_alpha(forcing, years, params):
    """Run FaIR v2.0.0-alpha1 (models/FaIR_V2) on each forcing variable alone.

    An adapter: it puts the forcing into the input format of FaIR's own
    run_FaIR, runs it, and returns the warming as an array.

    forcing: array (t, e, v), effective radiative forcing (W/m²) for each
        year, ERF member and forcing variable
    years: the years of the forcing's t axis, consecutive (timestep 1 year)
    params: DataFrame of FaIR thermal parameter sets, in FaIR's format
        (columns (parameter set name, parameter)), e.g. as read from
        models/FaIR_V2/FaIR_CMIP6_climresp.csv. Every set it holds is run,
        in the order they appear.
    Returns T_mod, array (p, e, t, v): warming (K) relative to the first
    year, which FaIR starts from zero.

    FaIR's thermal response is linear in forcing, so each variable's warming
    is its own run: FaIR is driven by that variable's forcing alone, with
    zero emissions. Each call runs one parameter set and one variable for
    every ERF member at once, as FaIR vectorises over its "scenarios" (here,
    the ERF members).
    """
    forcing = np.asarray(forcing, dtype=float)
    years = np.asarray(years)
    n_years, n_members, n_vars = forcing.shape
    # FaIR sorts its scenarios by name, and returns them in that order. So
    # the ERF members are renamed with zero-padded numbers, which sort in the
    # order given: FaIR's output then lines up with the input's e axis.
    members_renamed = [f'member_{e:06d}' for e in range(n_members)]
    parameter_set_names = params.columns.get_level_values(0).unique()

    T_mod = np.empty((len(parameter_set_names), n_members, n_years, n_vars))
    for p, parameter_set_name in enumerate(parameter_set_names):
        # One parameter set, with only that set in the column index: FaIR
        # counts the sets from the index's levels.
        params_one_set = params[parameter_set_name].copy()
        params_one_set.columns = pd.MultiIndex.from_product(
            [[parameter_set_name], params_one_set.columns])
        for v in range(n_vars):
            # FaIR won't run without emissions or concentrations, so give
            # it zero emissions; the forcing drives the model.
            emissions_in = fair.return_empty_emissions(
                df_to_copy=False, start_year=years.min(),
                end_year=years.max(), timestep=1, scen_names=members_renamed)
            forcing_in = fair.return_empty_forcing(
                df_to_copy=False, start_year=years.min(),
                end_year=years.max(), timestep=1, scen_names=members_renamed)
            for e, member in enumerate(members_renamed):
                forcing_in[member] = forcing[:, e, v]
            output = fair.run_FaIR(emissions_in=emissions_in,
                                   forcing_in=forcing_in,
                                   thermal_parameters=params_one_set,
                                   show_run_info=False)
            # Warming (years, members): one column per member, as there is
            # one parameter set
            T_mod[p, :, :, v] = output['T'].to_numpy().T
    return T_mod
