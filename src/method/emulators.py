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
    is its own run, driven by that variable's forcing alone. Each call runs
    one variable for every ERF member and every parameter set at once: FaIR
    vectorises over its "scenarios" (here, the ERF members) and over its
    thermal parameter sets.

    FaIR won't run without emissions (or concentrations) of at least one
    gas, so it is given zero emissions of one gas, CO2. With no emissions,
    CO2 stays at its pre-industrial concentration, and its forcing (which
    depends only on the concentration's departure from pre-industrial) is
    exactly zero in every year: the forcing given here drives the model
    alone. FaIR's default inputs hold all 52 of its gases, each zero in the
    same way, so the warming is identical to the last bit; but then every
    gas's cycle is integrated for every ERF member, which took almost all
    of the run time.
    """
    # Work in float64, whatever the forcing's precision
    forcing = np.asarray(forcing, dtype=float)
    # The years, as an array
    years = np.asarray(years)
    # The sizes of the forcing: years, ERF members, forcing variables
    n_years, n_members, n_vars = forcing.shape
    # FaIR sorts its scenarios by name, and returns them in that order. So
    # the ERF members are renamed with zero-padded numbers, which sort in the
    # order given: FaIR's output then lines up with the input's e axis.
    members_renamed = [f'member_{e:06d}' for e in range(n_members)]
    # The parameter sets, in the order they appear in the table: the p axis
    parameter_set_names = params.columns.get_level_values(0).unique()
    # The table of parameter sets, with only the sets it holds in its column
    # index (a selection from a larger table keeps the others' names, and
    # FaIR counts the sets from the index)
    params = params.copy()
    params.columns = params.columns.remove_unused_levels()

    # The one gas FaIR is given, and its gas cycle parameters, in FaIR's
    # format
    gases = ['carbon_dioxide']
    gas_parameters = fair.get_gas_parameter_defaults(choose_gases=gases)
    # Zero emissions of that gas for every ERF member, every year
    emissions_in = fair.return_empty_emissions(
        df_to_copy=False, start_year=years.min(), end_year=years.max(),
        timestep=1, scen_names=members_renamed, gases_in=gases)
    # FaIR's forcing table: one column per ERF member, in the order of
    # members_renamed, filled in below for each variable
    forcing_in = fair.return_empty_forcing(
        df_to_copy=False, start_year=years.min(), end_year=years.max(),
        timestep=1, scen_names=members_renamed)

    # The modelled warming, filled in one forcing variable at a time
    T_mod = np.empty((len(parameter_set_names), n_members, n_years, n_vars))
    for v in range(n_vars):
        # This variable's forcing for every ERF member: (t, e), whose columns
        # line up with the table's
        forcing_in.loc[:, :] = forcing[:, :, v]
        # Run every ERF member with every parameter set
        output = fair.run_FaIR(emissions_in=emissions_in,
                               forcing_in=forcing_in,
                               gas_parameters=gas_parameters,
                               thermal_parameters=params,
                               show_run_info=False)
        # Warming (years, columns), one column per (ERF member, parameter
        # set); FaIR orders the parameter sets by name, so take each by name
        warming = output['T']
        for p, parameter_set_name in enumerate(parameter_set_names):
            # This parameter set's warming, one column per ERF member:
            # (t, e)
            warming_one_set = warming.xs(
                parameter_set_name, axis=1, level='Thermal set').to_numpy()
            # Members first, as T_mod's axes: (e, t)
            T_mod[p, :, :, v] = warming_one_set.T
    return T_mod
