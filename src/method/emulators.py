"""Emulator adapters: the climate response to each forcing variable.

Stage A of the GWI:

    T_mod[p, e, t, v] = the warming in year t from forcing variable v of ERF
                        member e on its own, emulated with parameter set p

Each adapter takes the inputs as labelled DataArrays (the forcing, and the
emulator's parameters), feeds them to the emulator's own code in the input
format that emulator expects, and returns its output as a plain array. The
emulators themselves live, unchanged, in models/. This is the only module
that imports an emulator, so the rest of the code never sees an emulator's
own formats.
"""

import numpy as np
import pandas as pd
import xarray as xr

import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair


# The FaIR v2.0 thermal parameter sets, one per CMIP6 model, copied from
# models/FaIR_V2/FaIRv2_0_0_alpha1/fair/util/parameter-sets/CMIP6_climresp.csv
# and committed for transparency
FAIR_PARAMETERS_CSV = 'models/FaIR_V2/FaIR_CMIP6_climresp.csv'


def fair_parameter_sets():
    """The FaIR v2.0 thermal parameter sets, sorted by model name: the
    order of the p axis.

    Returns a DataArray (param_member, parameter, box): for each model's
    parameter set (each member of the p axis), FaIR's parameters (d, the
    time constants, and q, the sensitivities) of each of its three thermal
    boxes.

    The file is FaIR's own table, rows the parameters and columns (model,
    box); fair_parameter_table turns the DataArray back into it.
    """
    # The table, its columns sorted by model name
    df_params = pd.read_csv(
        FAIR_PARAMETERS_CSV, index_col=[0], header=[0, 1]).sort_index(
            axis=1, level=0, sort_remaining=False)
    # The models, in that order, and the boxes
    models = df_params.columns.get_level_values(0).unique().to_list()
    boxes = df_params.columns.get_level_values(1).unique().to_list()
    # Each model's parameters of each box: (param_member, parameter, box)
    values = np.stack([df_params[model][boxes].to_numpy()
                       for model in models])
    return xr.DataArray(
        values, dims=('param_member', 'parameter', 'box'),
        coords={'param_member': models,
                'parameter': df_params.index.to_list(),
                'box': boxes},
        name='fair_parameters')


def fair_parameter_table(params):
    """FaIR's own table of thermal parameter sets, from the DataArray.

    params:
        a DataArray (param_member, parameter, box), as from
        fair_parameter_sets, or a selection of its parameter sets
    Returns a DataFrame: rows the parameters, columns (parameter set,
    box), the parameter sets in the order of params, as FaIR's run_FaIR
    takes them (and as models/FaIR_V2/FaIR_CMIP6_climresp.csv lays them
    out).
    """
    # The columns: each parameter set's boxes
    columns = pd.MultiIndex.from_product(
        [params.param_member.to_numpy(), params.box.to_numpy()])
    # One row per parameter, its values for each (set, box) in that order:
    # (parameter, param_member · box)
    values = params.transpose('parameter', 'param_member', 'box').to_numpy()
    values = values.reshape(params.sizes['parameter'], -1)
    return pd.DataFrame(values, index=params.parameter.to_numpy(),
                        columns=columns)


def run_fair_v2_alpha(forcing, params):
    """Run FaIR v2.0.0-alpha1 (models/FaIR_V2) on each forcing variable alone.

    An adapter: it puts the forcing into the input format of FaIR's own
    run_FaIR, runs it, and returns the warming as an array.

    forcing: DataArray (year, erf_member, forcing_variable), effective
        radiative forcing (W/m²) for each year, ERF member and variable, the
        years consecutive (timestep 1 year), as from
        input_datasets.load_ERF
    params: DataArray (param_member, parameter, box) of FaIR thermal
        parameter sets, as from fair_parameter_sets (or a selection of its
        sets). Every set it holds is run, in its order.
    Returns T_mod, array (p, e, t, v): warming (K) relative to the first
    year, which FaIR starts from zero, the axes in the order of params'
    parameter sets and forcing's ERF members, years and variables.

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
    # The years, as an array
    years = forcing.year.to_numpy()
    # The forcing as an array, (t, e, v), in float64, whatever its
    # precision
    forcing = forcing.to_numpy().astype(float)
    # The sizes of the forcing: years, ERF members, forcing variables
    n_years, n_members, n_vars = forcing.shape
    # FaIR sorts its scenarios by name, and returns them in that order. So
    # the ERF members are renamed with zero-padded numbers, which sort in the
    # order given: FaIR's output then lines up with the input's e axis.
    members_renamed = [f'member_{e:06d}' for e in range(n_members)]
    # The parameter sets, in order: the p axis
    parameter_set_names = params.param_member.to_numpy()
    # ... in FaIR's own table
    params = fair_parameter_table(params)

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
