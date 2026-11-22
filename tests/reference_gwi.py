"""A slow, plain reference implementation of the GWI attribution.

    python tests/reference_gwi.py

It computes the attributed warming one ensemble member at a time, in plain
loops, on tiny inputs (a few members of each source). It has two jobs:

1. **To state the method plainly.** `attribute_member` below is the whole
   attribution for one member, in about forty lines.
2. **To check the fast engine.** Run as a script, it attributes the same tiny
   inputs with the reference and with the engine in gwi.py, for several
   configurations, and checks that they agree. It prints PASS or FAIL for
   each, and exits with status 1 if any fails.

The method, for one member: a member is one choice of each source of
uncertainty, (p, e, o, c):

    p  a FaIR parameter set (climate response)
    e  an ERF ensemble member (radiative forcing)
    o  an observed GMT ensemble member (observational uncertainty)
    c  a piControl segment (internal variability)

    T_mod[p, e](t, v) = FaIR's warming in response to forcing variable v of
                        member e alone, with parameter set p, minus its
                        pre-industrial mean
    β[p, e, o, c]     = least-squares fit of  T_obs[o] + T_pic[c]  onto the
                        regressed T_mod variables (plus a constant if used),
                        over the regression years
    T_att(t, v)       = β[v's regressed variable] · T_mod(t, v), for every
                        forcing variable v, including sub-variables (co2 is
                        scaled by the GHG coefficient)
    Tot = Σ regressed T_att (+ constant),  Res = T_obs[o] − Tot,  Ant = GHG + OHF

The engine computes β as two separate fits, one onto T_obs[o] and one onto
T_pic[c], and adds them. Least squares is linear in the target, so this
gives the same β, and it is what makes factor-then-sample possible. The
reference fits their sum directly, so agreement also checks that linearity.

Members are ordered (p, e, o, c), with the last index varying fastest, as in
the engine's results array.
"""

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from anywhere: the repository root holds gwi.py, src/ and models/
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import models.FaIR_V2.FaIRv2_0_0_alpha1.fair.fair_runner as fair  # noqa: E402
import src.inputs.input_datasets as datasets  # noqa: E402
from src.method.variables import SUB_VAR_MAPPING  # noqa: E402

# The variables each regression diagnoses after the fit. The regressed
# variables are given sorted, as gwi.py sorts them.
DIAGNOSED = {
    ('GHG', 'Nat', 'OHF'): ['Ant', 'Tot', 'Res'],
    ('Ant', 'Nat'): ['Tot', 'Res'],
    ('Tot',): ['Res'],
}

# Agreement required between the reference (float64) and the engine, which
# stores members as float32 (~1e-7 °C rounding for values of a few °C).
ATOL = 1e-5


###############################################################################
# THE METHOD, ONE MEMBER AT A TIME ############################################
###############################################################################

def emulate(forcing, params):
    """FaIR's warming in response to each forcing variable on its own.

    forcing: DataFrame, one ERF member: index year, one column per variable
    params: DataFrame, one FaIR parameter set, in FaIR's thermal-parameter
        format (columns (parameter set name, parameter))
    Returns a DataFrame of warming with the same index and columns.
    """
    years = forcing.index
    warming = {}
    for var in forcing.columns:
        # FaIR needs an emissions input too: an empty one, as forcing drives
        # the model here.
        emissions = fair.return_empty_emissions(
            df_to_copy=False, start_year=years.min(), end_year=years.max(),
            timestep=1, scen_names=['member'])
        forcing_in = fair.return_empty_forcing(
            df_to_copy=False, start_year=years.min(), end_year=years.max(),
            timestep=1, scen_names=['member'])
        forcing_in['member'] = forcing[var].to_numpy()
        output = fair.run_FaIR(emissions_in=emissions, forcing_in=forcing_in,
                               thermal_parameters=params,
                               show_run_info=False)
        warming[var] = output['T'].to_numpy()[:, 0]
    return pd.DataFrame(warming, index=years)


def regressed_ancestor(var, regress_vars):
    """The regressed variable whose coefficient scales `var`.

    That is `var` itself if it is regressed, or else the regressed variable
    it is part of, following SUB_VAR_MAPPING upwards (e.g. co2 -> GHG, or
    co2 -> GHG -> Ant when regressing Ant and Nat).
    """
    while var not in regress_vars:
        var = next(parent for parent, children in SUB_VAR_MAPPING.items()
                   if var in children)
    return var


def attribute_member(T_mod, T_obs, T_pic, regress_vars, inc_reg_const,
                     preindustrial, regress, trunc):
    """Attributed warming for one member (p, e, o, c).

    T_mod: DataFrame, FaIR's warming for (p, e) from `emulate`, by year
    T_obs: Series, observed GMT member o, by year
    T_pic: Series, piControl segment c, by year
    regress_vars: sorted list of regressed variables, e.g. ['GHG', 'Nat', 'OHF']
    inc_reg_const: whether the regression includes a constant
    preindustrial: (start, end) years of the pre-industrial period, or None
    regress, trunc: (start, end) years of the regression and of the output
    Returns a DataFrame indexed by the truncation years, with a column for
    every forcing variable and every diagnosed variable.
    """
    # 1. Express the modelled warming relative to its pre-industrial mean
    if preindustrial is not None:
        T_mod = T_mod - T_mod.loc[preindustrial[0]:preindustrial[1]].mean()

    # 2. Fit the observations, plus a sample of internal variability, onto
    #    the modelled warming of the regressed variables
    X = T_mod.loc[regress[0]:regress[1], regress_vars].to_numpy()
    if inc_reg_const:
        X = np.column_stack([X, np.ones(len(X))])
    y = (T_obs.loc[regress[0]:regress[1]].to_numpy() +
         T_pic.loc[regress[0]:regress[1]].to_numpy())
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    coef = dict(zip(regress_vars, beta))
    constant = beta[-1] if inc_reg_const else 0.0

    # 3. Scale every forcing variable by its regressed variable's coefficient
    T_trunc = T_mod.loc[trunc[0]:trunc[1]]
    att = pd.DataFrame(
        {var: coef[regressed_ancestor(var, regress_vars)] * T_trunc[var]
         for var in T_trunc.columns})

    # 4. The diagnosed variables
    total = att[regress_vars].sum(axis=1) + constant
    diagnosed = DIAGNOSED[tuple(regress_vars)]
    if 'Ant' in diagnosed:
        att['Ant'] = att['GHG'] + att['OHF']
    if 'Tot' in diagnosed:
        att['Tot'] = total
    # The residual exists only in years with observations
    att['Res'] = T_obs.reindex(att.index) - total
    return att


def reference_gwi(inputs):
    """Attribute every member of `inputs` (see tiny_inputs), one at a time.

    Returns (results, output_vars): results has shape (years, variables,
    members), members ordered (p, e, o, c); output_vars names the variable
    axis, in the engine's order (sorted forcing variables, then the
    diagnosed ones that are not forcing variables).
    """
    forc = inputs['df_forc']
    forc_vars = sorted(forc.columns.get_level_values('variable').unique())
    erf_members = forc.columns.get_level_values('ensemble').unique()
    diagnosed = DIAGNOSED[tuple(inputs['regress_vars'])]
    output_vars = forc_vars + [v for v in diagnosed if v not in forc_vars]

    members = []
    for p in inputs['models']:
        # FaIR's format for one parameter set: columns (name, parameter),
        # rebuilt so that the column index holds only this set (FaIR counts
        # the parameter sets from the index's levels)
        params = inputs['df_params'][p].copy()
        params.columns = pd.MultiIndex.from_product([[p], params.columns])
        for e in erf_members:
            forcing = forc.xs(e, axis=1, level='ensemble')[forc_vars]
            T_mod = emulate(forcing, params)
            for o in inputs['df_temp_Obs'].columns:
                for c in inputs['df_temp_PiC'].columns:
                    att = attribute_member(
                        T_mod, inputs['df_temp_Obs'][o],
                        inputs['df_temp_PiC'][c], inputs['regress_vars'],
                        inputs['inc_reg_const'], inputs['preindustrial'],
                        inputs['regress'], inputs['trunc'])
                    members.append(att[output_vars].to_numpy())
    return np.stack(members, axis=2), output_vars


###############################################################################
# TINY INPUTS #################################################################
###############################################################################

@functools.lru_cache(maxsize=None)
def _load_erf(erf_id, regress_vars, include_sub_vars):
    return datasets.load_ERF(erf_id, list(regress_vars), 'all',
                             include_sub_vars=include_sub_vars)


@functools.lru_cache(maxsize=None)
def _load_gmt(gmt_id, preindustrial):
    start_pi, end_pi = preindustrial if preindustrial else (None, None)
    return datasets.load_GMT(gmt_id, 'all', start_pi, end_pi)


@functools.lru_cache(maxsize=None)
def _load_pic(years, preindustrial):
    start_pi, end_pi = preindustrial if preindustrial else (None, None)
    df = datasets.load_PiC('CMIP6', list(years), start_pi, end_pi)
    return datasets.filter_PiControl(df, [1, 3, 30])


@functools.lru_cache(maxsize=None)
def _load_params():
    return pd.read_csv(REPO / 'models/FaIR_V2/FaIR_CMIP6_climresp.csv',
                       index_col=[0], header=[0, 1])


def tiny_inputs(erf_id='IGCC-2025', gmt_id='HadCRUT-2025',
                regress_vars=('GHG', 'Nat', 'OHF'), include_sub_vars=False,
                inc_reg_const=False, preindustrial=(1850, 1900),
                regress=(1850, 2025), trunc=(1850, 2050),
                n_params=2, n_erf=3, n_gmt=3, n_pic=3, seed=0):
    """A few members of each source of real input data, prepared as gwi.py
    prepares them, as a dict of everything an attribution needs.

    Members are chosen at random (with `seed`); fewer are taken when a
    dataset has fewer.
    """
    rng = np.random.default_rng(seed)
    regress_vars = sorted(regress_vars)

    def pick(labels, n):
        labels = list(labels)
        chosen = rng.choice(len(labels), size=min(n, len(labels)),
                            replace=False)
        return [labels[i] for i in sorted(chosen)]

    df_forc = _load_erf(erf_id, tuple(regress_vars), include_sub_vars)
    # As gwi.py: truncation is clipped to the ERF years
    trunc = (max(trunc[0], df_forc.index.min()),
             min(trunc[1], df_forc.index.max()))
    erf_members = pick(df_forc.columns.get_level_values('ensemble').unique(),
                       n_erf)
    df_forc = df_forc.loc[:, (slice(None), erf_members)]

    df_temp_Obs = _load_gmt(gmt_id, preindustrial)
    df_temp_Obs = df_temp_Obs[pick(df_temp_Obs.columns, n_gmt)].copy()

    # As gwi.py: segments as long as the GMT record, indexed by its years
    gmt_years = tuple(_load_gmt(gmt_id, preindustrial).index)
    df_temp_PiC = _load_pic(gmt_years, preindustrial)
    df_temp_PiC = df_temp_PiC[pick(df_temp_PiC.columns, n_pic)].copy()

    df_params = _load_params()
    models = pick(df_params.columns.levels[0].unique(), n_params)

    for source, n in [('ERF', len(erf_members)),
                      ('GMT', df_temp_Obs.shape[1]),
                      ('piControl', df_temp_PiC.shape[1]),
                      ('FaIR parameter', len(models))]:
        if n == 0:
            raise ValueError(f'No {source} members left for these settings.')

    return dict(df_forc=df_forc, df_params=df_params, models=models,
                df_temp_Obs=df_temp_Obs, df_temp_PiC=df_temp_PiC,
                regress_vars=regress_vars, inc_reg_const=inc_reg_const,
                preindustrial=preindustrial, regress=regress, trunc=trunc)


###############################################################################
# THE ENGINE, ON THE SAME INPUTS ##############################################
###############################################################################

def engine_gwi(inputs):
    """Attribute `inputs` with the engine in gwi.py, as `gwi.py` would.

    Returns (results, output_vars), laid out as for reference_gwi.
    """
    import gwi  # gwi.py's main block doesn't run on import
    start_pi, end_pi = inputs['preindustrial'] or (None, None)
    blocks = []
    for model in inputs['models']:
        results, output_vars = gwi.GWI_faster(
            model,
            inc_reg_const=inputs['inc_reg_const'],
            inc_pi_offset=inputs['preindustrial'] is not None,
            df_forc=inputs['df_forc'],
            df_params=inputs['df_params'],
            df_temp_PiC=inputs['df_temp_PiC'],
            df_temp_Obs=inputs['df_temp_Obs'],
            start_trunc=inputs['trunc'][0], end_trunc=inputs['trunc'][1],
            start_pi=start_pi, end_pi=end_pi,
            start_regress=inputs['regress'][0],
            end_regress=inputs['regress'][1],
            regress_vars=inputs['regress_vars'])
        blocks.append(results)
    return np.concatenate(blocks, axis=2), output_vars


###############################################################################
# THE CHECK ###################################################################
###############################################################################

# Each configuration exercises a different part of the method.
CONFIGURATIONS = {
    'three regressors; residual NaN beyond the observations':
        dict(),
    'sub-variables, with the regression constant':
        dict(include_sub_vars=True, inc_reg_const=True),
    'two regressors (Ant, Nat), with sub-variables':
        dict(regress_vars=('Ant', 'Nat'), include_sub_vars=True),
    'one regressor (Tot), with sub-variables':
        dict(regress_vars=('Tot',), include_sub_vars=True),
    'historical-only regression ending 1990':
        dict(regress=(1850, 1990), trunc=(1850, 2025)),
    'no pre-industrial offset, with the regression constant':
        dict(preindustrial=None, inc_reg_const=True),
    'a different pre-industrial period (1981-2010), with the constant':
        dict(preindustrial=(1981, 2010), inc_reg_const=True),
    'simulated large ensemble (one ERF member)':
        dict(erf_id='SMILE_ESM-SSP245', gmt_id='SMILE_ESM-SSP245',
             trunc=(1850, 2100)),
}


def compare(reference, engine):
    """Largest absolute difference, and whether NaNs fall in the same places."""
    nan_ref, nan_eng = np.isnan(reference), np.isnan(engine)
    same_nans = np.array_equal(nan_ref, nan_eng)
    both = ~nan_ref & ~nan_eng
    largest = np.abs(reference[both] - engine[both]).max()
    return largest, same_nans


def main():
    failures = 0
    for name, settings in CONFIGURATIONS.items():
        inputs = tiny_inputs(**settings)
        reference, ref_vars = reference_gwi(inputs)
        try:
            engine, eng_vars = engine_gwi(inputs)
        except Exception as error:  # report it, and go on to the next
            print(f'FAIL  {name}: the engine raised '
                  f'{type(error).__name__}: {error}')
            failures += 1
            continue
        if ref_vars != eng_vars or reference.shape != engine.shape:
            print(f'FAIL  {name}: variables {eng_vars} and shape '
                  f'{engine.shape}, expected {ref_vars} and '
                  f'{reference.shape}')
            failures += 1
            continue
        largest, same_nans = compare(reference, engine)
        ok = largest <= ATOL and same_nans
        failures += not ok
        print(f'{"PASS" if ok else "FAIL"}  {name}: '
              f'{reference.shape[2]} members x {len(ref_vars)} variables, '
              f'largest difference {largest:.1e} °C'
              + ('' if same_nans else ', NaNs in different places'))
    print(f'\n{len(CONFIGURATIONS) - failures} of {len(CONFIGURATIONS)} '
          f'configurations agree (tolerance {ATOL:g} °C).')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
