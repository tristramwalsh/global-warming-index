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

from concurrent.futures import ThreadPoolExecutor

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


def rebuild_into(out, T_mod, coefs, T_obs, member_index, forcing_vars,
                 regress_vars, inc_reg_const, output_vars, n_threads=1,
                 bytes_per_block=2**21):
    """Rebuild every member into `out`, in blocks of members, over threads.

    out: array (t, variables, N), to fill: the results array
    member_index: (p, e, o, c), four integer arrays of length N, e.g. from a
        sampling design; member n is written to out[:, :, n]
    T_mod, coefs, T_obs, forcing_vars, regress_vars, inc_reg_const,
        output_vars: as for `rebuild`
    n_threads: how many threads rebuild member blocks at once (e.g. the
        job's CPUs)
    bytes_per_block: the memory of one member block's modelled warming
        (every year and forcing variable of its members, in T_mod's
        precision), which sets how many members each block holds; default
        2 MiB

    A member block is a run of consecutive members, out[:, :, start:stop]:
    a slice along the member axis only, holding every year and variable.
    Each thread rebuilds one member block at a time, with `rebuild`, and
    writes it straight into its columns of `out`. While it works it holds
    about 3.5 × bytes_per_block (the block's modelled warming, its
    attributed warming, and their output), so the rebuild needs about
    3.5 × n_threads × bytes_per_block on top of `out`: about 0.2 GiB on 28
    threads. The block size doesn't depend on the memory or the CPUs
    available. Measured on 28 threads (on shared nodes, so ±15%): blocks of
    1 to 16 MiB are within ~20% of each other, and which is fastest shifts
    with the number of members; 2 MiB was within ~10% of the fastest both
    at 0.9 and at 6 million members. Blocks of 32 MiB and more are slower
    (32 MiB took 20–30% longer) and need far more memory; much smaller
    blocks, below ~1 MiB, spend more of their time in Python overhead.

    The results don't depend on the block size or on the number of threads.
    Threads, not processes: the rebuild is numpy array operations, which
    release Python's lock while they run, and every thread can write
    straight into `out` without copying it.
    """
    # How many members there are
    n_members = len(member_index[0])
    # The memory of one member's modelled warming: (t, v), in T_mod's
    # precision (itemsize is the bytes per number: 8 for float64)
    n_years, n_forcing_vars = T_mod.shape[-2:]
    bytes_per_member = n_years * n_forcing_vars * T_mod.itemsize
    # How many members fit in a block (at least one)
    members_per_block = max(1, bytes_per_block // bytes_per_member)
    # The member blocks, as slices along the member axis
    block_starts = range(0, n_members, members_per_block)
    member_blocks = [slice(start, min(start + members_per_block, n_members))
                     for start in block_starts]

    def rebuild_member_block(member_block):
        # This block's members: four index arrays
        block_index = tuple(index[member_block] for index in member_index)
        # Rebuild them, and write them into their columns: (t, variables,
        # members in the block)
        out[:, :, member_block] = rebuild(
            T_mod, coefs, T_obs, block_index, forcing_vars, regress_vars,
            inc_reg_const, output_vars)

    # Rebuild every member block; list() makes any error in a thread
    # surface here
    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(rebuild_member_block, member_blocks))
