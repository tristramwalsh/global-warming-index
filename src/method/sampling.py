"""Sampling designs: which ensemble members to rebuild.

The axes are named as everywhere in the engine:

    p  FaIR parameter set           P of them
    e  ERF ensemble member          E
    o  observed GMT member          O
    c  piControl segment            C

A member is one choice of each source of uncertainty: one position along
each axis. A design returns the members to rebuild as their member index:
four integer arrays of the same length N, (p, e, o, c), member n being
(p[n], e[n], o[n], c[n]); n is its place along the member axis (see
src/method/members.py).
"""

import numpy as np


def cube(params, erf, obs, pic):
    """Every combination of the members given: their grid.

    params, erf, obs, pic: the members to combine from each source, as
        integer positions along its axis (p, e, o and c), in the order
        wanted: e.g. np.arange(P) for every parameter set, or a random draw
        of positions for the ERF members
    Returns the member index, (p, e, o, c): four integer arrays of length
    N = len(params) · len(erf) · len(obs) · len(pic), positions along the
    whole axes, member n being (p[n], e[n], o[n], c[n]).

    The members are ordered with the parameter sets varying slowest and the
    piControl segments fastest, each source in the order given:

        n = ((i · E + j) · O + l) · C + m,
        member n = (params[i], erf[j], obs[l], pic[m])

    where here E, O and C are the lengths of erf, obs and pic (the drawn
    members, not the whole axes). So the member axis holds one block per
    parameter set, one sub-block per ERF member within it, and so on.

    The legacy "cube" design draws a random subset of the ERF members, the
    observed members and the piControl segments (each up to --samples
    members), keeps every parameter set, and takes this grid of them. The
    random draws are made before this, in gwi.py.
    """
    # Each member's four positions, on the grid: four arrays
    # (len(params), len(erf), len(obs), len(pic))
    grid = np.meshgrid(np.asarray(params), np.asarray(erf),
                       np.asarray(obs), np.asarray(pic), indexing='ij')
    # Flattened in C order, so pic varies fastest and params slowest: four
    # (N,)
    p, e, o, c = (positions.ravel() for positions in grid)
    return p, e, o, c
