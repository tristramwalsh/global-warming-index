"""Sampling designs: which ensemble members to rebuild.

The axes are named as everywhere in the engine:

    p  FaIR parameter set
    e  ERF ensemble member
    o  observed GMT member
    c  piControl segment

A member is one choice of each source of uncertainty, (p, e, o, c). A design
returns the members to rebuild as four integer index arrays of the same
length N: member n is (p[n], e[n], o[n], c[n]), and its position n is its
place along the results array's member axis.
"""

import numpy as np


def cube(params, erf, obs, pic):
    """Every combination of the members given: their grid.

    params, erf, obs, pic: the members to combine from each source, as
        integer positions along its axis (p, e, o and c), in the order
        wanted: e.g. np.arange(P) for every parameter set, or a random draw
        of positions for the ERF members
    Returns (p, e, o, c), four integer arrays of length
    N = len(params) · len(erf) · len(obs) · len(pic): member n is
    (p[n], e[n], o[n], c[n]), positions along the full axes.

    The members are ordered with the parameter sets varying slowest and the
    piControl segments fastest, each source in the order given:

        n = ((i · E + j) · O + l) · C + m,
        member n = (params[i], erf[j], obs[l], pic[m])

    where E, O and C are the lengths of erf, obs and pic. So the results
    array's member axis holds one block per parameter set, one sub-block per
    ERF member within it, and so on.

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
