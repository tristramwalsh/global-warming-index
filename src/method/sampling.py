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


def cube(n_params, n_erf, n_obs, n_pic):
    """Every combination of the members given: the full grid.

    n_params, n_erf, n_obs, n_pic: how many of each source there are (P, E,
        O and C), i.e. the lengths of the p, e, o and c axes
    Returns (p, e, o, c), four integer arrays of length N = P · E · O · C.

    The members are ordered with p varying slowest and c fastest:

        n = ((p · E + e) · O + o) · C + c

    so the results array's member axis holds one block per parameter set,
    one sub-block per ERF member within it, and so on.

    The legacy "cube" design draws a random subset of the ERF members, the
    observed members and the piControl segments (each up to --samples
    members), keeps every parameter set, and takes this grid of them. The
    random draws are made before this, in gwi.py.
    """
    # Each member's four indices, on the full grid: four arrays (P, E, O, C)
    grid = np.meshgrid(np.arange(n_params), np.arange(n_erf),
                       np.arange(n_obs), np.arange(n_pic), indexing='ij')
    # Flattened in C order, so c varies fastest and p slowest: four (N,)
    p, e, o, c = (index.ravel() for index in grid)
    return p, e, o, c
