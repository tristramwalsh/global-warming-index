"""Sampling designs: which members of the complete ensemble to rebuild.

The axes are named as everywhere in the engine:

    p  FaIR parameter set           P of them
    e  ERF ensemble member          E
    o  observed GMT member          O
    c  piControl segment            C

A member is one choice of each source of uncertainty: one position along
each axis. The complete ensemble is every combination, P · E · O · C
members, all equally weighted. A design chooses N of them to rebuild, and
returns them as their member index: four integer arrays of the same length
N, (p, e, o, c), member n being (p[n], e[n], o[n], c[n]); n is its place
along the member axis (see src/method/members.py).

The designs (DESIGNS), each treating the four sources alike:

    cube   every combination of a random subset of each source, k members
           of each (or all of a source with fewer than k)
    mc     N members drawn independently: each member's position along
           each axis drawn uniformly, with replacement (Monte Carlo)
    latin  N members in which every position along each axis is used
           equally often (to within one), the axes paired at random (a
           Latin hypercube over the four axes), and no member is used
           twice until every member has been used once: the complete
           ensemble when N is its size, and beyond it, every member
           ⌊N / M⌋ or ⌈N / M⌉ times
    all    every member of the complete ensemble: no sampling error, and
           the same members every time

Each takes the axis lengths (P, E, O, C), the ensemble size N asked for,
and a numpy Generator (np.random.default_rng(seed)), the run's only source
of random numbers; `all` uses neither N nor the Generator.

Positions are stored in the smallest unsigned integer type that holds the
axis's last position (uint8 up to 255, uint16 up to 65,535): 1 or 2 bytes
per member rather than 8. The positions are exact in any integer type, so
only the memory changes, but do arithmetic on them in a wider type: for the
four arrays, 185 MiB → 29 MiB at 6 million members, and 25 GB → 4 GB for
the complete ensemble (790 million members; 28 parameter sets, 1000 ERF
members, 200 observed members, 141 piControl segments: 5 bytes per member).
"""

import numpy as np


def position_type(n_axis):
    """The smallest unsigned integer type that holds every position along
    an axis of n_axis members, 0 to n_axis - 1.

    n_axis:
        the axis's length
    """
    return np.min_scalar_type(n_axis - 1)


def grid(axis_positions, axis_lengths):
    """Every combination of the positions given: their grid.

    axis_positions:
        (params, erf, obs, pic), the positions to combine along each axis
        (p, e, o and c), in the order wanted: e.g. np.arange(P) for every
        parameter set
    axis_lengths:
        (P, E, O, C), the whole axes' lengths, for the positions' type
    Returns the member index, (p, e, o, c):
        four arrays of length N = len(params) · len(erf) · len(obs) ·
        len(pic), positions along the whole axes, each in its axis's
        position_type

    Any number of axes works the same way (e.g. two, in
    tests/plot_latin_sampling.py).

    The members are ordered with the parameter sets varying slowest and the
    piControl segments fastest, each axis in the order given:

        n = ((i · E' + j) · O' + l) · C' + m,
        member n = (params[i], erf[j], obs[l], pic[m])

    where E', O' and C' are the lengths of erf, obs and pic (the positions
    given, not the whole axes). So the member axis holds one block per
    parameter set, one sub-block per ERF member within it, and so on.
    """
    # The positions along each axis, in its compact type: four 1-D arrays
    positions = [np.asarray(drawn).astype(position_type(n_axis))
                 for drawn, n_axis in zip(axis_positions, axis_lengths)]
    # The grid's shape: (len(params), len(erf), len(obs), len(pic))
    shape = tuple(len(axis_positions) for axis_positions in positions)
    # Each axis's positions, broadcast along the grid's other axes and
    # flattened in C order, so pic varies fastest and params slowest. The
    # broadcast makes no copy, and ravel copies in the compact type, so no
    # int64 grid is ever made (25 GB for the complete ensemble): four (N,)
    member_index = []
    for axis, axis_positions in enumerate(positions):
        # This axis's positions as a 4-D array, long along this axis only:
        # e.g. (len(erf), 1, 1) after one leading 1 for erf
        along_axis = axis_positions.reshape(
            [-1 if other == axis else 1 for other in range(len(shape))])
        member_index.append(np.broadcast_to(along_axis, shape).ravel())
    return tuple(member_index)


def cube_samples_per_axis(ensemble_size, axis_lengths):
    """How many members of each source the cube design draws for an
    ensemble size: the largest k whose cube, with min(k, n) members along
    an axis of n, has no more than ensemble_size members.

    ensemble_size:
        N, the members asked for
    axis_lengths:
        (P, E, O, C)
    Returns k:
        at least 1; the longest axis's length if N reaches the complete
        ensemble, so that the cube is then every member

    The cube's size steps from k⁴ to (k + 1)⁴ (while every axis is longer
    than k), so the cube can fall short of N, by up to ~20% at N ~ 1e5.
    E.g. with P=28, E=1000, O=200, C=141: N = 1e4 gives k = 10 (10,000
    members), 1e5 gives 17 (83,521), 1e6 gives 32 (28 · 32³ = 917,504).
    """
    # The cube's size for k members of each source
    def cube_size(k):
        return int(np.prod([min(k, n_axis) for n_axis in axis_lengths],
                           dtype=np.int64))
    # The largest k whose cube fits, counting up from 1 (k is at most the
    # longest axis, ~1000, so counting is quick)
    k = 1
    while k < max(axis_lengths) and cube_size(k + 1) <= ensemble_size:
        k += 1
    return k


def cube(axis_lengths, ensemble_size, rng):
    """The cube design: every combination of a random subset of each source.

    axis_lengths:
        (P, E, O, C)
    ensemble_size:
        N, the members asked for; the cube has the largest k members of
        each source (all of a source with fewer) that keeps it within N
        (see cube_samples_per_axis), so it can have fewer
    rng:
        the run's numpy Generator
    Returns the member index, (p, e, o, c), four (N',) arrays, N' ≤ N (see
    grid)

    Each source's k members are drawn without replacement, in the order
    p, e, o, c, each in the random order drawn. Before Phase 2 the cube kept
    every parameter set and drew the other sources from numpy's global
    random state; those draws can't be reproduced with this design.
    """
    # The members of each source to draw
    k = cube_samples_per_axis(ensemble_size, axis_lengths)
    # A random subset of each axis, without replacement: four 1-D arrays
    drawn = [rng.choice(n_axis, min(k, n_axis), replace=False)
             for n_axis in axis_lengths]
    # Every combination of them
    return grid(drawn, axis_lengths)


def mc(axis_lengths, ensemble_size, rng):
    """The Monte Carlo design: N members drawn independently and uniformly
    from the complete ensemble.

    axis_lengths:
        (P, E, O, C)
    ensemble_size:
        N, the members to draw
    rng:
        the run's numpy Generator
    Returns the member index, (p, e, o, c), four (N,) arrays

    Each member's position along each axis is drawn uniformly, with
    replacement, independently of its other positions and of the other
    members, which is a uniform draw from the complete ensemble. So a
    member can be drawn more than once (rarely, when N is far below the
    complete ensemble's size).
    """
    # Each axis's N positions, drawn uniformly with replacement: four (N,)
    return tuple(rng.integers(n_axis, size=ensemble_size,
                              dtype=position_type(n_axis))
                 for n_axis in axis_lengths)


def balanced_positions(n_axis, ensemble_size, rng):
    """N positions along one axis, every position used as equally often as
    N allows, in a random order.

    n_axis:
        n, the axis's length (e.g. 1000 ERF members)
    ensemble_size:
        N, how many positions to give (one for each member of the sample)
    rng:
        the run's numpy Generator
    Returns an (N,) array of positions, in the axis's position_type.

    In plain words: deal the axis's positions out like cards from a deck in
    which each position appears equally often, then shuffle the deck.

    A small example: an axis of n = 5 positions, and N = 12 members.
      1. 12 / 5 = 2, remainder 2: every position is used twice (10 uses),
         and the 2 uses left over go to 2 positions drawn at random, say 0
         and 3. The counts are then [3, 2, 2, 3, 2].
      2. Each position written out as many times as its count:
         [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4].
      3. Shuffled: e.g. [3, 1, 0, 4, 2, 3, 0, 1, 3, 4, 2, 0]. Member j of
         the sample takes the j-th of these.

    So every position is used ⌊N / n⌋ or ⌈N / n⌉ times: here 2 or 3. With
    N < n, there are only leftover uses, so a random N of the positions are
    used once each. The positions drawn for the extra use are random, not
    spread along the axis, since the positions are labels (ERF member
    numbers, models, ...) whose order means nothing.

    latin calls this once for each axis, each with its own shuffle; placing
    the shuffled axes side by side pairs them at random (see latin).
    """
    # 1. How often each position is used: every position the same whole
    # number of times, ⌊N / n⌋, and the uses left over, N mod n, given one
    # each to positions drawn at random: (n,)
    uses_each, uses_left_over = divmod(ensemble_size, n_axis)
    counts = np.full(n_axis, uses_each)
    counts[rng.choice(n_axis, uses_left_over, replace=False)] += 1

    # 2. Each position written out as many times as its count, in order,
    # in the axis's compact type: (N,)
    positions = np.repeat(
        np.arange(n_axis, dtype=position_type(n_axis)), counts)

    # 3. Shuffled (in place), so member j takes a random one of them
    rng.shuffle(positions)
    return positions


# The most rounds of swaps remove_repeats makes before giving up. With the
# sample at most half the complete ensemble the repeats shrink by a factor of
# ~4 or more a round, so even 1e9 repeats need ~15; this many means
# something is wrong. Set from the 2026-11-24 tests (a toy ensemble of 3024
# members at N/M up to 0.5, and IGCC-2025 with HadCRUT-2025).
EMPIRICALLY_TUNED_MAX_REPAIR_ROUNDS = 200


def remove_repeats(positions, axis_lengths, rng):
    """Swap positions between members of a sample until no member appears
    twice, without changing how often each position is used.

    positions:
        the sample's positions, one row per axis and one column per member:
        (axes, N), changed in place; N must be no more than half the
        complete ensemble (latin makes sure of that), so that most swaps
        succeed
    axis_lengths:
        (P, E, O, C), or as many axes as positions has rows
    rng:
        the run's numpy Generator

    In plain words: find the members that appear twice; swap one of their
    positions with another member's; keep each swap only if it makes two
    members not already in the sample; repeat until none appear twice.

    A small example with two axes of 3 positions each (a 3 × 3 grid of 9
    members) and a sample of 6, each column one member, (axis 1, axis 2):

        axis 1:  0  0  1  1  2  2
        axis 2:  1  2  0  0  2  1
        members (0,1) (0,2) (1,0) (1,0) (2,2) (2,1)

    Each axis uses each of its positions twice, as balanced as it can be,
    but (1,0) appears twice. Pair the second (1,0) with a random other
    member, say (2,2), and swap their positions along a random axis, say
    axis 2:

        axis 1:  0  0  1  1  2  2
        axis 2:  1  2  0  2  0  1
        members (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)

    Both members the swap made, (1,2) and (2,0), were not in the sample,
    so it is kept, and no member repeats now. A swap only moves positions
    between members, so each axis still uses each position twice: every
    count is unchanged.

    The general case does this for every repeat at once, in rounds:
      1. Give each member one number, its place in the complete ensemble,
         and count how often each member appears (sorting the sample once).
      2. Note the members that repeat one drawn before them.
      3. Each round:
         a. drop the members no longer repeating;
         b. give each a random partner: a member that isn't itself a
            repeat, partnered once at most, so no member is in two swaps;
         c. pick a random axis for each pair;
         d. work out the two members each swap would make;
         e. keep the swaps whose two new members are both unused, and made
            by no other swap this round, so a round never adds a repeat;
         f. make those swaps, and update the counts.

    With the sample at most half the complete ensemble, each new member is
    unused at least half the time, so at least ~1/4 of swaps are kept, and
    the repeats shrink geometrically: e.g. ~10 rounds for the ~6 million
    repeats among 1e8 members drawn from 790 million. The counts are kept
    in one byte for every member of the complete ensemble (790 MB for
    IGCC-2025 with HadCRUT-2025), of which only the parts written to take
    memory: little at small N. Every step treats the positions along an
    axis alike, so the sample is still equally likely to hold any member.
    """
    n_axes, n_members = positions.shape
    n_complete = int(np.prod(axis_lengths, dtype=np.int64))
    # Each member's column, numbered 0 to N - 1: (N,)
    columns = np.arange(n_members)

    # 1. Each member as one number, its place in the complete ensemble's
    # grid (int64, as that has up to ~1e9 members): (N,)
    ids = np.ravel_multi_index(positions, axis_lengths)
    # ... and how often each member appears in the sample, one count for
    # every member of the complete ensemble (sorting the sample once, in
    # np.unique): (M,)
    used_ids, first_columns, uses = np.unique(
        ids, return_index=True, return_counts=True)
    times_drawn = np.zeros(n_complete, dtype=np.uint8)
    times_drawn[used_ids] = uses

    # 2. The members repeating one drawn before them: every column except
    # the first to hold each member: (R,)
    is_first = np.zeros(n_members, dtype=bool)
    is_first[first_columns] = True
    repeats = columns[~is_first]
    del used_ids, first_columns, uses, is_first

    # 3. Rounds of swaps, until no member repeats
    for _ in range(EMPIRICALLY_TUNED_MAX_REPAIR_ROUNDS):
        # a. The members still repeating: a member stops when a swap
        # changes it, or moves the other copies of it away: (R,)
        repeats = repeats[times_drawn[ids[repeats]] > 1]
        if len(repeats) == 0:
            return

        # b. A random partner for each: drawn from every member, then kept
        # only if it isn't a repeat itself, and only the first time it is
        # drawn, so no member is in two swaps: (S,)
        partners = rng.integers(n_members, size=len(repeats))
        first_time = np.zeros(len(partners), dtype=bool)
        first_time[np.unique(partners, return_index=True)[1]] = True
        usable = first_time & ~np.isin(partners, repeats)
        movers, partners = repeats[usable], partners[usable]

        # c. The axis along which each pair swaps positions: (S,)
        axes = rng.integers(n_axes, size=len(movers))
        pairs = np.arange(len(movers))

        # d. The two members each swap would make: the repeating member
        # with its partner's position along that axis, and the partner with
        # the repeating member's: (axes, S) each, then one number each: (S,)
        mover_after = positions[:, movers]
        partner_after = positions[:, partners]
        mover_after[axes, pairs] = positions[axes, partners]
        partner_after[axes, pairs] = positions[axes, movers]
        mover_after_ids = np.ravel_multi_index(mover_after, axis_lengths)
        partner_after_ids = np.ravel_multi_index(partner_after, axis_lengths)

        # e. The swaps to keep: both new members unused in the sample, and
        # each made by this swap alone (not by another swap this round, nor
        # by both sides of this one): (S,)
        new_ids = np.concatenate([mover_after_ids, partner_after_ids])
        made_ids, times_made = np.unique(new_ids, return_counts=True)
        is_new = ((times_drawn[new_ids] == 0)
                  & np.isin(new_ids, made_ids[times_made == 1]))
        keep = is_new[:len(movers)] & is_new[len(movers):]
        movers, partners, axes = movers[keep], partners[keep], axes[keep]

        # f. Make the kept swaps: the members given up are drawn once less
        # (np.subtract.at, as both sides of two swaps can give up the same
        # member), the members made are drawn once ...
        np.subtract.at(times_drawn,
                       np.concatenate([ids[movers], ids[partners]]), 1)
        times_drawn[mover_after_ids[keep]] = 1
        times_drawn[partner_after_ids[keep]] = 1
        ids[movers] = mover_after_ids[keep]
        ids[partners] = partner_after_ids[keep]
        # ... and the positions swapped along each pair's axis
        positions[axes, movers], positions[axes, partners] = (
            positions[axes, partners], positions[axes, movers])

    raise RuntimeError(
        f'The Latin design still repeats members after '
        f'{EMPIRICALLY_TUNED_MAX_REPAIR_ROUNDS} rounds of swaps.')


def latin(axis_lengths, ensemble_size, rng, replace=False):
    """The Latin hypercube design: N members in which every member of each
    source is used equally often, the sources paired at random, and no
    member drawn twice until every member has been drawn.

    axis_lengths:
        (P, E, O, C)
    ensemble_size:
        N, the members to draw
    rng:
        the run's numpy Generator
    replace:
        True to let members repeat (the sources balanced and paired at
        random, nothing more), for comparisons; the design `latin` is
        False
    Returns the member index, (p, e, o, c), four (N,) arrays

    In plain words, with M members in the complete ensemble:
      1. Take whole copies of the complete ensemble while N allows: ⌊N / M⌋
         of them, usually none.
      2. Draw the rest, R = N mod M members, without repeats:
         - if R is no more than half the complete ensemble, draw the R
           members to include;
         - if it is more than half, draw the M − R members to leave out,
           and take every other member, which is quicker (fewer members to
           draw) and needs no new method (see below).
      3. Draw those members so: for each source on its own, deal out its
         members equally often in a random order (balanced_positions);
         place the sources side by side, which pairs them at random; then
         swap positions between members until none appears twice
         (remove_repeats).

    A small example with two sources of 4 and 3 members (12 members in all,
    a 4 × 3 grid) and N = 6: R = 6 is half of 12, so the 6 members are
    drawn directly. Source 1's members are dealt out 6 / 4 = 1 or 2 times
    each, source 2's 6 / 3 = 2 times each, each shuffled, then placed side
    by side; any repeated member is swapped away. With N = 9, R = 9 is
    more than half, so 3 members are drawn to leave out, and the other 9
    taken. With N = 12, the sample is the complete ensemble; with N = 15,
    the complete ensemble and 3 more members.

    Why this design:
    - The balance (every member of each source used ⌊N / n⌋ or ⌈N / n⌉
      times) removes the sampling variance that comes from each source on
      its own (its main effect), which is most of it: on a toy ensemble,
      the error in the mean was ~3× smaller than Monte Carlo's at the same
      N.
    - Drawing without repeats lowers the variance left after the balancing
      by the share of the complete ensemble drawn, 1 − N/M: ~0.1% at
      N = 1e6, ~1% at 1e7 and ~12% at 1e8 of the 790 million members of
      IGCC-2025 with HadCRUT-2025, and all of it at N = M, where the
      sample *is* the complete ensemble (the same as the `all` design).
      Paired at random alone, the sources would repeat ~0.06% of members
      at 1e6 and ~6% at 1e8.
    - Leaving out M − R members gives the same kind of sample as drawing R:
      the complete ensemble uses every member of each source equally
      often, so taking away a balanced set leaves a balanced set, without
      repeats.

    Leaving out members needs a flag for every member of the complete
    ensemble (790 MB for IGCC-2025 with HadCRUT-2025), but only when more
    than half of it is drawn. Drawing took 0.8 s at N = 1e6, 3.6 s at 1e7
    and 68 s at 1e8 (IGCC-2025 with HadCRUT-2025, head node, 2026-11-24),
    peaking at ~5 GB at 1e8, before the members are rebuilt.
    """
    n_complete = int(np.prod(axis_lengths, dtype=np.int64))

    # With repeats allowed (for comparisons): each source balanced on its
    # own and the sources placed side by side, which pairs them at random
    if replace:
        return tuple(balanced_positions(n_axis, ensemble_size, rng)
                     for n_axis in axis_lengths)

    # Without repeats (the design):
    # 1. The whole copies of the complete ensemble, and the members left
    # to draw after them
    n_copies, n_rest = divmod(ensemble_size, n_complete)

    # 2. How many members to draw: those to include, or, if they are more
    # than half the complete ensemble, those to leave out
    leave_out = n_rest > n_complete - n_rest
    n_to_draw = n_complete - n_rest if leave_out else n_rest

    # 3. Draw them: each source balanced and shuffled on its own, then the
    # sources side by side, one row each, in one integer type wide enough
    # for every source: (axes, n_to_draw) ...
    positions = np.stack([
        balanced_positions(n_axis, n_to_draw, rng).astype(
            position_type(max(axis_lengths)))
        for n_axis in axis_lengths])
    # ... with any repeated member swapped away
    remove_repeats(positions, axis_lengths, rng)

    # 2, again. If those were the members to leave out, take every other
    # member instead: flag the ones left out, among all M, then list the
    # members not flagged, as positions along each axis: (axes, n_rest)
    if leave_out:
        is_left_out = np.zeros(n_complete, dtype=bool)
        is_left_out[np.ravel_multi_index(positions, axis_lengths)] = True
        positions = np.stack(np.unravel_index(np.flatnonzero(~is_left_out),
                                              axis_lengths))

    # Each source's positions in its own compact type: four (n_rest,)
    rest = [axis_positions.astype(position_type(n_axis))
            for axis_positions, n_axis in zip(positions, axis_lengths)]

    # 1, again. The copies of the complete ensemble, if any, then the rest
    if n_copies == 0:
        return tuple(rest)
    copies = all_members(axis_lengths)
    return tuple(np.concatenate([np.tile(copy, n_copies), axis_rest])
                 for copy, axis_rest in zip(copies, rest))


def all_members(axis_lengths, ensemble_size=None, rng=None):
    """The `all` design: every member of the complete ensemble.

    axis_lengths:
        (P, E, O, C)
    ensemble_size, rng:
        unused: the design has P · E · O · C members, the same every time
    Returns the member index, (p, e, o, c), four (P · E · O · C,) arrays,
    ordered as grid orders them
    """
    # Every position along every axis, combined
    return grid([np.arange(n_axis) for n_axis in axis_lengths],
                axis_lengths)


# How many members describe_coverage numbers at a time: 2**24 (~17
# million), so its temporary arrays (the members' numbers, and np.unique's
# sorted copies) stay at a few hundred MB however large the sample.
MEMBERS_PER_CHUNK = 2**24


def describe_coverage(member_index, axis_lengths):
    """How much of the complete ensemble a sample covers, and how much of
    it is drawn more than once.

    member_index:
        (p, e, o, c), four (N,) arrays (any design)
    axis_lengths:
        (P, E, O, C)
    Returns (n_different, n_drawn_more_than_once):
        n_different: how many different members the sample holds (its
            coverage of the complete ensemble)
        n_drawn_more_than_once: how many of those it holds twice or more

    In plain words: number every member by its place in the complete
    ensemble, and tick off each member of the sample on a list of all of
    them, counting how many times each is ticked.

    The tally has one byte for every member of the complete ensemble (790
    MB for IGCC-2025 with HadCRUT-2025), of which only the parts written
    to take memory: little for a small sample. The sample is numbered a
    chunk at a time (MEMBERS_PER_CHUNK), and the totals are kept up as the
    tally is, so the tally is never read in full.
    """
    n_members = len(member_index[0])
    n_complete = int(np.prod(axis_lengths, dtype=np.int64))
    # The tally: how many times each member of the complete ensemble is
    # drawn, up to 2 (all that matters here): (M,)
    times_drawn = np.zeros(n_complete, dtype=np.uint8)
    n_different, n_drawn_more_than_once = 0, 0

    for start in range(0, n_members, MEMBERS_PER_CHUNK):
        # This chunk's members, each as one number, and how many times
        # each different one appears in the chunk: (K,), (K,)
        chunk = slice(start, start + MEMBERS_PER_CHUNK)
        ids = np.ravel_multi_index(
            [positions[chunk] for positions in member_index], axis_lengths)
        chunk_ids, times_in_chunk = np.unique(ids, return_counts=True)

        # Their tallies before and after this chunk (stopping at 2)
        before = times_drawn[chunk_ids]
        after = np.minimum(before + times_in_chunk, 2).astype(np.uint8)
        times_drawn[chunk_ids] = after

        # The members drawn for the first time, and for a second time
        n_different += np.count_nonzero(before == 0)
        n_drawn_more_than_once += np.count_nonzero((before < 2)
                                                   & (after == 2))

    return n_different, n_drawn_more_than_once


# The designs, by the name --sampling-design takes
DESIGNS = {
    'cube': cube,
    'mc': mc,
    'latin': latin,
    'all': all_members,
}
