"""The release: a small set of members, representative of the run's
ensemble, for users to take whole.

The run's ensemble (the parent, N members, e.g. a million) is too big to
hand to anyone member by member, and a user drawing from it in their own
Monte Carlo would have to subsample it properly to ensure they are using a
representative subsample. We therefore provide small representative subsamples
here.

The attribution run releases M of its members (e.g. 1000), the same for every
data user, chosen so that their distribution has the parent's shape. Each
released member is one whole draw (p, e, o, c): all its variables, years and
definitions come from the same rebuild, so the joint structure (the covariance
of the variables, and the correlation between years) is the parent's, but for
the noise of M members.

The methods choose which M members to keep (the M positions in the parent's
member index); gwi.py then rebuilds just those members over every year and
definition, and distributions.py writes them.

    stratified  one member from each of M bands of equal probability of
                one variable in one year (e.g. Ant in 2025): that
                variable's distribution in that year is then as close to
                the parent's as M members can be
    random      M members at random, without repeats: a baseline, to
                measure what stratified gains; not for production

ONE IDEA, DELIVERED IN TWO STEPS

The stratified release rests on one idea: put the parent's members in
nested order, an order by the target (e.g. Ant in 2025) in which every
first k members, for any k, are spread evenly across its distribution.
The release is the parent's first M members in that order; a user who
takes the release's first k (e.g. 100 of 1000) takes the parent's first k.
Choosing M from the parent's N, and a user's k from the release's M, are
the same choice, made at two sizes.

It is delivered in two steps, one at each size, by two techniques that give
the same result (exactly, when N and M are powers of 2; otherwise step 1 is
the more even of the two):

    step 1, stratified selection (N to M; ours, inside the run): cut the
        parent's ranks into M bands of equal probability, and keep one
        member at random from each band
    step 2, nested order (M to k; for the user, in the file): order the
        M members by halving the bands again and again, interleaving the
        halves with the leading half chosen at random

These are the same thing: the first M members of the nested order of all N
are one member, at random, from each of M blocks of consecutive ranks, the
M bands of step 1. They need different techniques for two reasons:

- Cost. Putting all N members in nested order needs every member ranked
  (a full sort) and all N ordered: minutes for 6 million members in
  Python, and out of reach for the complete ensemble's 790 million. Only
  the first M are ever needed, and step 1 finds exactly those without
  ordering the rest (np.argpartition, at M ranks, in time linear in N).
  Ordering M = 1000 members (step 2) takes milliseconds.
- Evenness at any size. Halving a block whose size isn't a power of 2
  splits it unevenly (a block of 3 into 1 and 2, so its first member is one
  of three with chances of 1/2, 1/4 and 1/4), so the halving is a little
  uneven unless N is a power of 2. Step 1's bands are as even as N and M
  allow, each member of a band equally likely. Step 2 halves only M's
  bands, and leaves the choice within each band to step 1.

STEP 1, STRATIFIED SELECTION (N TO M), IN PLAIN WORDS

1. Rank the parent's N members by the stratifying value, e.g. each
   member's Ant in 2025, from the smallest (rank 0) to the largest
   (rank N - 1).

2. Cut the ranks into M bands of equal size, so of equal probability: band
   b holds ranks ⌊b N / M⌋ to ⌊(b + 1) N / M⌋ - 1.

3. Pick one rank in each band, uniformly at random, and keep the member
   that has it.

4. Put the members in nested order (step 2, below), so that the first k
   of them, for any k, are spread across the distribution too.

For example, 12 parent members into M = 4 bands. Their Ant in 2025, by
position in the member index:

    position  0    1    2    3    4    5    6    7    8    9    10   11
    Ant       1.31 1.02 1.45 1.18 1.27 1.39 1.10 1.22 1.50 1.35 1.15 1.25

Ranked, and cut into 4 bands of 3:

    band 0: 1.02 (1)  1.10 (6)  1.15 (10)      ranks 0-2
    band 1: 1.18 (3)  1.22 (7)  1.25 (11)      ranks 3-5
    band 2: 1.27 (4)  1.31 (0)  1.35 (9)       ranks 6-8
    band 3: 1.39 (5)  1.45 (2)  1.50 (8)       ranks 9-11

Picking ranks 1, 5, 6 and 10 at random, one in each band, releases the
members at positions 6, 11, 4 and 2 (Ant 1.10, 1.25, 1.27, 1.45): one from
each quarter of the distribution. A random choice of 4 could have taken
all four from the top half.

STEP 2, THE NESTED ORDER (M TO k), IN PLAIN WORDS

A user may take only the first k members of a release (e.g. 100 of 1000).
In band order (smallest value first) those would be the bottom tenth of
the distribution. So the members are put in an order in which every first
k is spread across all the bands:

1. Split the bands into a lower half and an upper half.

2. Order each half in the same way (steps 1 to 3, within the half).

3. Interleave the two halves' orders, one from each in turn, the half that
   goes first chosen at random.

The first 2 members are then one from each half, the first 4 one from each
quarter, the first 8 one from each eighth, and so on; which band in each
quarter (or eighth) comes first is random. For example, for M = 8 bands
(0 the lowest), one such order is

    5 1 6 3 4 0 7 2

The first 2 (5, 1) are one from each half; the first 4 (5, 1, 6, 3) one
from each quarter (0-1, 2-3, 4-5, 6-7). When M isn't a power of 2, the
halves differ by one band, and the spread of each first k is nearly as
even.

Why random: the plain van der Corput order (band b's place set by b's
binary digits reversed, 0 4 2 6 1 5 3 7 for 8 bands) always takes the
lowest band of each quarter first, so every first k leans low, by about
half of a k-th of the distribution. Choosing at random which half leads
removes the lean (Owen's scrambling of the van der Corput order, 1995),
and keeps the spread. On a skewed test distribution with standard
deviation 1, the first 100 of a 1000-member release (300 repeats): plain
order, mean error of the 5th to 95th percentiles -0.02 to -0.12 and RMS
error at the 5th 0.11; random-led order, -0.05 to +0.03 and 0.06; the
first 10, -0.1 to -0.9 against -0.25 to +0.24. (Band order: -0.9 to
-3.1 for the first 100.)

WHY THIS DESIGN

- Stratifying makes the release's error in that variable shrink like 1/M,
  not 1/√M as for a random choice. For 1000 members released from a
  million (RMSE against the exact answer over 200 repeats; plan §1.3):
  Ant 2025's 5th percentile 0.0004-0.0006 °C stratified against
  0.0055-0.0061 °C random, its median 0.0001 against 0.0028 °C.
- Variables correlated with the stratifying one gain too, less: Ant in
  2024 (correlation 0.993 with Ant in 2025) 0.0025 against 0.0056 °C.
  Others gain nothing, but lose nothing: GHG 2025's 5th percentile, 0.027
  against 0.028 °C. A release matched on several variables at once is
  planned (the matched method).
- The members are real draws, never averaged or reweighted, so every user
  can treat them as an equal-weight ensemble.

HOW IT IS DONE, AND WHAT IT COSTS

The ranks are never sorted in full: np.argpartition finds the members at
the M chosen ranks directly, as np.percentile finds its levels. It needs
the stratifying value of every parent member (4 bytes each, 3.2 GB for the
complete ensemble's 790 million) and their positions in rank order (8 bytes
each, made by argpartition), and takes about as long as the 1001-level
quantiles of one year: e.g. 0.3 s for 6 million members.

Rebuilding the chosen members over every year of every definition then
takes longer than choosing them, though little next to the run: 12.6 s for
1000 members from 6 million, every definition, on 28 CPUs, much of it
applying each definition to T_mod's windows (the same as for the
timeseries, whatever the number of members).
"""

import numpy as np


# The release methods, by their name in --release-method (see select)
METHODS = ('stratified', 'random')


def band_ranks(n_parent, n_release, rng):
    """One rank, uniformly at random, in each of n_release bands of equal
    size: steps 2 and 3 of stratified (see the module's docstring).

    n_parent:
        N, the parent's number of members (ranks 0 to N - 1)
    n_release:
        M, the number of bands, at most N
    rng:
        a numpy random Generator
    Returns the ranks, (M,) int64, one per band, in band order (so
    increasing).

    Band b holds ranks ⌊b N / M⌋ to ⌊(b + 1) N / M⌋ - 1: when M doesn't
    divide N, the bands differ in size by at most one member.
    """
    # Each band's first rank, and the next band's (one past its last):
    # (M + 1,)
    edges = (np.arange(n_release + 1, dtype=np.int64) * n_parent
             // n_release)
    # Each band's size: (M,)
    band_sizes = np.diff(edges)
    # A rank in each band, uniformly: its first rank plus a random offset
    # within its size: (M,)
    offsets = np.floor(rng.random(n_release) * band_sizes).astype(np.int64)
    return edges[:-1] + offsets


def stratified(stratify_values, n_release, rng):
    """The positions of n_release members, one from each band of equal
    probability of the stratifying value (see the module's docstring).

    stratify_values:
        array (N,), each parent member's stratifying value, e.g. its Ant
        in 2025, in the order of the member index
    n_release:
        M, the number of members to keep, at most N
    rng:
        a numpy random Generator
    Returns the positions, (M,) int64, in nested order (see nested_order):
    every first k of them spread across the bands.
    """
    # The rank to keep in each band: (M,)
    ranks = band_ranks(len(stratify_values), n_release, rng)
    # The parent's positions arranged so that those at the chosen ranks
    # are the members with those ranks (the rest only partly ordered): (N,)
    by_rank = np.argpartition(stratify_values, ranks)
    # The members at the chosen ranks, in band order: (M,)
    members = by_rank[ranks]
    # ... in nested order
    return members[nested_order(n_release, rng)]


def nested_order(n_bands, rng):
    """An order of the bands 0 to n_bands - 1 in which every first k of
    them is spread across all of them (see STEP 2 in the module's
    docstring).

    n_bands:
        M, the number of bands
    rng:
        a numpy random Generator
    Returns the bands, (M,) int64, in that order: e.g. 5 1 6 3 4 0 7 2 for
    8 bands.
    """
    def order(bands):
        # One band (or none) is in order already
        if len(bands) <= 1:
            return list(bands)
        # Each half of the bands, in order
        middle = len(bands) // 2
        first, second = order(bands[:middle]), order(bands[middle:])
        # The half that leads, at random
        if rng.random() < 0.5:
            first, second = second, first
        # One from each in turn (the longer half's last band at the end)
        interleaved = []
        for i in range(max(len(first), len(second))):
            interleaved += first[i:i + 1] + second[i:i + 1]
        return interleaved
    return np.array(order(list(range(n_bands))), dtype=np.int64)


def random(n_parent, n_release, rng):
    """The positions of n_release members chosen at random, without
    repeats: the baseline the stratified release is judged against.

    n_parent:
        N, the parent's number of members
    n_release:
        M, the number of members to keep, at most N
    rng:
        a numpy random Generator
    Returns the positions, (M,) int64, in the order drawn, so every first k
    of them is a random choice too.
    """
    return rng.choice(n_parent, size=n_release, replace=False)


def select(method, n_parent, n_release, rng, stratify_values=None):
    """The members of a release: their positions in the parent's member
    index.

    method:
        'stratified' or 'random' (see METHODS)
    n_parent:
        N, the parent's number of members
    n_release:
        M, the number to keep, from 1 to N
    rng:
        a numpy random Generator (see release_rng)
    stratify_values:
        array (N,), each parent member's stratifying value; needed for
        'stratified'
    Returns the positions, (M,) int64.
    """
    # Stop if the release is empty or bigger than its parent
    if not 1 <= n_release <= n_parent:
        raise ValueError(
            f'A release of {n_release:,} members from {n_parent:,}: a '
            'release has from 1 member to as many as the run has.')
    if method == 'stratified':
        return stratified(stratify_values, n_release, rng)
    if method == 'random':
        return random(n_parent, n_release, rng)
    raise ValueError(f'Unknown release method {method!r}: use one of '
                     f'{METHODS}.')


def release_rng(seed, n_release):
    """The random numbers of a release of n_release members: a generator
    of its own, from the run's seed and n_release.

    seed:
        the run's seed (see gwi.read_argv_config)
    n_release:
        M
    Returns a numpy random Generator.

    Its own, so the release draws nothing from the generator that drew the
    members (whose draws, and so every other output, stay as they are), and
    a release of M members is the same whether made alone or beside
    releases of other sizes. Reproducible from the seed, as the members
    are.
    """
    return np.random.default_rng([seed, n_release])
