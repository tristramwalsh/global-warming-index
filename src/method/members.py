"""Stage C: rebuilding each drawn member's attributed warming from stage
A's modelled warming, T_mod, and stage B's scaling factors, β_obs and β_pic.

The axes are named as everywhere in the engine, each with its length in
capitals:

    p  FaIR parameter set           P of them
    e  ERF ensemble member          E
    o  observed GMT member          O
    c  piControl segment            C
    t  year
    v  forcing variable (e.g. GHG, Nat, OHF, and any sub-variables, e.g. co2)
    k  regressor: one per regressed variable, plus one for the constant term
       if it is used
    n  member of the sample         N

A member is one choice of each source of uncertainty: one position along
each of the four axes. Positions along an axis appear in three forms:

    the whole axis   arrays from stages A and B hold every position: T_mod
                     (p, e, t, v) has all P parameter sets
    per member       p, an integer array of length N: p[n] is member n's
                     parameter set; each position appears many times, in
                     the pattern of the sampling design (see sampling.cube)
    one member's     p_n = p[n], a single integer: member n's parameter set

and likewise e, o and c. The four arrays together are the member index,
(p, e, o, c), and member n is (p_n, e_n, o_n, c_n). In a shape, a letter
names the axis, e.g. T_mod (p, e, t, v); in code, it is the per-member
array. numpy's indexing reads the same for one member as for all of them:
T_mod[p_n, e_n] is member n's modelled warming, (t, v), and T_mod[p, e] is
every member's at once, (N, t, v), its n-th being T_mod[p[n], e[n]].

Member n is rebuilt from what stages A and B made:

    β_n          = β_obs[p_n, e_n, o_n] + β_pic[p_n, e_n, c_n]
                                          its k scaling factors
    T_att,n(t, v) = β_n[k(v)] · T_mod[p_n, e_n, t, v]
                                          for every forcing variable v

where k(v) is the regressor that scales v: v itself if it is regressed, or
the regressed variable it is part of (e.g. co2 is scaled by GHG's factor).
Then the diagnosed variables:

    Tot_n(t) = Σ over the regressed variables of T_att,n(t, v)
               (+ the constant)
    Res_n(t) = T_obs[t, o_n] − Tot_n(t)
               (NaN in years without observations)
    Ant_n(t) = T_att,n(t, GHG) + T_att,n(t, OHF)

Tot and Ant are written only when they are diagnosed, i.e. above the
regressed variables (see variables.extra_vars): when Tot is itself
regressed, its attributed warming is its own T_att, without the constant.

Three lists of variables, each a list of names:

    forcing_vars  every variable FaIR runs on its own in stage A (the v axis
                  of T_mod): the regressed variables and, with
                  --include-sub-vars, their sub-variables (e.g. co2, ch4)
    regress_vars  the forcing variables whose modelled warming is regressed
                  onto the observations in stage B (the k axis, before the
                  constant), e.g. GHG, Nat, OHF; every other forcing
                  variable takes the factor of the regressed variable it is
                  part of
    output_vars   the variables of each member's attributed warming, and so
                  of the outputs: the forcing variables, then the diagnosed
                  ones (Ant, Tot, Res) that aren't among them

`scaling_factors` gives each member's β; `rebuild` rebuilds some members;
`rebuild_into` rebuilds many, in blocks of members over threads; `Sample`
holds T_mod, β, T_obs and the member index for the whole sample, and
rebuilds every member for whichever years an output asks for.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import src.method.attribution as attribution
import src.method.variables as variables

# The precision of every member's attributed warming. float32 halves the
# memory against float64; its precision, about 7 significant digits (~1e-7
# °C on warming of order 1 °C), is far finer than any output reports.
MEMBER_DTYPE = np.float32


def scaling_factors(coefs, member_index):
    """Each member's scaling factors: member n's are
    β_n = β_obs[p_n, e_n, o_n] + β_pic[p_n, e_n, c_n].

    coefs: regression.Coefficients from stage B: obs (p, e, o, k) and pic
        (p, e, c, k)
    member_index: (p, e, o, c), four integer arrays of length N (see the
        module's docstring)
    Returns an array (N, k), float64.
    """
    # Each member's position along each axis: four (N,)
    p, e, o, c = member_index
    # Each member's fit to its observed member: (N, k)
    beta_obs = coefs.obs[p, e, o]
    # Each member's fit to its piControl segment: (N, k)
    beta_pic = coefs.pic[p, e, c]
    # Each member's scaling factors, the sum of its two fits: (N, k)
    return beta_obs + beta_pic


def rebuild(T_mod, beta, T_obs, member_index, forcing_vars, regress_vars,
            inc_reg_const, output_vars):
    """Rebuild members' attributed warming from their modelled warming,
    T_mod, and their scaling factors, β (and their observed members, T_obs,
    for the residual).

    T_mod: array (p, e, t, v), the modelled warming from stage A, relative to
        the pre-industrial period if used, over the years to rebuild, for
        the whole p and e axes
    beta: array (N, k), the members' scaling factors (see scaling_factors)
    T_obs: array (t, o), the observed GMT members over the same years, for
        the whole o axis, NaN in years without observations
    member_index: (p, e, o, c), four integer arrays of length N (see the
        module's docstring)
    forcing_vars, regress_vars, output_vars: the three lists of variables
        (see the module's docstring); regress_vars in the order of the k
        axis, the constant (if used) being the last regressor
    inc_reg_const: whether the regression has a constant term
    Returns an array (t, variables, N) in MEMBER_DTYPE: every member's
    attributed warming, its variables along output_vars.
    """
    # Each member's position along each axis: four (N,); c isn't needed
    # here, as the piControl fit is already in beta
    p, e, o, _ = member_index
    # Each member's modelled warming, member n's being T_mod[p[n], e[n]]:
    # (N, t, v)
    T_member = T_mod[p, e]

    # The regressor that scales each forcing variable: its own name if it is
    # regressed, or the regressed variable it is part of
    regressed_ancestor = variables.map_var_to_regression_aggregate(
        forcing_vars, regress_vars)
    # ... as positions on the k axis, one per forcing variable: (v,)
    k_of_v = [regress_vars.index(regressed_ancestor[var])
              for var in forcing_vars]
    # Each member's factor for each forcing variable, with a year axis of
    # length 1 to broadcast along: (N, 1, v)
    beta_of_v = beta[:, np.newaxis, k_of_v]
    # Every forcing variable's attributed warming: (N, t, v) * (N, 1, v)
    T_att = T_member * beta_of_v

    # Where the regressed variables sit on the v axis
    v_regressed = [forcing_vars.index(var) for var in regress_vars]
    # The total fitted warming, the sum of the regressed variables: (N, t)
    total = T_att[:, :, v_regressed].sum(axis=-1)
    if inc_reg_const:
        # Plus each member's constant term, the last regressor: (N, 1)
        total += beta[:, -1:]

    # Every member's attributed warming: (t, variables, N), filled one
    # variable at a time
    n_members, n_years = T_member.shape[:2]
    members_warming = np.empty((n_years, len(output_vars), n_members),
                               dtype=MEMBER_DTYPE)
    for v, var in enumerate(forcing_vars):
        # This forcing variable's attributed warming: (N, t)
        T_att_var = T_att[:, :, v]
        # ... with the member axis last, as in members_warming: (t, N)
        members_warming[:, output_vars.index(var), :] = T_att_var.T
    # The variables diagnosed after the regression
    diagnosed = variables.extra_vars(regress_vars)
    if 'Tot' in diagnosed:
        # The total, when it is above the regressed variables
        members_warming[:, output_vars.index('Tot'), :] = total.T
    if 'Ant' in diagnosed:
        # Anthropogenic warming: greenhouse gases plus other human forcings
        members_warming[:, output_vars.index('Ant'), :] = (
            T_att[:, :, forcing_vars.index('GHG')] +
            T_att[:, :, forcing_vars.index('OHF')]).T
    # Each member's observed member, member n's being T_obs[:, o[n]]:
    # (t, N)
    T_obs_member = T_obs[:, o]
    # The residual, observations minus the total: (t, N), NaN where there
    # are no observations
    members_warming[:, output_vars.index('Res'), :] = T_obs_member - total.T
    return members_warming


def rebuild_into(members_warming, T_mod, beta, T_obs, member_index,
                 forcing_vars, regress_vars, inc_reg_const, output_vars,
                 n_threads=1, bytes_per_block=2**21):
    """Rebuild every member into `members_warming`, in blocks of members,
    over threads.

    members_warming: array (t, variables, N), to fill with every member's
        attributed warming; member n is written to members_warming[:, :, n]
    T_mod, beta, T_obs, member_index, forcing_vars, regress_vars,
        inc_reg_const, output_vars: as for `rebuild`, for all N members
    n_threads: how many threads rebuild member blocks at once (e.g. the
        job's CPUs)
    bytes_per_block: the memory of one member block's modelled warming
        (every year and forcing variable of its members, in T_mod's
        precision), which sets how many members each block holds; default
        2 MiB

    A member block is a run of consecutive members,
    members_warming[:, :, start:stop]: a slice along the member axis only,
    holding every year and variable. Each thread rebuilds one member block
    at a time, with `rebuild`, and writes it straight into its positions
    along the member axis.
    While it works it holds about 3.5 × bytes_per_block (the block's
    modelled warming, its attributed warming, and their output), so the
    rebuild needs about 3.5 × n_threads × bytes_per_block on top of
    `members_warming`: about 0.2 GiB on 28 threads. The block size doesn't
    depend on the memory or the CPUs available. Measured on 28 threads (on
    shared nodes, so ±15%): blocks of 1 to 16 MiB are within ~20% of each
    other, and which is fastest shifts with the number of members; 2 MiB
    was within ~10% of the fastest both at 0.9 and at 6 million members.
    Blocks of 32 MiB and more are slower (32 MiB took 20–30% longer) and
    need far more memory; much smaller blocks, below ~1 MiB, spend more of
    their time in Python overhead.

    The results don't depend on the block size or on the number of threads.
    Threads, not processes: the rebuild is numpy array operations, which
    release Python's lock while they run, and every thread can write
    straight into `members_warming` without copying it.
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
        # This block's part of the member index: four arrays, each as long
        # as the block
        block_index = tuple(positions[member_block]
                            for positions in member_index)
        # Rebuild its members, and write them into their positions along
        # the member axis: (t, variables, members in the block)
        members_warming[:, :, member_block] = rebuild(
            T_mod, beta[member_block], T_obs, block_index, forcing_vars,
            regress_vars, inc_reg_const, output_vars)

    # Rebuild every member block; list() makes any error in a thread
    # surface here
    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(rebuild_member_block, member_blocks))


class Sample:
    """The sample: every member drawn from the full ensemble, ready to be
    rebuilt for any block of years.

    It holds what the members are rebuilt from (T_mod, β, T_obs and the
    member index), not the members: the members of every year at once, (t,
    variables, N), can need tens of GB, so each output rebuilds just the
    years it needs, with `attributed_warming` (see
    src/outputs/summaries.py).

    T_mod: array (p, e, t, v), stage A's modelled warming, over `years`
    coefs: regression.Coefficients from stage B: obs (p, e, o, k) and pic
        (p, e, c, k)
    T_obs: array (t, o), the observed GMT members over `years`, NaN in years
        without observations
    member_index: (p, e, o, c), four integer arrays of length N: the drawn
        members (see the module's docstring)
    forcing_vars, regress_vars: the forcing and regressed variables (see the
        module's docstring)
    inc_reg_const: whether the regression has a constant term
    years: the years of T_mod's and T_obs's t axis, consecutive (e.g. the
        forcing years); the members can be rebuilt for any of them
    n_threads: how many threads rebuild members at once (see rebuild_into)

    Each member's scaling factors (N × k numbers, about as much memory as
    one year of every member) are summed once, here, not for every block of
    years.
    """

    def __init__(self, T_mod, coefs, T_obs, member_index, forcing_vars,
                 regress_vars, inc_reg_const, years, n_threads=1):
        # What the members are rebuilt from, kept as given
        self.T_mod = T_mod
        self.T_obs = T_obs
        self.member_index = member_index
        self.forcing_vars = forcing_vars
        self.regress_vars = regress_vars
        self.inc_reg_const = inc_reg_const
        self.n_threads = n_threads
        # The years of the t axis, as an array: (t,)
        self.years = np.asarray(years)
        # The variables of the members' attributed warming: the forcing
        # variables, then the diagnosed ones (Ant, Tot, Res) not among them
        self.output_vars = attribution.attribution_output_vars(
            forcing_vars, regress_vars)
        # Each member's scaling factors: (N, k)
        self.beta = scaling_factors(coefs, member_index)

    @property
    def n_members(self):
        """N, the number of members in the sample."""
        return len(self.member_index[0])

    @property
    def nbytes(self):
        """The memory the sample holds: T_mod, T_obs, the scaling factors
        and the member index."""
        return (self.T_mod.nbytes + self.T_obs.nbytes + self.beta.nbytes +
                sum(positions.nbytes for positions in self.member_index))

    @property
    def bytes_per_year(self):
        """The memory of one year of every member's attributed warming:
        variables × N numbers in MEMBER_DTYPE."""
        return (len(self.output_vars) * self.n_members *
                np.dtype(MEMBER_DTYPE).itemsize)

    def year_positions(self, first_year, last_year):
        """Where the years first_year to last_year (both included) sit
        along the t axis, as a slice; an error if any is outside
        self.years."""
        # Stop if the years asked for are outside the sample's years
        if first_year < self.years[0] or last_year > self.years[-1]:
            raise ValueError(
                f'Years {first_year}-{last_year} are outside the sample\'s '
                f'years, {self.years[0]}-{self.years[-1]}.')
        # slice(a, b) is what a:b means inside brackets: x[slice(a, b)] is
        # x[a:b], positions a to b - 1. Here a is first_year's position and
        # b one past last_year's
        return slice(first_year - self.years[0],
                     last_year - self.years[0] + 1)

    def attributed_warming(self, first_year, last_year, into=None):
        """Every member's attributed warming, rebuilt for the years
        first_year to last_year (both included) only.

        first_year, last_year: the years an output needs, within self.years:
            e.g. one block of the output years for the timeseries (see
            summaries.years_per_block), or every output year at once
        into: array (t, variables, N) in MEMBER_DTYPE to write it into;
            made if not given
        Returns that array: (t, variables, N), t the years asked for, the
        variables along output_vars.

        Only the years asked for are rebuilt. The rebuild treats each year
        on its own (a member's β is the same in every year), so rebuilding
        some years gives exactly what rebuilding every year and then
        selecting those would, without the memory for the rest.
        """
        # The years asked for, as positions along the t axis
        years_asked = self.year_positions(first_year, last_year)
        # The modelled warming over those years: a view, (p, e, t, v)
        T_mod_years = self.T_mod[:, :, years_asked, :]
        # The observed members over those years: a view, (t, o)
        T_obs_years = self.T_obs[years_asked, :]
        if into is None:
            # The array to fill: (t, variables, N)
            into = np.empty((last_year - first_year + 1,
                             len(self.output_vars), self.n_members),
                            dtype=MEMBER_DTYPE)
        # Rebuild every member over those years into it
        rebuild_into(into, T_mod_years, self.beta, T_obs_years,
                     self.member_index, self.forcing_vars, self.regress_vars,
                     self.inc_reg_const, self.output_vars,
                     n_threads=self.n_threads)
        return into
