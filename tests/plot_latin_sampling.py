"""Picture the Latin design on a toy ensemble of two sources, 20 x 20.

    python tests/plot_latin_sampling.py [output.png]

The complete ensemble here is every combination of 20 members of one source
and 20 of another: 400 members, a 20 x 20 grid. The figure shows:

- top three rows: which members each design draws, at N = 40, 200, 400 and
  800 (0.1, 0.5, 1 and 2 times the complete ensemble), each cell coloured
  by how many times its member is drawn (white: never; then once, twice,
  three or more times):
    - latin, the design gwi.py runs (src/method/sampling.py): each source's
      members used equally often, and no member repeated until every member
      is used, so N = 400 is the complete ensemble;
    - latin with repeats allowed (sampling.latin(..., replace=True)): the
      same balanced sources, but paired at random, so members can repeat;
    - mc: every member drawn independently, with replacement;
  each panel's title gives the share of the complete ensemble drawn at
  least once, the members drawn more than once, and the range of each
  source's counts (how evenly each source's members are used);
- bottom row: those shares over N, averaged over 50 seeds.

Writes plots/tests/latin_sampling_2d.png unless given another path.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
import numpy as np  # noqa: E402

# The repository root, so that src is importable from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.method.sampling as sampling  # noqa: E402

# The toy ensemble: two sources of 20 members each
AXIS_LENGTHS = (20, 20)
N_COMPLETE = int(np.prod(AXIS_LENGTHS))
# The ensemble sizes pictured, as multiples of the complete ensemble
SIZES = [N_COMPLETE // 10, N_COMPLETE // 2, N_COMPLETE, 2 * N_COMPLETE]
# The designs: name, and a function (axis_lengths, N, rng) -> member index
DESIGNS = [
    ('latin (no repeats)', sampling.latin),
    ('latin, repeats allowed',
     lambda lengths, n, rng: sampling.latin(lengths, n, rng, replace=True)),
    ('mc (with replacement)', sampling.mc),
]
# The seeds averaged over in the bottom row
N_SEEDS = 50


def use_counts(member_index):
    """How many times each member of the toy ensemble is drawn: (20, 20)."""
    # Each member's place in the grid, then its count
    ids = np.ravel_multi_index(
        [positions.astype(np.int64) for positions in member_index],
        AXIS_LENGTHS)
    counts = np.bincount(ids, minlength=N_COMPLETE)
    return counts.reshape(AXIS_LENGTHS)


def describe(counts, member_index):
    """The panel's title: coverage, repeats, and each source's balance."""
    # The share of members drawn at least once, and of draws that repeat
    n_drawn = counts.sum()
    covered = np.count_nonzero(counts) / N_COMPLETE
    repeated = (n_drawn - np.count_nonzero(counts)) / n_drawn
    # Each source's counts: how often each of its members is used
    ranges = []
    for positions, n_axis in zip(member_index, AXIS_LENGTHS):
        source_counts = np.bincount(positions, minlength=n_axis)
        ranges.append(f'{source_counts.min()}-{source_counts.max()}')
    return (f'N={n_drawn}: {covered:.0%} covered, {repeated:.0%} repeats\n'
            f'source counts {ranges[0]} and {ranges[1]}')


def main(path):
    """Draw the figure and write it to path."""
    fig, axes = plt.subplots(len(DESIGNS) + 1, len(SIZES),
                             figsize=(3.2 * len(SIZES),
                                      3.4 * (len(DESIGNS) + 1)))
    # Colours by count: never, once, twice, three or more times
    colours = ListedColormap(['white', '#9ecae1', '#fd8d3c', '#bd0026'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 1e9], colours.N)

    # The top rows: one design per row, one ensemble size per column
    for row, (name, design) in enumerate(DESIGNS):
        for column, n_members in enumerate(SIZES):
            # The members drawn, with the same seed in every panel
            member_index = design(AXIS_LENGTHS, n_members,
                                  np.random.default_rng(1))
            counts = use_counts(member_index)
            ax = axes[row, column]
            ax.imshow(counts, cmap=colours, norm=norm, origin='lower')
            ax.set_title(describe(counts, member_index), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(name, fontsize=10)

    # The bottom row: coverage and repeats over N, averaged over seeds
    sizes = np.unique(np.linspace(1, 3 * N_COMPLETE, 120).astype(int))
    shares = {name: np.zeros((len(sizes), 2)) for name, _ in DESIGNS}
    for name, design in DESIGNS:
        for i, n_members in enumerate(sizes):
            for seed in range(N_SEEDS):
                counts = use_counts(design(AXIS_LENGTHS, n_members,
                                           np.random.default_rng(seed)))
                shares[name][i, 0] += np.count_nonzero(counts) / N_COMPLETE
                shares[name][i, 1] += (
                    (n_members - np.count_nonzero(counts)) / n_members)
        shares[name] /= N_SEEDS
    for column, (label, index) in enumerate(
            [('share of the complete ensemble drawn', 0),
             ('share of draws repeating a member', 1)]):
        ax = axes[-1, column]
        for name, _ in DESIGNS:
            ax.plot(sizes / N_COMPLETE, shares[name][:, index], label=name)
        ax.set_xlabel('N / complete ensemble size')
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1, 0].legend(fontsize=7)
    # A key to the colours, in the spare panels
    for ax in axes[-1, 2:]:
        ax.axis('off')
    for count, label in enumerate(['never', 'once', 'twice', '3+ times']):
        axes[-1, 2].scatter([], [], marker='s', s=80,
                            color=colours(count), edgecolor='grey',
                            label=label)
    axes[-1, 2].legend(title='member drawn', loc='center', fontsize=8)

    fig.suptitle('Sampling designs on a 20 x 20 toy ensemble (400 members)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else 'plots/tests/latin_sampling_2d.png')
