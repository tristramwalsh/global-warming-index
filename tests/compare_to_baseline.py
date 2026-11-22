"""Compare two results folders file by file, e.g. a check run against a baseline.

    python tests/compare_to_baseline.py results_engine-baseline results_engine-check

Run the same seeded configurations twice, on two versions of the code, into
two output tags (recipes and run-gwi.sh, as usual). This script then says
whether the two versions give the same results:

- Files are paired by their path inside each folder, ignoring the
  DATE-CALCULATED--<timestamp> part of the name, which differs on every run.
- Files present in only one folder are listed.
- Paired CSV files are compared cell by cell. Cells that are numbers must
  agree to within an absolute tolerance (--atol, default 1e-5, in each
  file's own units: °C for temperatures, W/m² for forcing); NaN only matches
  NaN. Every other cell (headers, labels) must match exactly.
- Other files are compared byte for byte.

Prints one line per file and a summary with the largest difference found.
Exits with status 0 if every file is present in both folders and agrees,
and 1 otherwise, so the script can also be used in other scripts.

Within one folder, two files must not share a path once the timestamp is
removed (e.g. two runs of the same configuration in one output tag): the
script cannot tell which to pair with which, so it reports them and stops.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# e.g. _DATE-CALCULATED--2026-11-22-14-03-59, before the file extension
DATE_PATTERN = re.compile(r'_DATE-CALCULATED--\d{4}(-\d{2}){5}')


def index_files(root):
    """Map each file's timestamp-free path inside root to its full path.

    Returns the map and a list of keys that more than one file maps to.
    """
    found = defaultdict(list)
    for path in sorted(Path(root).rglob('*')):
        if path.is_file():
            key = DATE_PATTERN.sub('', str(path.relative_to(root)))
            found[key].append(path)
    clashes = [key for key, paths in found.items() if len(paths) > 1]
    return {key: paths[0] for key, paths in found.items()}, clashes


def compare_csv(path_a, path_b, atol):
    """Compare two CSV files cell by cell.

    Returns (agrees, largest absolute numeric difference, note).
    """
    # Read every cell as text, with no header, so files of any layout (one
    # or several header rows, an index column or not) compare the same way.
    a = pd.read_csv(path_a, header=None, dtype=str, keep_default_na=False)
    b = pd.read_csv(path_b, header=None, dtype=str, keep_default_na=False)
    if a.shape != b.shape:
        return False, np.inf, f'shape {a.shape} vs {b.shape}'

    a_num = a.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    b_num = b.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    # Cells that are text in either file (labels, headers) must match
    # exactly. Empty cells and 'nan' are numbers here (NaN), not text.
    is_text = ((np.isnan(a_num) & ~a.isin(['', 'nan', 'NaN']).to_numpy()) |
               (np.isnan(b_num) & ~b.isin(['', 'nan', 'NaN']).to_numpy()))
    text_differs = is_text & (a.to_numpy() != b.to_numpy())
    if text_differs.any():
        row, col = np.argwhere(text_differs)[0]
        return False, np.inf, (
            f'{text_differs.sum()} text cells differ, first at row {row}, '
            f'column {col}: {a.iat[row, col]!r} vs {b.iat[row, col]!r}')

    numeric = ~is_text
    nan_a, nan_b = np.isnan(a_num) & numeric, np.isnan(b_num) & numeric
    if (nan_a != nan_b).any():
        return False, np.inf, f'{(nan_a != nan_b).sum()} cells are NaN in one file only'
    both = numeric & ~nan_a
    diff = np.abs(a_num[both] - b_num[both])
    largest = diff.max() if diff.size else 0.0
    return largest <= atol, largest, ''


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        epilog='See the module docstring for how files are paired and compared.')
    parser.add_argument('baseline', help='results folder to compare against')
    parser.add_argument('check', help='results folder to check')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='largest absolute difference allowed (default 1e-5)')
    args = parser.parse_args()

    files_a, clashes_a = index_files(args.baseline)
    files_b, clashes_b = index_files(args.check)
    for root, clashes in [(args.baseline, clashes_a), (args.check, clashes_b)]:
        for key in clashes:
            print(f'AMBIGUOUS  {root}: several files are {key} '
                  '(apart from their timestamps)')
    if clashes_a or clashes_b:
        print('Stopping: each folder must hold one file per configuration.')
        sys.exit(1)

    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    paired = sorted(set(files_a) & set(files_b))

    failures = 0
    largest_overall, largest_file = 0.0, None
    for key in paired:
        if key.endswith('.csv'):
            agrees, largest, note = compare_csv(files_a[key], files_b[key],
                                                args.atol)
        else:
            agrees = files_a[key].read_bytes() == files_b[key].read_bytes()
            largest, note = (0.0 if agrees else np.inf), 'binary comparison'
        if np.isfinite(largest) and largest > largest_overall:
            largest_overall, largest_file = largest, key
        if agrees:
            status = 'IDENTICAL' if largest == 0 else 'SAME'
        else:
            status = 'DIFFERENT'
            failures += 1
        print(f'{status:<10} max |diff| {largest:9.2e}  {key}'
              + (f'  ({note})' if note else ''))

    for key in only_a:
        print(f'MISSING    only in {args.baseline}: {key}')
    for key in only_b:
        print(f'EXTRA      only in {args.check}: {key}')

    print()
    print(f'{len(paired)} files paired: {len(paired) - failures} agree '
          f'(absolute tolerance {args.atol:g}), {failures} differ.')
    print(f'{len(only_a)} missing from {args.check}, '
          f'{len(only_b)} extra in {args.check}.')
    if largest_file is not None:
        print(f'Largest numeric difference: {largest_overall:.3e}, '
              f'in {largest_file}')
    sys.exit(0 if failures == 0 and not only_a and not only_b else 1)


if __name__ == '__main__':
    main()
