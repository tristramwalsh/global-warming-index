"""Pin the optimised numerical routines to their reference implementations.

The vectorised forms in src/definitions.py replace np.polyfit with the
closed-form least-squares solution, which is mathematically identical but
orders the floating-point operations differently. They are therefore expected
to agree with the originals to rounding, not exactly.

percentile_threaded is different: chunking the leading axis does not change
any individual percentile, so it must agree EXACTLY, and is asserted as such.

This guards those equivalences: if any implementation is edited such that it
diverges, this fails.

Run from the repository root:

    python3 tests/test_trend_vectorisation.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np  # noqa: E402
import definitions as defs  # noqa: E402


# Tolerance on agreement with the per-series originals. The measured worst
# case is ~1.3e-07 for final_value_of_trend (float32 inputs, values of order
# 1) and ~8e-17 for rate_func; 1e-6 leaves room for platform variation without
# being loose enough to hide a real divergence.
TOL = 1e-6

failures = []


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def reference_final_value(array):
    """Apply the original per-series function down axis 0."""
    flat = array.reshape(array.shape[0], -1)
    out = np.array([defs.final_value_of_trend(flat[:, i])
                    for i in range(flat.shape[1])])
    return out.reshape(array.shape[1:])


def test_final_value_of_trend():
    print("final_value_of_trend_vectorised vs final_value_of_trend")
    rng = np.random.default_rng(0)

    # 3-D, as used for the attributed SR1.5 temps: (years, vars, ensemble).
    # 16 years is the SR1.5 window (year-15 .. year inclusive).
    a = rng.random((16, 6, 400)).astype(np.float32)
    got, ref = defs.final_value_of_trend_vectorised(a), reference_final_value(a)
    check(f"3-D shape {a.shape} -> {got.shape}", got.shape == ref.shape)
    check(f"3-D agrees within {TOL:g} "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < TOL)

    # 2-D, as used for the observational SR1.5 temps: (years, ensemble).
    b = rng.random((16, 200)).astype(np.float32)
    got, ref = defs.final_value_of_trend_vectorised(b), reference_final_value(b)
    check(f"2-D shape {b.shape} -> {got.shape}", got.shape == ref.shape)
    check(f"2-D agrees within {TOL:g} "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < TOL)

    # float64 input should be tighter still.
    c = rng.random((16, 50)).astype(np.float64)
    got, ref = defs.final_value_of_trend_vectorised(c), reference_final_value(c)
    check(f"float64 agrees within 1e-12 "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < 1e-12)

    # Window length is not hard-coded anywhere.
    for n in (5, 10, 16, 31):
        d = rng.random((n, 3, 20)).astype(np.float64)
        got, ref = (defs.final_value_of_trend_vectorised(d),
                    reference_final_value(d))
        check(f"window n={n:<2d} agrees", np.abs(got - ref).max() < 1e-12)

    # An exact straight line must return its own final value: the clearest
    # statement of what the SR1.5 estimator is supposed to do.
    slope, intercept, n = 0.03, 0.4, 16
    line = (intercept + slope * np.arange(n)).astype(np.float64)
    exact = intercept + slope * (n - 1)
    got = defs.final_value_of_trend_vectorised(line[:, None])[0]
    check(f"exact line recovers endpoint ({got:.10f} vs {exact:.10f})",
          abs(got - exact) < 1e-12)


def reference_rate(array):
    """Apply the original per-series function down axis 0."""
    flat = array.reshape(array.shape[0], -1)
    out = np.array([defs.rate_func(flat[:, i]) for i in range(flat.shape[1])])
    return out.reshape(array.shape[1:])


def test_rate_func():
    print("rate_func_vectorised vs rate_func")
    rng = np.random.default_rng(1)

    # 3-D, as used for the attributed and ERF rates: (years, vars, ensemble).
    # 10 years is the AR6 rate window (year-9 .. year inclusive).
    a = rng.random((10, 6, 400)).astype(np.float32)
    got, ref = defs.rate_func_vectorised(a), reference_rate(a)
    check(f"3-D shape {a.shape} -> {got.shape}", got.shape == ref.shape)
    check(f"3-D agrees within {TOL:g} "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < TOL)

    # 2-D: (years, ensemble).
    b = rng.random((10, 200)).astype(np.float32)
    got, ref = defs.rate_func_vectorised(b), reference_rate(b)
    check(f"2-D shape {b.shape} -> {got.shape}", got.shape == ref.shape)
    check(f"2-D agrees within {TOL:g} "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < TOL)

    # The slope is a single weighted sum, so float64 agreement should be at
    # machine epsilon rather than merely within tolerance.
    c = rng.random((10, 50)).astype(np.float64)
    got, ref = defs.rate_func_vectorised(c), reference_rate(c)
    check(f"float64 agrees within 1e-14 "
          f"(max {np.abs(got - ref).max():.2e})",
          np.abs(got - ref).max() < 1e-14)

    # Window length is not hard-coded anywhere.
    for n in (5, 10, 20):
        d = rng.random((n, 3, 20)).astype(np.float64)
        got, ref = defs.rate_func_vectorised(d), reference_rate(d)
        check(f"window n={n:<2d} agrees", np.abs(got - ref).max() < 1e-14)

    # An exact straight line must return its own slope: the clearest statement
    # of what the AR6 rate estimator is supposed to do.
    slope, intercept, n = 0.017, -0.2, 10
    line = (intercept + slope * np.arange(n)).astype(np.float64)
    got = defs.rate_func_vectorised(line[:, None])[0]
    check(f"exact line recovers slope ({got:.12f} vs {slope:.12f})",
          abs(got - slope) < 1e-12)

    # A flat series has zero trend.
    flat = np.full((10, 4), 1.234, dtype=np.float64)
    check("flat series gives zero rate",
          np.abs(defs.rate_func_vectorised(flat)).max() < 1e-15)


def test_percentile_threaded():
    print("percentile_threaded vs np.percentile")
    sig = [0.3, 5, 17, 33, 50, 67, 83, 95, 99.7]
    rng = np.random.default_rng(2)

    # Unlike the trend functions, this one must be EXACTLY equal, not merely
    # close: chunking the leading axis does not alter any individual
    # percentile computation. If this ever weakens to approximate agreement,
    # something is wrong with the chunk assembly.
    cases = [
        ((201, 6, 5000), 2, "attribution array (years, vars, ensemble)"),
        ((201, 6, 5000), -1, "negative axis"),
        ((6, 200000), 1, "headline/rate shape (vars, ensemble)"),
        ((201, 6, 2000), 2, "priors shape"),
        ((1, 6, 5000), 2, "single leading element -> serial fallback"),
        ((3, 6, 500), 2, "fewer leading elements than threads"),
        ((50, 4, 300), 1, "reduce a middle axis"),
        ((40, 900), 1, "2-D input"),
    ]
    for shape, axis, label in cases:
        a = rng.random(shape).astype(np.float32)
        ref = np.percentile(a, sig, axis=axis)
        got = defs.percentile_threaded(a, sig, axis=axis, n_threads=14)
        check(f"{label:44s} exactly equal",
              got.shape == ref.shape and np.array_equal(got, ref))

    # Thread count must not affect the result at all.
    a = rng.random((201, 6, 4000)).astype(np.float32)
    ref = np.percentile(a, sig, axis=2)
    for nt in (1, 2, 7, 14, 28, 64):
        got = defs.percentile_threaded(a, sig, axis=2, n_threads=nt)
        check(f"n_threads={nt:<3d} exactly equal", np.array_equal(got, ref))

    # axis=0 is the split axis, so reducing along it would silently produce
    # wrong numbers. It must raise instead.
    raised = False
    try:
        defs.percentile_threaded(a, sig, axis=0)
    except ValueError:
        raised = True
    check("axis=0 raises ValueError rather than corrupting", raised)


def test_contiguous_slice():
    print("contiguous_slice vs boolean-mask indexing")
    rng = np.random.default_rng(3)
    yrs = np.arange(1850, 2051)
    a = rng.random((201, 6, 2000)).astype(np.float32)
    sig = [5, 50, 95]

    # The three real window shapes: AR6 (10yr), SR1.5 (16yr), CGWL (20yr,
    # extending past the headline year).
    for lo, hi, label in [(2016, 2025, "AR6 10yr"),
                          (2010, 2025, "SR1.5 16yr"),
                          (2016, 2035, "CGWL 20yr"),
                          (1850, 1859, "first window"),
                          (2041, 2050, "last window")]:
        mask = ((lo <= yrs) * (yrs <= hi))
        sl = defs.contiguous_slice(mask)
        cp, vw = a[mask, :, :], a[sl, :, :]
        check(f"{label:14s} selects same block", np.array_equal(cp, vw))
        check(f"{label:14s} is a view, not a copy", np.shares_memory(vw, a))
        # The property that makes this safe: identical strides, so every
        # downstream reduction sees the same layout and is bitwise unchanged.
        check(f"{label:14s} strides match the copy", cp.strides == vw.strides)
        check(f"{label:14s} mean(axis=0) bitwise equal",
              np.array_equal(cp.mean(axis=0), vw.mean(axis=0)))
        check(f"{label:14s} rate bitwise equal",
              np.array_equal(defs.rate_func_vectorised(cp),
                             defs.rate_func_vectorised(vw)))
        check(f"{label:14s} percentile bitwise equal",
              np.array_equal(np.percentile(cp, sig, axis=2),
                             np.percentile(vw, sig, axis=2)))

    # An empty selection must not blow up.
    check("empty mask -> empty slice",
          defs.contiguous_slice(np.zeros(201, dtype=bool)) == slice(0, 0))

    # A non-contiguous mask must raise rather than silently select a
    # different set of years.
    gappy = np.zeros(201, dtype=bool)
    gappy[[10, 11, 50]] = True
    raised = False
    try:
        defs.contiguous_slice(gappy)
    except ValueError:
        raised = True
    check("non-contiguous mask raises ValueError", raised)


def main():
    test_final_value_of_trend()
    print()
    test_rate_func()
    print()
    test_percentile_threaded()
    print()
    test_contiguous_slice()
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
