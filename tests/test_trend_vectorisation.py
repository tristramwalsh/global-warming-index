"""Pin the vectorised trend functions to the per-series originals.

The vectorised forms in src/definitions.py replace np.polyfit with the
closed-form least-squares solution, which is mathematically identical but
orders the floating-point operations differently. They are therefore expected
to agree with the originals to rounding, not exactly.

This guards that equivalence: if either implementation is edited such that they
diverge beyond rounding, this fails.

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


def main():
    test_final_value_of_trend()
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
