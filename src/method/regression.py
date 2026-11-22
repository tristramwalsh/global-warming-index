"""Regression methods: the scaling factors that fit the modelled warming to
the observations, and their uncertainty from internal variability.

Stage B of the GWI. The axes are named as everywhere in the engine:

    p  FaIR parameter set
    e  ERF ensemble member
    o  observed GMT member
    c  piControl segment
    t  year (here, the regression years)
    k  regressor: one per regressed variable (e.g. GHG, Nat, OHF), plus one
       for the constant term if it is used

For each regressor matrix X (t × k), the modelled warming of the regressed
variables over the regression years:

    β_obs[o] = the least-squares fit of observed GMT member o onto X
    β_pic[c] = the least-squares fit of piControl segment c onto X

each a vector of k scaling factors. A member (o, c) is scaled by

    β = β_obs[o] + β_pic[c]

which is the least-squares fit of T_obs[o] + T_pic[c] onto X, because the
least-squares solution is linear in the target (see `ols`). So each fit is
done once per observed member and once per segment, not once per member,
and the obs and piControl parts are stored apart.

There is one regressor matrix per ERF member and parameter set: the leading
axes (...) of X are e.g. (e,) for one parameter set, or (p, e).

Each method returns its Coefficients. Any other method (e.g. a Bayesian
regression) fits this interface only if it too is linear (affine) in the
target, so that its obs and piControl parts can be added.
"""

from typing import NamedTuple

import numpy as np


class Coefficients(NamedTuple):
    """Scaling factors from a regression method.

    obs: array (..., o, k), the fit of each observed GMT member
    pic: array (..., c, k), the fit of each piControl segment
    """
    obs: np.ndarray
    pic: np.ndarray


def ols(X, Y_obs, Y_pic):
    """Ordinary least squares, every fit at once.

    X: array (..., t, k), the regressors over the regression years: the
        modelled warming of each regressed variable, and a column of ones
        for the constant if it is used
    Y_obs: array (t, o), observed GMT members over the same years
    Y_pic: array (t, c), piControl segments over the same years
    Returns Coefficients, obs (..., o, k) and pic (..., c, k).

    The problem. For each regressor matrix X (t × k) and each target y (a
    vector over the t years: one observed member or one segment), find the
    scaling factors β (k of them) that minimise the sum of squared errors

        β̂ = argmin_β  Σ_t ( y_t − Σ_k X_tk β_k )²  =  argmin_β ‖y − Xβ‖²

    Setting the derivative to zero gives the normal equations,
    XᵀX β̂ = Xᵀy, so when the k columns of X are linearly independent

        β̂ = (XᵀX)⁻¹ Xᵀ y  =  X⁺ y

    The pseudo-inverse. X⁺ = (XᵀX)⁻¹ Xᵀ (k × t) is the Moore-Penrose
    pseudo-inverse of X: the nearest thing to an inverse that a non-square
    matrix has. It depends on X alone, not on y, so one X⁺ per regressor
    matrix serves every target: β̂_obs[o] = X⁺ y_obs[o] for every o, and
    β̂_pic[c] = X⁺ y_pic[c] for every c. The same formula also shows that β̂
    is linear in y: X⁺ (y_obs + y_pic) = X⁺ y_obs + X⁺ y_pic.

    How it is computed. Through the singular value decomposition
    X = U S Vᵀ, where U (t × k) and V (k × k) have orthonormal columns and
    S = diag(σ_1 ≥ σ_2 ≥ … ≥ σ_k ≥ 0) holds the singular values. They say
    how strongly X stretches each of k independent combinations of the
    regressors (the columns of V). Then X⁺ = V S⁺ Uᵀ, with S⁺ = diag(1/σ_i).
    This is equivalent to (XᵀX)⁻¹ Xᵀ but numerically safer: it never forms
    XᵀX, which would square the condition number σ_1/σ_k.

    The cut-off. A singular value near zero means that some combination of
    the regressors barely varies over the regression years (e.g. GHG and OHF
    responses of almost the same shape), so the data can't pin down that
    combination of scaling factors, and 1/σ_i would mostly amplify rounding
    error. So singular values below  ε · max(t, k) · σ_1  are treated as
    zero (1/σ_i replaced by 0), which gives the smallest β̂ among the
    equally good fits. ε = 2.2 × 10⁻¹⁶ is float64's machine epsilon: the
    relative rounding error of a single arithmetic operation. With t = 176
    years the cut-off is ~4 × 10⁻¹⁴ σ_1, which discards only combinations
    indistinguishable from rounding noise. For the GWI's regressors,
    σ_1/σ_k is at most ~10³, so nothing is cut off and X⁺ = (XᵀX)⁻¹ Xᵀ. This
    is the cut-off np.linalg.lstsq uses by default (rcond=None), which this
    function replaces: one lstsq call per (X, y) pair gives the same β̂ to
    machine precision.
    """
    # Regress in float64, whatever the inputs' precision
    X = np.asarray(X, dtype=float)
    # The sizes of each regressor matrix: years (t) by regressors (k)
    n_years, n_regressors = X.shape[-2:]
    # Singular values smaller than this fraction of the largest count as zero
    # (np.linalg.lstsq's default)
    cutoff = np.finfo(float).eps * max(n_years, n_regressors)
    # The pseudo-inverse of every regressor matrix at once: (..., k, t)
    X_pinv = np.linalg.pinv(X, rcond=cutoff)
    # Fit every observed member: (..., k, t) @ (t, o) -> (..., k, o), then
    # swap the last two axes, to put each member's coefficients last:
    # (..., o, k)
    beta_obs = np.swapaxes(X_pinv @ np.asarray(Y_obs, dtype=float), -1, -2)
    # Fit every piControl segment in the same way: (..., c, k)
    beta_pic = np.swapaxes(X_pinv @ np.asarray(Y_pic, dtype=float), -1, -2)
    # Both sets of scaling factors, kept apart so they can be added per member
    return Coefficients(obs=beta_obs, pic=beta_pic)
