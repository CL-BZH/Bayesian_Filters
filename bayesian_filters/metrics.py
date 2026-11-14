"""How good is a filter?  Accuracy (RMSE) and honesty (NEES)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .world import wrap_angle


def errors(means, truth):
    """Estimation errors, heading wrapped.  ``means`` may be 2-D (position only)."""
    n = means.shape[1]
    e = means - truth[:, :n]
    if n > 2:
        e[:, 2] = wrap_angle(e[:, 2])
    return e


def position_error(means, truth):
    """Euclidean position error at every step."""
    e = errors(means, truth)
    return np.hypot(e[:, 0], e[:, 1])


def position_rmse(means, truth, skip=0):
    """Root-mean-square position error, ignoring the first ``skip`` steps."""
    pe = position_error(means, truth)[skip:]
    return float(np.sqrt(np.nanmean(pe**2)))


def nees(means, covs, truth, dims=None):
    """Normalised estimation error squared ``e^T P^{-1} e`` at every step.

    For a *consistent* filter (one whose covariance is honest) the NEES of an
    ``n``-dimensional state is chi-square distributed with ``n`` degrees of
    freedom, so it averages ``n``.  Much larger: over-confident filter.
    Much smaller: pessimistic filter.
    """
    e = errors(means, truth)
    if dims is not None:
        e = e[:, dims]
        covs = covs[:, dims][:, :, dims]
    out = np.full(len(e), np.nan)
    for t in range(len(e)):
        if np.all(np.isfinite(covs[t])) and np.all(np.isfinite(e[t])):
            try:
                out[t] = e[t] @ np.linalg.solve(covs[t], e[t])
            except np.linalg.LinAlgError:  # degenerate covariance (e.g. collapsed particles)
                out[t] = np.inf
    return out


def nees_bounds(dof, n_runs=1, p=0.95):
    """Two-sided ``p`` acceptance interval for the NEES averaged over ``n_runs``
    independent runs (``n_runs * mean NEES`` is chi-square with ``n_runs * dof``)."""
    lo, hi = stats.chi2.ppf([(1 - p) / 2, (1 + p) / 2], dof * n_runs)
    return lo / n_runs, hi / n_runs
