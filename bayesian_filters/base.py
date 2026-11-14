"""Common interface for every filter, and a driver that runs one over a :class:`Run`."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


class Filter:
    """A recursive Bayesian filter.

    Every subclass implements:

    * ``predict(u)``  -- time update with control ``u = (step, dtheta)``
    * ``update(z, visible)`` -- measurement update
    * ``mean`` / ``cov`` -- the current Gaussian summary of the belief
      (for a particle filter: the weighted sample mean and covariance)
    """

    name = "filter"
    #: which measurement of a :class:`~bayesian_filters.world.Run` it consumes
    measurement = "range_bearing"

    def predict(self, u):  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, z, visible=None):  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def mean(self):  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def cov(self):  # pragma: no cover - interface
        raise NotImplementedError

    # optional hooks used by the driver
    def predict_from_run(self, run, t):
        self.predict(run.controls[t])

    def update_from_run(self, run, t):
        self.update(run.z_rb[t], run.visible[t])

    def snapshot(self):
        """Extra per-step state to record (e.g. particles).  Default: nothing."""
        return None


@dataclass
class FilterResult:
    name: str
    means: np.ndarray  # (T+1, n)
    covs: np.ndarray  # (T+1, n, n)
    seconds: float
    snapshots: list | None = None
    diverged: bool = False


def run_filter(flt, run, keep_snapshots=False, snapshot_every=1):
    """Run ``flt`` over every step of ``run`` and record its belief.

    ``means[0]`` is the initial belief; ``means[t + 1]`` the belief after
    predicting with ``controls[t]`` and updating with the ``t``-th measurement.
    If the filter raises (e.g. a covariance stops being positive definite),
    the remaining steps are filled with NaN and ``diverged`` is set.
    """
    T = run.n_steps
    n = len(flt.mean)
    means = np.full((T + 1, n), np.nan)
    covs = np.full((T + 1, n, n), np.nan)
    means[0], covs[0] = flt.mean, flt.cov
    snaps = [flt.snapshot()] if keep_snapshots else None
    diverged = False
    t0 = time.perf_counter()
    for t in range(T):
        try:
            flt.predict_from_run(run, t)
            flt.update_from_run(run, t)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            diverged = True
            break
        means[t + 1], covs[t + 1] = flt.mean, flt.cov
        if keep_snapshots and (t + 1) % snapshot_every == 0:
            snaps.append(flt.snapshot())
    seconds = time.perf_counter() - t0
    return FilterResult(flt.name, means, covs, seconds, snaps, diverged)
