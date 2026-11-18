"""Closing the loop: a planned path, a controller that follows it, and a
simulation in which the filter itself steers the robot.

In :func:`~bayesian_filters.world.simulate` the filters replay a recorded
run.  On a real robot the filter is *in the loop*: at every step the
controller looks at the filter's current estimate, decides the next command,
the robot moves, the sensors read, and the filter is updated.  A wrong
estimate then becomes a wrong command, and the robot leaves its path.
"""

from __future__ import annotations

import time

import numpy as np

from .base import FilterResult
from .ekf import EKF
from .world import Run, wrap_angle


# --------------------------------------------------------------------------
# The planned path
# --------------------------------------------------------------------------
def stripes_path(world, spacing=40.0, margin=30.0):
    """Waypoints of a mowing pattern: parallel stripes, driven back and forth.

    The stripes run along ``x`` between ``-half_width + margin`` and
    ``half_width - margin``; consecutive stripes are ``spacing`` apart in
    ``y``, starting at the bottom.  Returns an array of shape ``(K, 2)``.
    """
    x_lo, x_hi = -world.half_width + margin, world.half_width - margin
    ys = np.arange(-world.half_height + margin, world.half_height - margin + 1e-9, spacing)
    pts = []
    for k, y in enumerate(ys):
        a, b = (x_lo, x_hi) if k % 2 == 0 else (x_hi, x_lo)
        pts += [(a, y), (b, y)]
    return np.array(pts)


def distance_to_path(points, waypoints):
    """Distance from each of ``points`` (``(n, 2)``) to the polyline through
    ``waypoints``: the *cross-track error* of a robot that should be on it."""
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    a, b = waypoints[:-1], waypoints[1:]
    ab = b - a
    t = np.einsum("nkd,kd->nk", points[:, None] - a[None], ab) / np.einsum("kd,kd->k", ab, ab)
    proj = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None] - proj, axis=-1).min(axis=1)


class PathFollower:
    """Follow a polyline: aim at a point a little further along the path.

    At each call the controller receives a pose ``(px, py, theta)``, which
    is the robot's *estimate* of where it is.  It finds where that position
    is along the path (never going backwards), takes the point ``lookahead``
    further on as a target, and commands a turn towards it, limited to
    ``max_turn``, followed by a step of length ``step`` ("pure pursuit").
    ``done`` becomes true at the end of the path.
    """

    def __init__(self, waypoints, step=4.0, lookahead=12.0, max_turn=0.5):
        self.w = np.asarray(waypoints, dtype=float)
        seg = np.diff(self.w, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.step, self.lookahead, self.max_turn = step, lookahead, max_turn
        self.progress = 0.0  # arc length reached so far
        self.done = False

    def point_at(self, s):
        """The point at arc length ``s`` along the path."""
        s = float(np.clip(s, 0.0, self.length))
        k = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg_len) - 1)
        return self.w[k] + (s - self.cum[k]) * self.seg_dir[k]

    def _advance(self, p):
        """Move ``progress`` to the point of the path closest to ``p``,
        looking only a short way ahead so that a neighbouring stripe is never
        mistaken for the current one."""
        window = np.linspace(self.progress, min(self.progress + 3 * self.lookahead, self.length), 60)
        pts = np.array([self.point_at(s) for s in window])
        self.progress = float(window[np.argmin(np.linalg.norm(pts - p, axis=1))])

    def __call__(self, pose):
        pose = np.asarray(pose, dtype=float)
        p, theta = pose[:2], pose[2]
        self._advance(p)
        if self.length - self.progress < self.step / 2:
            self.done = True
            return np.zeros(2)
        target = self.point_at(self.progress + self.lookahead)
        d = target - p
        turn = np.clip(wrap_angle(np.arctan2(d[1], d[0]) - theta), -self.max_turn, self.max_turn)
        return np.array([min(self.step, np.hypot(*d)), float(turn)])


# --------------------------------------------------------------------------
# A "filter" that does not filter
# --------------------------------------------------------------------------
class DeadReckoning(EKF):
    """Integrates the commands and never looks at a sensor (the prediction
    step of the EKF alone).  In the loop, a robot steered by it believes it
    is exactly on its path while it drifts away."""

    name = "dead reckoning"

    def update(self, z, visible=None):
        return


# --------------------------------------------------------------------------
# The closed loop
# --------------------------------------------------------------------------
def simulate_closed_loop(
    world,
    flt,
    controller,
    n_steps=400,
    x0=(0.0, 0.0, 0.0),
    seed=0,
    kidnap=None,
    motion_noise=True,
    sensor_noise=True,
):
    """Simulate the robot with the filter ``flt`` in the loop.

    At step ``t``::

        u_t     = controller(estimate of the pose, from flt)
        x_{t+1} = f(x_t, u_t + noise)            (the true robot moves)
        z_{t+1} = sensors at x_{t+1}
        flt.predict(u_t); flt.update(z_{t+1})

    The command is computed from the filter's estimate, hence from the
    measurements received so far, and never from the true pose.  A filter
    that estimates the position only (the linear Kalman filter) is completed
    with the latest compass reading.

    Returns ``(run, result)``: the :class:`~bayesian_filters.world.Run` that
    was produced (true poses, commands, sensor readings) and the
    :class:`~bayesian_filters.base.FilterResult` of ``flt`` on it.  The
    simulation stops early when the controller reports ``done``.
    """
    rng = np.random.default_rng(seed)
    T, M = n_steps, world.n_landmarks
    run = Run(
        world,
        truth=np.zeros((T + 1, 3)),
        controls=np.zeros((T, 2)),
        z_rb=np.zeros((T, M, 2)),
        visible=np.zeros((T, M), dtype=bool),
        z_gps=np.zeros((T, 2)),
        z_compass=np.zeros(T),
        kidnap_step=None if kidnap is None else kidnap[0],
        meta=dict(seed=seed, x0=tuple(x0), closed_loop=True),
    )
    n = len(flt.mean)
    means = np.full((T + 1, n), np.nan)
    covs = np.full((T + 1, n, n), np.nan)
    means[0], covs[0] = flt.mean, flt.cov
    x = np.array(x0, dtype=float)
    run.truth[0] = x
    heading = float(x0[2])  # used only by position-only filters
    t_end = T
    t0 = time.perf_counter()
    for t in range(T):
        if kidnap is not None and t == kidnap[0]:
            x = np.array(kidnap[1], dtype=float)
        est = flt.mean
        if len(est) < 3:
            est = np.array([est[0], est[1], heading])
        u = np.asarray(controller(est), dtype=float)
        if getattr(controller, "done", False):
            t_end = t
            break
        x = world.move(x, u, rng, noise=motion_noise)
        run.controls[t] = u
        run.truth[t + 1] = x
        run.z_rb[t], run.visible[t] = world.sense_range_bearing(x, rng, noise=sensor_noise)
        run.z_gps[t] = world.sense_gps(x, rng, noise=sensor_noise)
        run.z_compass[t] = world.sense_compass(x, rng, noise=sensor_noise)
        heading = run.z_compass[t]
        flt.predict_from_run(run, t)
        flt.update_from_run(run, t)
        means[t + 1], covs[t + 1] = flt.mean, flt.cov
    seconds = time.perf_counter() - t0
    if t_end < T:  # trim to the steps actually driven
        run.truth, run.controls = run.truth[: t_end + 1], run.controls[:t_end]
        run.z_rb, run.visible = run.z_rb[:t_end], run.visible[:t_end]
        run.z_gps, run.z_compass = run.z_gps[:t_end], run.z_compass[:t_end]
        means, covs = means[: t_end + 1], covs[: t_end + 1]
    return run, FilterResult(flt.name, means, covs, seconds)
