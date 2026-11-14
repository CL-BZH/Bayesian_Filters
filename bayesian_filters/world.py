"""The turtle's world: arena, landmarks, motion model, sensors, and a simulator.

Every filter in this package is evaluated on data produced here, so they all
see *exactly* the same ground truth and the same noisy measurements.

Conventions
-----------
* A pose is ``x = (px, py, theta)`` with ``theta`` in radians, wrapped to
  ``[-pi, pi)``.  Arrays of poses have shape ``(..., 3)``.
* A control is ``u = (step, dtheta)``: the turtle first *turns* by ``dtheta``
  and then moves ``step`` forward along its new heading.
* A range-bearing measurement of landmark ``j`` is ``(r_j, phi_j)`` where
  ``phi_j`` is the bearing *relative to the turtle's heading*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------
def wrap_angle(a):
    """Wrap angle(s) to ``[-pi, pi)``."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles, weights=None, axis=0):
    """Weighted mean of angles, computed on the unit circle.

    The arithmetic mean of ``+179 deg`` and ``-179 deg`` is ``0 deg``; the
    circular mean is (correctly) ``180 deg``.
    """
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        s, c = np.sin(angles).mean(axis=axis), np.cos(angles).mean(axis=axis)
    else:
        w = np.asarray(weights, dtype=float)
        w = w.reshape(w.shape + (1,) * (angles.ndim - w.ndim)) if axis == 0 else w
        s = np.sum(w * np.sin(angles), axis=axis)
        c = np.sum(w * np.cos(angles), axis=axis)
    return np.arctan2(s, c)


# --------------------------------------------------------------------------
# Models (pure functions, vectorised over leading dimensions)
# --------------------------------------------------------------------------
def motion_model(x, u):
    """Noise-free unicycle step ``x_{t+1} = f(x_t, u_t)``.

    ``x`` has shape ``(..., 3)`` and ``u`` shape ``(..., 2)``; they broadcast.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    theta = x[..., 2] + u[..., 1]
    return np.stack(
        [
            x[..., 0] + u[..., 0] * np.cos(theta),
            x[..., 1] + u[..., 0] * np.sin(theta),
            wrap_angle(theta),
        ],
        axis=-1,
    )


def motion_jacobians(x, u):
    """Jacobians of :func:`motion_model` at ``(x, u)``.

    Returns ``F = df/dx`` (3x3) and ``V = df/du`` (3x2).
    """
    step = u[0]
    theta = x[2] + u[1]
    c, s = np.cos(theta), np.sin(theta)
    F = np.array([[1.0, 0.0, -step * s], [0.0, 1.0, step * c], [0.0, 0.0, 1.0]])
    V = np.array([[c, -step * s], [s, step * c], [0.0, 1.0]])
    return F, V


def range_bearing(x, landmarks):
    """Noise-free range-bearing measurements ``h(x)``.

    ``x``: ``(..., 3)``; ``landmarks``: ``(M, 2)``.  Returns ``(..., M, 2)``.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(landmarks, dtype=float) - x[..., None, :2]
    r = np.hypot(d[..., 0], d[..., 1])
    phi = wrap_angle(np.arctan2(d[..., 1], d[..., 0]) - x[..., None, 2])
    return np.stack([r, phi], axis=-1)


def range_bearing_jacobian(x, landmarks):
    """Jacobian of :func:`range_bearing` (stacked), shape ``(2M, 3)``.

    Row order matches ``range_bearing(x, landmarks).ravel()``:
    ``(r_1, phi_1, r_2, phi_2, ...)``.
    """
    d = np.asarray(landmarks, dtype=float) - x[:2]
    q = np.sum(d**2, axis=1)
    r = np.sqrt(q)
    H = np.zeros((2 * len(d), 3))
    H[0::2, 0] = -d[:, 0] / r
    H[0::2, 1] = -d[:, 1] / r
    H[1::2, 0] = d[:, 1] / q
    H[1::2, 1] = -d[:, 0] / q
    H[1::2, 2] = -1.0
    return H


def stack_residual(z, z_pred):
    """``z - z_pred`` for stacked ``(r, phi, r, phi, ...)`` vectors, bearings wrapped."""
    y = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
    y[..., 1::2] = wrap_angle(y[..., 1::2])
    return y


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------
#: Landmark colours used by the live demo.
LANDMARK_COLORS = ["#ffd700", "#ffa500", "#8a2be2", "#ff7f50", "#2e8b57", "#1e90ff", "#c71585", "#8b4513"]


@dataclass
class World:
    """Arena, landmarks and the *true* noise levels of the turtle and its sensors.

    The noise levels here are what nature does.  A filter is free to *assume*
    different values (and the notebooks show what happens when it does).
    """

    landmarks: np.ndarray
    half_width: float = 160.0
    half_height: float = 160.0
    # motion noise (per step)
    sigma_step: float = 0.4  # std of the distance actually travelled
    sigma_turn: float = 0.03  # std of the angle actually turned (rad)
    # sensor noise
    sigma_range: float = 5.0
    sigma_bearing: float = 0.05  # rad (about 3 degrees)
    sigma_gps: float = 8.0  # position fix, used by the linear Kalman filter
    sigma_compass: float = 0.02  # heading sensor, used by the linear KF and the grid filter
    max_range: float = np.inf

    def __post_init__(self):
        self.landmarks = np.asarray(self.landmarks, dtype=float).reshape(-1, 2)

    # -- construction -------------------------------------------------------
    @classmethod
    def random(cls, n_landmarks=5, min_spacing=100.0, seed=None, **kwargs):
        """Landmarks drawn uniformly in 85% of the arena with a minimum spacing."""
        rng = np.random.default_rng(seed)
        hw = kwargs.get("half_width", cls.half_width)
        hh = kwargs.get("half_height", cls.half_height)
        pts = []
        for _ in range(10_000):
            p = rng.uniform([-0.85 * hw, -0.85 * hh], [0.85 * hw, 0.85 * hh])
            if all(np.hypot(*(p - q)) > min_spacing for q in pts):
                pts.append(p)
                if len(pts) == n_landmarks:
                    break
        else:  # pragma: no cover
            raise RuntimeError("could not place landmarks; lower min_spacing")
        return cls(np.array(pts), **kwargs)

    @classmethod
    def default(cls, **kwargs):
        """The four fixed landmarks used throughout the notebooks."""
        lm = [(-39.9, 1.64), (59.3, -39.8), (62.3, 117.1), (-58.2, 103.15)]
        return cls(np.array(lm), **kwargs)

    # -- geometry -----------------------------------------------------------
    @property
    def n_landmarks(self):
        return len(self.landmarks)

    def inside(self, xy, margin=0.0):
        xy = np.asarray(xy)
        return (np.abs(xy[..., 0]) <= self.half_width + margin) & (
            np.abs(xy[..., 1]) <= self.half_height + margin
        )

    def sample_uniform_poses(self, n, rng):
        """``n`` poses uniform over the arena and over headings."""
        return np.column_stack(
            [
                rng.uniform(-self.half_width, self.half_width, n),
                rng.uniform(-self.half_height, self.half_height, n),
                rng.uniform(-np.pi, np.pi, n),
            ]
        )

    @property
    def motion_cov(self):
        """Covariance ``M`` of the control noise ``(step, dtheta)``."""
        return np.diag([self.sigma_step**2, self.sigma_turn**2])

    @property
    def range_bearing_cov(self):
        """Covariance of one range-bearing measurement."""
        return np.diag([self.sigma_range**2, self.sigma_bearing**2])

    # -- nature -------------------------------------------------------------
    def move(self, x, u, rng, noise=True):
        """Move the *true* turtle (control corrupted by noise)."""
        u = np.asarray(u, dtype=float)
        if noise:
            u = u + rng.normal(0.0, [self.sigma_step, self.sigma_turn])
        return motion_model(x, u)

    def sense_range_bearing(self, x, rng, noise=True):
        """Returns ``z`` of shape ``(M, 2)`` and the boolean ``visible`` mask."""
        z = range_bearing(x, self.landmarks)
        if noise:
            z = z + rng.normal(0.0, [self.sigma_range, self.sigma_bearing], size=z.shape)
            z[:, 1] = wrap_angle(z[:, 1])
        visible = z[:, 0] < self.max_range
        return z, visible

    def sense_gps(self, x, rng, noise=True):
        return np.asarray(x[:2], dtype=float) + (rng.normal(0.0, self.sigma_gps, 2) if noise else 0.0)

    def sense_compass(self, x, rng, noise=True):
        return float(wrap_angle(x[2] + (rng.normal(0.0, self.sigma_compass) if noise else 0.0)))


# --------------------------------------------------------------------------
# Trajectories and simulation
# --------------------------------------------------------------------------
def figure_eight_control(t, step=4.0, turn=np.pi / 50):
    """Nominal control: a figure of eight.

    One loop takes ``2 pi / turn`` steps (100 by default); the turning
    direction flips after every loop.
    """
    n_loop = int(round(2 * np.pi / turn))
    sign = 1.0 if (t // n_loop) % 2 == 0 else -1.0
    return np.array([step, sign * turn])


def fence_correction(world, x, start=0.8, gain=0.6, max_turn=0.25):
    """Extra turn that steers the turtle back towards the centre when it
    leaves the inner ``start`` fraction of the arena.

    A smooth steering command (rather than, say, a sudden half-turn at the
    wall) keeps the motion followable by every filter.  The pilot knows the true pose; the
    filters only ever see the *commanded* control, which includes this term.
    """
    if abs(x[0]) < start * world.half_width and abs(x[1]) < start * world.half_height:
        return 0.0
    to_centre = np.arctan2(-x[1], -x[0])
    return float(np.clip(gain * wrap_angle(to_centre - x[2]), -max_turn, max_turn))


@dataclass
class Run:
    """One simulated episode.  Index ``t`` of the measurement arrays refers to
    the measurement taken *after* applying control ``controls[t]``, i.e. at
    pose ``truth[t + 1]``."""

    world: World
    truth: np.ndarray  # (T+1, 3)
    controls: np.ndarray  # (T, 2), the commanded controls (what the filters know)
    z_rb: np.ndarray  # (T, M, 2)
    visible: np.ndarray  # (T, M)
    z_gps: np.ndarray  # (T, 2)
    z_compass: np.ndarray  # (T,)
    kidnap_step: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self):
        return len(self.controls)


def simulate(
    world,
    n_steps=400,
    x0=(0.0, 0.0, 0.0),
    seed=0,
    control_fn=figure_eight_control,
    fence=True,
    kidnap=None,
    motion_noise=True,
    sensor_noise=True,
):
    """Simulate the turtle and all its sensors.

    Parameters
    ----------
    kidnap : ``None`` or ``(step, pose)``
        Teleport the true turtle to ``pose`` just before control ``step`` is
        applied (the "kidnapped robot" problem).  The filters are not told.
    """
    rng = np.random.default_rng(seed)
    T, M = n_steps, world.n_landmarks
    truth = np.zeros((T + 1, 3))
    truth[0] = x0
    controls = np.zeros((T, 2))
    z_rb = np.zeros((T, M, 2))
    visible = np.zeros((T, M), dtype=bool)
    z_gps = np.zeros((T, 2))
    z_compass = np.zeros(T)
    x = np.array(x0, dtype=float)
    for t in range(T):
        if kidnap is not None and t == kidnap[0]:
            x = np.array(kidnap[1], dtype=float)
        u = np.array(control_fn(t), dtype=float)
        if fence:
            u[1] += fence_correction(world, x)
        x = world.move(x, u, rng, noise=motion_noise)
        controls[t] = u
        truth[t + 1] = x
        z_rb[t], visible[t] = world.sense_range_bearing(x, rng, noise=sensor_noise)
        z_gps[t] = world.sense_gps(x, rng, noise=sensor_noise)
        z_compass[t] = world.sense_compass(x, rng, noise=sensor_noise)
    return Run(
        world,
        truth,
        controls,
        z_rb,
        visible,
        z_gps,
        z_compass,
        kidnap_step=None if kidnap is None else kidnap[0],
        meta=dict(seed=seed, x0=tuple(x0)),
    )
