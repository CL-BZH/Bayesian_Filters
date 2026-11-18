"""The linear Kalman filter.

:class:`KalmanFilter` is the textbook filter for

    x_{t+1} = F x_t + B u_t + w_t,      w_t ~ N(0, Q)
    z_t     = H x_t + v_t,              v_t ~ N(0, R)

:class:`CompassGpsKF` applies it to the mower.  The pose model of the mower
is *not* linear (the heading enters through cos/sin), so we make the problem
linear honestly: the mower carries a **compass**.  With the heading read off
the compass, the displacement ``step * (cos theta, sin theta)`` is a known
input and the position obeys a linear model; a GPS-like sensor gives a noisy
position fix.  The filter estimates ``(px, py)`` only.
"""

from __future__ import annotations

import numpy as np

from .base import Filter


class KalmanFilter:
    """Generic linear Kalman filter (no mower-specific code)."""

    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        # kept for inspection / teaching
        self.K = None
        self.S = None
        self.y = None

    def predict(self, F, Q, B=None, u=None):
        self.x = F @ self.x + (B @ u if B is not None else 0.0)
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, z, H, R):
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + R  # innovation covariance
        K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^{-1}  (S, P symmetric)
        self.x = self.x + K @ y
        # Joseph form: stays symmetric positive semi-definite in floating point
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.y, self.S, self.K = y, S, K
        return self.x


class CompassGpsKF(Filter):
    """Linear KF on the mower's position, using compass + odometry + GPS.

    Prediction: ``p_{t+1} = p_t + step_t * (cos c_{t+1}, sin c_{t+1}) + w_t``
    where ``c_{t+1}`` is the compass reading of the heading ``theta_{t+1}``
    that the mower has after its turn, the heading along which it drives
    (same notation as the book, Section 3.3; in a ``Run`` it is
    ``z_compass[t]``).  The noise ``w_t`` comes from the step noise (along
    the heading) and from the compass noise (across it); to first order its
    covariance is ``G diag(sigma_step^2, sigma_compass^2) G^T`` with
    ``G = [[cos c, -step sin c], [sin c, step cos c]]``.
    """

    name = "KF"
    measurement = "gps"

    def __init__(self, x0, P0, sigma_step, sigma_compass, sigma_gps):
        self.kf = KalmanFilter(np.asarray(x0, dtype=float)[:2], np.asarray(P0, dtype=float)[:2, :2])
        self.sigma_step = sigma_step
        self.sigma_compass = sigma_compass
        self.R = np.eye(2) * sigma_gps**2
        self.H = np.eye(2)
        self.heading = float(np.asarray(x0)[2]) if len(x0) > 2 else 0.0

    @classmethod
    def from_world(cls, world, x0, P0):
        return cls(x0, P0, world.sigma_step, world.sigma_compass, world.sigma_gps)

    def predict(self, u, heading=None):
        if heading is not None:
            self.heading = heading
        step = u[0]
        cos_c, sin_c = np.cos(self.heading), np.sin(self.heading)
        G = np.array([[cos_c, -step * sin_c], [sin_c, step * cos_c]])
        Q = G @ np.diag([self.sigma_step**2, self.sigma_compass**2]) @ G.T
        self.kf.predict(np.eye(2), Q, B=np.eye(2), u=step * np.array([cos_c, sin_c]))

    def update(self, z, visible=None):
        self.kf.update(np.asarray(z, dtype=float), self.H, self.R)

    def predict_from_run(self, run, t):
        self.predict(run.controls[t], heading=run.z_compass[t])

    def update_from_run(self, run, t):
        self.update(run.z_gps[t])

    @property
    def mean(self):
        return self.kf.x.copy()

    @property
    def cov(self):
        return self.kf.P.copy()
