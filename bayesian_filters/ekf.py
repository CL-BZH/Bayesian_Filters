"""Extended Kalman filter for the mower: unicycle motion + range-bearing landmarks."""

from __future__ import annotations

import numpy as np

from .base import Filter
from .world import (
    motion_jacobians,
    motion_model,
    range_bearing,
    range_bearing_jacobian,
    stack_residual,
    wrap_angle,
)


class EKF(Filter):
    """EKF on the pose ``(px, py, theta)``.

    Predict::

        x <- f(x, u)
        P <- F P F^T + V M V^T            (F = df/dx, V = df/du at the old mean)

    Update with all visible landmarks stacked into one vector::

        y = z - h(x)                      (bearing residuals wrapped!)
        S = H P H^T + R
        K = P H^T S^{-1}
        x <- x + K y,   P <- (I - K H) P (I - K H)^T + K R K^T     (Joseph form)

    With ``association="known"`` the sensor reports which landmark it saw.  With ``association="nearest"`` it does not,
    and each measurement is matched to the most likely landmark first.
    """

    name = "EKF"

    def __init__(self, landmarks, x0, P0, motion_cov, meas_cov, association="known"):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.x[2] = wrap_angle(self.x[2])
        self.P = np.asarray(P0, dtype=float).copy()
        self.M = np.asarray(motion_cov, dtype=float)
        self.R1 = np.asarray(meas_cov, dtype=float)  # for one landmark
        self.association = association  # "known" or "nearest"
        self.last_associations = None
        self.last_innovation = None
        self.last_S = None

    @classmethod
    def from_world(cls, world, x0, P0, inflate_motion=1.0, **kw):
        return cls(world.landmarks, x0, P0, world.motion_cov * inflate_motion**2, world.range_bearing_cov, **kw)

    def predict(self, u):
        u = np.asarray(u, dtype=float)
        F, V = motion_jacobians(self.x, u)
        self.x = motion_model(self.x, u)
        self.P = F @ self.P @ F.T + V @ self.M @ V.T

    def update(self, z, visible=None):
        z = np.asarray(z, dtype=float)
        if visible is None:
            visible = np.ones(len(z), dtype=bool)
        if not np.any(visible):
            return
        if self.association == "nearest":
            idx = self.associate(z[visible])
        else:
            idx = np.flatnonzero(visible)
        self.last_associations = idx
        lm = self.landmarks[idx]
        zv = z[visible].ravel()
        H = range_bearing_jacobian(self.x, lm)
        y = stack_residual(zv, range_bearing(self.x, lm).ravel())
        R = np.kron(np.eye(len(lm)), self.R1)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.last_innovation, self.last_S = y, S

    def associate(self, zs):
        """Nearest-neighbour (maximum-likelihood) data association: each
        measurement is attributed to the landmark with the smallest
        Mahalanobis distance of its innovation.  This is what an EKF must do
        when the sensor does not say *which* landmark it saw."""
        idx = []
        for zk in zs:
            best, best_d = 0, np.inf
            for j, lm in enumerate(self.landmarks):
                H = range_bearing_jacobian(self.x, lm[None])
                y = stack_residual(zk, range_bearing(self.x, lm[None])[0])
                S = H @ self.P @ H.T + self.R1
                d = y @ np.linalg.solve(S, y) + np.log(np.linalg.det(S))
                if d < best_d:
                    best, best_d = j, d
            idx.append(best)
        return np.array(idx)

    @property
    def mean(self):
        return self.x.copy()

    @property
    def cov(self):
        return self.P.copy()
