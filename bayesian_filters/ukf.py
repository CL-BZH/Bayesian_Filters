"""Unscented Kalman filter for the mower.

The building blocks (:func:`merwe_weights`, :func:`sigma_points`,
:func:`unscented_transform`) are generic; :class:`UKF` plugs the mower's
motion and range-bearing models into them.

Design (see notebook 04):

* the filter uses the *commanded* control (localisation);
* scaled sigma points (alpha, beta, kappa) keep the centre weight
  non-negative, unlike ``lambda = 3 - n`` for ``n > 3``;
* sigma points are regenerated from the prior before the update.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky

from .base import Filter
from .world import circular_mean, motion_model, range_bearing, wrap_angle


# --------------------------------------------------------------------------
# Generic unscented machinery
# --------------------------------------------------------------------------
def merwe_weights(n, alpha=1.0, beta=2.0, kappa=0.0):
    """Weights of the scaled unscented transform (van der Merwe).

    Returns ``(lam, Wm, Wc)`` with ``lam = alpha^2 (n + kappa) - n``.
    """
    lam = alpha**2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 0.5 / (n + lam))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    return lam, Wm, Wc


def sigma_points(mean, cov, lam):
    """The ``2n + 1`` sigma points of ``N(mean, cov)``; shape ``(2n+1, n)``.

    ``cov = L L^T`` and the points are ``mean``, ``mean +/- sqrt(n+lam) L[:, i]``.
    """
    mean = np.asarray(mean, dtype=float)
    n = len(mean)
    L = cholesky((n + lam) * cov, lower=True)  # raises LinAlgError if not PD
    return np.vstack([mean, mean + L.T, mean - L.T])


def unscented_transform(sigmas, Wm, Wc, mean_fn=None, residual_fn=None, noise_cov=None):
    """Mean and covariance of transformed sigma points ``sigmas`` (``(2n+1, m)``).

    ``mean_fn(sigmas, Wm)`` and ``residual_fn(a, b)`` default to the
    Euclidean ones; pass angle-aware versions when a component is an angle.
    """
    mean = mean_fn(sigmas, Wm) if mean_fn else Wm @ sigmas
    d = residual_fn(sigmas, mean) if residual_fn else sigmas - mean
    cov = (Wc[:, None] * d).T @ d
    if noise_cov is not None:
        cov = cov + noise_cov
    return mean, cov, d


# --------------------------------------------------------------------------
# Angle-aware means/residuals for the mower
# --------------------------------------------------------------------------
def pose_mean(sigmas, W):
    m = W @ sigmas
    m[2] = circular_mean(sigmas[:, 2], W)
    return m


def pose_residual(a, b):
    d = a - b
    d[..., 2] = wrap_angle(d[..., 2])
    return d


def meas_mean(sigmas, W):
    """Stacked ``(r, phi, r, phi, ...)``: ranges averaged, bearings on the circle."""
    m = W @ sigmas
    m[1::2] = circular_mean(sigmas[:, 1::2], W)
    return m


def meas_residual(a, b):
    d = a - b
    d[..., 1::2] = wrap_angle(d[..., 1::2])
    return d


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------
class UKF(Filter):
    """UKF on the pose ``(px, py, theta)``.

    The control noise enters non-additively (it is turned before it is
    driven), so the prediction uses an **augmented** state
    ``(px, py, theta, n_step, n_turn)`` with covariance ``diag(P, M)``.  The
    measurement noise is additive.
    """

    name = "UKF"

    def __init__(self, landmarks, x0, P0, motion_cov, meas_cov, alpha=1.0, beta=2.0, kappa=0.0):
        self.landmarks = np.asarray(landmarks, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.x[2] = wrap_angle(self.x[2])
        self.P = np.asarray(P0, dtype=float).copy()
        self.M = np.asarray(motion_cov, dtype=float)
        self.R1 = np.asarray(meas_cov, dtype=float)
        self.params = dict(alpha=alpha, beta=beta, kappa=kappa)
        self.w_aug = merwe_weights(5, **self.params)
        self.w_x = merwe_weights(3, **self.params)
        self.last_sigmas = None

    @classmethod
    def from_world(cls, world, x0, P0, inflate_motion=1.0, **kw):
        return cls(world.landmarks, x0, P0, world.motion_cov * inflate_motion**2, world.range_bearing_cov, **kw)

    def predict(self, u):
        lam, Wm, Wc = self.w_aug
        mean_aug = np.concatenate([self.x, [0.0, 0.0]])
        cov_aug = np.zeros((5, 5))
        cov_aug[:3, :3] = self.P
        cov_aug[3:, 3:] = self.M
        chi = sigma_points(mean_aug, cov_aug, lam)  # (11, 5)
        prop = motion_model(chi[:, :3], np.asarray(u, dtype=float) + chi[:, 3:])  # (11, 3)
        self.x, P, _ = unscented_transform(prop, Wm, Wc, pose_mean, pose_residual)
        self.P = 0.5 * (P + P.T)
        self.last_sigmas = prop

    def update(self, z, visible=None):
        z = np.asarray(z, dtype=float)
        if visible is None:
            visible = np.ones(len(z), dtype=bool)
        if not np.any(visible):
            return
        lm = self.landmarks[visible]
        lam, Wm, Wc = self.w_x
        chi = sigma_points(self.x, self.P, lam)  # (7, 3), fresh from the prior
        Z = range_bearing(chi, lm).reshape(len(chi), -1)  # (7, 2 Mv)
        R = np.kron(np.eye(len(lm)), self.R1)
        z_mean, S, dz = unscented_transform(Z, Wm, Wc, meas_mean, meas_residual, R)
        dx = pose_residual(chi, self.x)
        Pxz = (Wc[:, None] * dx).T @ dz
        K = np.linalg.solve(S, Pxz.T).T  # Pxz S^{-1}
        y = meas_residual(z[visible].ravel(), z_mean)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        P = self.P - K @ S @ K.T
        self.P = 0.5 * (P + P.T)

    @property
    def mean(self):
        return self.x.copy()

    @property
    def cov(self):
        return self.P.copy()
