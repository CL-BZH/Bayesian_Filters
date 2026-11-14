"""Particle filter (Monte Carlo localisation) for the turtle.

Design (see notebook 05):

* particles are rows of an ``(N, 3)`` NumPy array;
* weights are kept as **log**-weights and normalised;
* resampling happens only when the effective sample size drops below a
  threshold;
* recovery from a lost / kidnapped turtle uses *augmented MCL*
  (Thrun, Burgard & Fox, Probabilistic Robotics, Table 8.3);
* the likelihood uses the same sensor model as the Kalman filters, and data
  association can be *known* (as for the KFs) or *unknown* (a mixture over
  landmarks).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .base import Filter
from .world import circular_mean, motion_model, range_bearing, wrap_angle


def systematic_resample(weights, rng):
    """Indices drawn by systematic resampling (one uniform, ``N`` evenly spaced
    pointers into the cumulative weights).  Low variance, ``O(N)``."""
    N = len(weights)
    positions = (rng.uniform() + np.arange(N)) / N
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # guard against round-off
    return np.searchsorted(cdf, positions, side="left")


def multinomial_resample(weights, rng):
    """Indices drawn i.i.d. from the weights (higher variance; for comparison)."""
    return rng.choice(len(weights), size=len(weights), p=weights)


def effective_sample_size(weights):
    return 1.0 / np.sum(np.asarray(weights) ** 2)


class ParticleFilter(Filter):
    """Monte Carlo localisation.

    Parameters
    ----------
    world : World
        Supplies the landmarks, the arena (for global initialisation and for
        random particle injection) and the nominal noise levels.
    n_particles : int
    init : ``"global"`` or ``(mean, cov)``
        Uniform over the arena (the turtle could be anywhere), or Gaussian.
    motion_inflation : float
        Multiplies the motion-noise standard deviations used to *propagate*
        particles.  Some inflation (> 1) counters sample impoverishment.
    resample_threshold : float
        Resample when ``ESS < resample_threshold * N``.
    association : ``"known"`` or ``"unknown"``
        With ``"unknown"`` the filter ignores which landmark produced which
        measurement and uses ``p(z | x) = (1/M) sum_j N(z; h_j(x), R)``.
    recovery : ``None`` or ``(alpha_slow, alpha_fast)``
        Enable augmented MCL random-particle injection.
    """

    name = "PF"

    def __init__(
        self,
        world,
        n_particles=1000,
        init="global",
        motion_inflation=1.5,
        resample_threshold=0.5,
        resampler=systematic_resample,
        association="known",
        recovery=None,
        roughening=0.0,
        seed=None,
    ):
        self.world = world
        self.landmarks = world.landmarks
        self.N = int(n_particles)
        self.rng = np.random.default_rng(seed)
        self.sig_u = np.array([world.sigma_step, world.sigma_turn]) * motion_inflation
        self.R_sig = np.array([world.sigma_range, world.sigma_bearing])
        self.resample_threshold = resample_threshold
        self.resampler = resampler
        self.association = association
        self.recovery = recovery
        self.roughening = roughening
        self.w_slow = self.w_fast = None
        if isinstance(init, str) and init == "global":
            self.particles = world.sample_uniform_poses(self.N, self.rng)
        else:
            mean, cov = init
            self.particles = self.rng.multivariate_normal(mean, cov, self.N)
            self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.logw = np.full(self.N, -np.log(self.N))
        # diagnostics
        self.ess_history = []
        self.resampled_history = []
        self.injected_history = []

    # -- helpers ---------------------------------------------------------------
    @property
    def weights(self):
        return np.exp(self.logw)

    def log_likelihood(self, particles, z, visible):
        """``log p(z | x_i)`` for every particle (with normalising constants,
        which augmented MCL needs)."""
        lm = self.landmarks[visible]
        zv = z[visible]  # (Mv, 2)
        log_norm = -np.log(2 * np.pi * self.R_sig[0] * self.R_sig[1])
        if self.association == "known":
            d = zv[None] - range_bearing(particles, lm)  # (N, Mv, 2)
            d[..., 1] = wrap_angle(d[..., 1])
            maha = np.sum((d / self.R_sig) ** 2, axis=-1)  # (N, Mv)
            return np.sum(log_norm - 0.5 * maha, axis=1)
        # unknown association: every measurement may come from any landmark
        h = range_bearing(particles, self.landmarks)  # (N, M, 2)
        d = zv[None, :, None, :] - h[:, None, :, :]  # (N, Mv, M, 2)
        d[..., 1] = wrap_angle(d[..., 1])
        maha = np.sum((d / self.R_sig) ** 2, axis=-1)  # (N, Mv, M)
        ll = logsumexp(log_norm - 0.5 * maha, axis=2) - np.log(len(self.landmarks))
        return np.sum(ll, axis=1)

    # -- the filter ------------------------------------------------------------
    def predict(self, u):
        noise = self.rng.normal(0.0, self.sig_u, size=(self.N, 2))
        self.particles = motion_model(self.particles, np.asarray(u, dtype=float) + noise)

    def update(self, z, visible=None):
        z = np.asarray(z, dtype=float)
        if visible is None:
            visible = np.ones(len(z), dtype=bool)
        if np.any(visible):
            ll = self.log_likelihood(self.particles, z, visible)
            # a particle outside the arena is impossible
            ll = np.where(self.world.inside(self.particles[:, :2], margin=5.0), ll, -np.inf)
            # average likelihood (augmented MCL), using the *prior* weights
            log_avg = logsumexp(self.logw + ll)
            self.logw = self.logw + ll
            total = logsumexp(self.logw)
            if not np.isfinite(total):  # every particle is impossible: start over
                self.particles = self.world.sample_uniform_poses(self.N, self.rng)
                self.logw = np.full(self.N, -np.log(self.N))
            else:
                self.logw -= total
            if self.recovery is not None and np.isfinite(log_avg):
                # track short- and long-term averages of the *per-landmark*
                # likelihood (geometric mean) so the scale does not depend on
                # how many landmarks are visible
                w_avg = np.exp(log_avg / visible.sum())
                a_slow, a_fast = self.recovery
                if self.w_slow is None:
                    self.w_slow = self.w_fast = w_avg
                self.w_slow += a_slow * (w_avg - self.w_slow)
                self.w_fast += a_fast * (w_avg - self.w_fast)
        w = self.weights
        ess = effective_sample_size(w)
        self.ess_history.append(ess)
        p_inject = 0.0
        if self.recovery is not None and self.w_slow:
            p_inject = max(0.0, 1.0 - self.w_fast / self.w_slow)
        do_resample = ess < self.resample_threshold * self.N or p_inject > 0.0
        n_inj = 0
        if do_resample:
            idx = self.resampler(w, self.rng)
            self.particles = self.particles[idx]
            if self.roughening > 0:
                self.particles += self.rng.normal(0.0, self.roughening * np.array([1.0, 1.0, 0.01]), self.particles.shape)
                self.particles[:, 2] = wrap_angle(self.particles[:, 2])
            n_inj = self.rng.binomial(self.N, min(p_inject, 1.0))
            if n_inj:
                slots = self.rng.choice(self.N, n_inj, replace=False)
                self.particles[slots] = self.world.sample_uniform_poses(n_inj, self.rng)
            self.logw = np.full(self.N, -np.log(self.N))
        self.resampled_history.append(do_resample)
        self.injected_history.append(n_inj)

    # -- summaries -------------------------------------------------------------
    @property
    def mean(self):
        w = self.weights
        m = w @ self.particles
        m[2] = circular_mean(self.particles[:, 2], w)
        return m

    @property
    def cov(self):
        w = self.weights
        d = self.particles - self.mean
        d[:, 2] = wrap_angle(d[:, 2])
        return (w[:, None] * d).T @ d

    def snapshot(self):
        return self.particles.copy(), self.weights.copy()
