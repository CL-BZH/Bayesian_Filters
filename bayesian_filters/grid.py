"""A histogram (grid) Bayes filter over the turtle's *position*.

The belief is a probability table over the cells of a regular grid.  With the
heading known from the compass, the two steps of the Bayes filter become
image operations:

* **predict**: shift the table by the commanded displacement and blur it with
  the motion noise (a convolution) -- the Chapman-Kolmogorov equation;
* **update**: multiply cell-wise by the likelihood ``p(z | cell)`` and
  renormalise -- Bayes' rule.

It is exact up to the discretisation, it can represent *any* shape of belief
(rings, several bumps), and it costs ``O(cells)`` per step: fine in 2-D,
hopeless in high dimension.  That trade-off is what motivates both the Kalman
family (assume a Gaussian) and particle filters (sample the table instead of
storing it).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import Filter
from .world import range_bearing, wrap_angle


class GridFilter(Filter):
    name = "Grid"

    def __init__(self, world, cell=4.0, sensor="range", sigma_step=None, sigma_compass=None):
        self.world = world
        self.cell = float(cell)
        self.xs = np.arange(-world.half_width + cell / 2, world.half_width, cell)
        self.ys = np.arange(-world.half_height + cell / 2, world.half_height, cell)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)  # (ny, nx)
        self.belief = np.full(self.X.shape, 1.0 / self.X.size)
        self.sensor = sensor  # "range" (rings) or "range_bearing"
        self.sigma_step = world.sigma_step if sigma_step is None else sigma_step
        self.sigma_compass = world.sigma_compass if sigma_compass is None else sigma_compass
        self.heading = 0.0

    def predict(self, u, heading=None):
        if heading is not None:
            self.heading = heading
        step, h = u[0], self.heading
        # shift (in cells; rows are y, columns are x); mass leaving the arena is lost
        shifted = ndimage.shift(self.belief, (step * np.sin(h) / self.cell, step * np.cos(h) / self.cell), order=1, mode="constant")
        # blur: along-track step noise + across-track compass noise, isotropic approximation
        sigma = np.hypot(self.sigma_step, step * self.sigma_compass) / self.cell
        blurred = ndimage.gaussian_filter(shifted, sigma=max(sigma, 0.3), mode="constant")
        self.belief = blurred / blurred.sum()

    def likelihood(self, z, visible):
        cells = np.stack([self.X, self.Y, np.full(self.X.shape, self.heading)], axis=-1)
        h = range_bearing(cells, self.world.landmarks[visible])  # (ny, nx, Mv, 2)
        d = z[visible][None, None] - h
        ll = -0.5 * (d[..., 0] / self.world.sigma_range) ** 2
        if self.sensor == "range_bearing":
            ll = ll - 0.5 * (wrap_angle(d[..., 1]) / self.world.sigma_bearing) ** 2
        ll = ll.sum(axis=-1)
        return np.exp(ll - ll.max())

    def update(self, z, visible=None):
        z = np.asarray(z, dtype=float)
        if visible is None:
            visible = np.ones(len(z), dtype=bool)
        if not np.any(visible):
            return
        post = self.belief * self.likelihood(z, visible)
        s = post.sum()
        self.belief = post / s if s > 0 else np.full(self.X.shape, 1.0 / self.X.size)

    def predict_from_run(self, run, t):
        self.predict(run.controls[t], heading=run.z_compass[t])

    @property
    def mean(self):
        return np.array([np.sum(self.belief * self.X), np.sum(self.belief * self.Y)])

    @property
    def cov(self):
        m = self.mean
        dx, dy = self.X - m[0], self.Y - m[1]
        b = self.belief
        return np.array([[np.sum(b * dx * dx), np.sum(b * dx * dy)], [np.sum(b * dx * dy), np.sum(b * dy * dy)]])

    @property
    def mode(self):
        i = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return np.array([self.X[i], self.Y[i]])

    def snapshot(self):
        return self.belief.copy()
