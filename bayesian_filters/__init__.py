"""Bayesian_Filters: Bayesian filtering for a turtle, from Kalman to particles.

    >>> from bayesian_filters import World, simulate, EKF, run_filter
    >>> world = World.default()
    >>> run = simulate(world, n_steps=200, seed=1)
    >>> ekf = EKF.from_world(world, x0=run.truth[0], P0=np.eye(3))
    >>> result = run_filter(ekf, run)
"""

from .base import Filter, FilterResult, run_filter
from .ekf import EKF
from .grid import GridFilter
from .kf import CompassGpsKF, KalmanFilter
from .metrics import nees, nees_bounds, position_error, position_rmse
from .pf import ParticleFilter, effective_sample_size, multinomial_resample, systematic_resample
from .ukf import UKF, merwe_weights, sigma_points, unscented_transform
from .world import (
    Run,
    World,
    circular_mean,
    figure_eight_control,
    motion_jacobians,
    motion_model,
    range_bearing,
    range_bearing_jacobian,
    simulate,
    wrap_angle,
)

__all__ = [
    "World", "Run", "simulate", "figure_eight_control", "wrap_angle", "circular_mean",
    "motion_model", "motion_jacobians", "range_bearing", "range_bearing_jacobian",
    "Filter", "FilterResult", "run_filter",
    "KalmanFilter", "CompassGpsKF", "EKF", "UKF", "ParticleFilter", "GridFilter",
    "merwe_weights", "sigma_points", "unscented_transform",
    "systematic_resample", "multinomial_resample", "effective_sample_size",
    "position_error", "position_rmse", "nees", "nees_bounds",
]
