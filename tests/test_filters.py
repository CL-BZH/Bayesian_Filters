import numpy as np
import pytest

from bayesian_filters import (
    EKF,
    UKF,
    CompassGpsKF,
    GridFilter,
    KalmanFilter,
    ParticleFilter,
    World,
    circular_mean,
    merwe_weights,
    motion_jacobians,
    motion_model,
    nees,
    position_rmse,
    range_bearing,
    range_bearing_jacobian,
    run_filter,
    sigma_points,
    simulate,
    systematic_resample,
    unscented_transform,
    wrap_angle,
)


@pytest.fixture(scope="module")
def world():
    return World.default()


@pytest.fixture(scope="module")
def run(world):
    return simulate(world, n_steps=200, seed=3)


# ---------------------------------------------------------------- world -----
def test_wrap_and_circular_mean():
    assert np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    m = circular_mean(np.radians([179.0, -179.0]))
    assert np.isclose(abs(m), np.pi)


def numeric_jacobian(f, x, eps=1e-6):
    fx = f(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        d = f(x + dx) - f(x - dx)
        J[:, i] = d / (2 * eps)
    return J


def test_motion_jacobians():
    x, u = np.array([3.0, -2.0, 0.7]), np.array([4.0, 0.1])
    F, V = motion_jacobians(x, u)
    assert np.allclose(F, numeric_jacobian(lambda s: motion_model(s, u), x), atol=1e-6)
    assert np.allclose(V, numeric_jacobian(lambda v: motion_model(x, v), u), atol=1e-6)


def test_range_bearing_jacobian(world):
    x = np.array([10.0, 20.0, -0.4])
    H = range_bearing_jacobian(x, world.landmarks)
    assert np.allclose(H, numeric_jacobian(lambda s: range_bearing(s, world.landmarks).ravel(), x), atol=1e-6)


def test_simulation_stays_in_arena(world):
    for seed in range(5):
        r = simulate(world, 400, seed=seed)
        assert np.all(world.inside(r.truth[:, :2]))


# ------------------------------------------------------ unscented -----------
@pytest.mark.parametrize("alpha,kappa", [(1.0, 0.0), (0.5, 1.0), (1.0, 2.0)])
def test_unscented_transform_is_exact_for_linear_maps(alpha, kappa):
    rng = np.random.default_rng(0)
    n = 4
    A = rng.normal(size=(n, n))
    P = A @ A.T + np.eye(n)
    m = rng.normal(size=n)
    lam, Wm, Wc = merwe_weights(n, alpha=alpha, kappa=kappa)
    assert np.isclose(Wm.sum(), 1.0)
    chi = sigma_points(m, P, lam)
    assert chi.shape == (2 * n + 1, n)
    B = rng.normal(size=(3, n))
    mean, cov, _ = unscented_transform(chi @ B.T, Wm, Wc)
    assert np.allclose(mean, B @ m)
    assert np.allclose(cov, B @ P @ B.T)


# ------------------------------------------------------------ KF ------------
def test_kalman_filter_scalar_closed_form():
    kf = KalmanFilter(np.array([0.0]), np.array([[4.0]]))
    kf.update(np.array([2.0]), np.eye(1), np.array([[1.0]]))
    # posterior of N(0, 4) x N(2, 1):  mean 8/5, variance 4/5
    assert np.isclose(kf.x[0], 1.6) and np.isclose(kf.P[0, 0], 0.8)


# ------------------------------------------------------- filters ------------
P0 = np.diag([10.0, 10.0, 0.05])


@pytest.mark.parametrize(
    "make,limit",
    [
        (lambda w, x0: CompassGpsKF.from_world(w, x0, P0), 4.0),
        (lambda w, x0: EKF.from_world(w, x0, P0), 2.0),
        (lambda w, x0: UKF.from_world(w, x0, P0), 2.0),
        (lambda w, x0: ParticleFilter(w, 1000, init=(x0, P0), seed=0), 2.5),
    ],
)
def test_filters_track_the_mower(world, run, make, limit):
    res = run_filter(make(world, run.truth[0]), run)
    assert not res.diverged
    assert position_rmse(res.means, run.truth, skip=10) < limit


@pytest.mark.parametrize("cls", [EKF, UKF])
def test_kalman_filters_are_consistent(world, cls):
    """Average NEES over several runs should be close to the state dimension."""
    vals = []
    for seed in range(6):
        r = simulate(world, 200, seed=seed)
        res = run_filter(cls.from_world(world, r.truth[0], P0), r)
        vals.append(np.nanmean(nees(res.means, res.covs, r.truth)[20:]))
    assert 1.5 < np.mean(vals) < 5.0


def test_pf_global_localisation(world, run):
    pf = ParticleFilter(world, 3000, init="global", roughening=0.5, seed=1)
    res = run_filter(pf, run)
    assert position_rmse(res.means, run.truth, skip=40) < 3.0


def test_pf_unknown_association_and_kidnap(world):
    r = simulate(world, 400, seed=2, kidnap=(200, (100.0, -100.0, np.pi)))
    pf = ParticleFilter(world, 3000, init="global", association="unknown", roughening=0.5, recovery=(0.001, 0.1), seed=2)
    res = run_filter(pf, r)
    err = np.hypot(*(res.means[-50:, :2] - r.truth[-50:, :2]).T)
    assert np.median(err) < 3.0


def test_systematic_resampling_counts():
    rng = np.random.default_rng(0)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    idx = systematic_resample(np.repeat(w, 250) / 250, rng)
    counts = np.bincount(idx // 250, minlength=4)
    assert np.all(np.abs(counts - 1000 * w) <= 1)


def test_grid_filter_localises(world, run):
    g = GridFilter(world, cell=4.0, sensor="range_bearing")
    res = run_filter(g, run)
    assert position_rmse(res.means, run.truth, skip=10) < 4.0
