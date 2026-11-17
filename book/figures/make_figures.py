"""Regenerate every figure of the book (book/figures/*.pdf).

    python book/figures/make_figures.py          # from the repository root

The experiments use the same seeds and settings as the notebooks, so the
numbers quoted in the book match the notebooks' outputs.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesian_filters import (EKF, UKF, CompassGpsKF, GridFilter, ParticleFilter, World,
                              merwe_weights, motion_jacobians, motion_model, multinomial_resample,
                              nees, nees_bounds, position_error, run_filter, sigma_points, simulate,
                              systematic_resample, unscented_transform, viz)
from bayesian_filters.ukf import pose_mean, pose_residual

viz.style()
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
W = 6.3  # text width in inches


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- chapter 1
def fig_transition():
    world = World.default()
    rng = np.random.default_rng(0)
    n, T = 1000, 25
    x = np.zeros((n, 3))
    u = np.array([4.0, np.pi / 50])
    clouds = {}
    for t in range(1, T + 1):
        x = motion_model(x, u + rng.normal(0, [world.sigma_step, world.sigma_turn], size=(n, 2)))
        if t in (5, 15, 25):
            clouds[t] = x.copy()
    nominal = [np.zeros(3)]
    for _ in range(T):
        nominal.append(motion_model(nominal[-1], u))
    nominal = np.array(nominal)
    fig, ax = plt.subplots(figsize=(W * 0.7, W * 0.5))
    ax.plot(nominal[:, 0], nominal[:, 1], color=viz.INK, ls="--", lw=1.0, label="commanded path")
    for t, c in clouds.items():
        ax.scatter(c[:, 0], c[:, 1], s=2, alpha=0.35, color=viz.COLORS["EKF"], lw=0)
        ax.annotate(f"after {t} steps", c[:, :2].mean(0), xytext=(12, -12), textcoords="offset points", color=viz.INK_2)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(loc="upper left")
    save(fig, "ch01_transition")


def fig_dead_reckoning():
    world = World.default()
    run = simulate(world, n_steps=400, seed=1)
    dr = [run.truth[0]]
    for u in run.controls:
        dr.append(motion_model(dr[-1], u))
    dr = np.array(dr)
    err = position_error(dr, run.truth)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.42), gridspec_kw=dict(width_ratios=[1, 1.3]))
    viz.plot_world(a1, world)
    viz.plot_path(a1, run.truth, "truth", label="true path")
    a1.plot(dr[:, 0], dr[:, 1], color=viz.COLORS["UKF"], lw=1.4, label="dead reckoning")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5)
    a1.set_xlabel("")
    a2.plot(err, color=viz.COLORS["UKF"], lw=1.6)
    a2.set_xlabel("step"); a2.set_ylabel("position error"); a2.set_ylim(bottom=0)
    plt.tight_layout()
    save(fig, "ch01_dead_reckoning")


# ---------------------------------------------------------------- chapter 2
def fig_grid():
    world = World(World.default().landmarks[:2])
    run = simulate(world, n_steps=60, seed=1)
    grid = GridFilter(world, cell=4.0, sensor="range")
    snap_at = [0, 1, 4, 9, 40]
    snaps = {}
    for t in range(run.n_steps):
        grid.predict_from_run(run, t)
        grid.update_from_run(run, t)
        if t in snap_at:
            snaps[t] = grid.belief.copy()
    from matplotlib.colors import PowerNorm
    ext = [-world.half_width, world.half_width, -world.half_height, world.half_height]
    fig, axes = plt.subplots(1, 5, figsize=(W, W * 0.27))
    for ax, t in zip(axes, snap_at):
        ax.imshow(snaps[t], origin="lower", extent=ext, cmap=viz.SEQ_CMAP, interpolation="nearest",
                  norm=PowerNorm(0.4, vmin=0, vmax=snaps[t].max()))
        ax.scatter(*world.landmarks.T, marker="*", s=60, c=viz.INK_2, ec="white", lw=0.5)
        ax.scatter(*run.truth[t + 1, :2], marker="o", s=110, fc="none", ec="#e34948", lw=1.2)
        ax.set_title(f"measurement {t + 1}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    plt.tight_layout()
    save(fig, "ch02_grid_bimodal")


# ---------------------------------------------------------------- chapter 3
def fig_kf_nees():
    world = World.default()
    P0 = np.diag([10.0, 10.0, 0.05])
    N = 50
    runs = [simulate(world, n_steps=200, seed=100 + k) for k in range(N)]
    lo, hi = nees_bounds(2, N)
    cases = {"correct $R$": 1.0, r"$R$ too small ($\sigma/4$)": 0.25, r"$R$ too large ($4\sigma$)": 4.0}
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.32), sharey=True)
    for ax, (label, k) in zip(axes, cases.items()):
        ne = []
        for i, r in enumerate(runs):
            x0 = r.truth[0] + np.r_[np.random.default_rng(i).multivariate_normal([0, 0], P0[:2, :2]), 0.0]
            f = CompassGpsKF(x0, P0, world.sigma_step, world.sigma_compass, k * world.sigma_gps)
            rr = run_filter(f, r)
            ne.append(nees(rr.means, rr.covs, r.truth))
        ne = np.mean(ne, axis=0)
        ax.axhspan(lo, hi, color=viz.GRID)
        ax.plot(np.arange(1, len(ne)), ne[1:], color=viz.COLORS["KF"], lw=1.2)
        ax.set_yscale("log"); ax.set_title(label, fontsize=8.5); ax.set_xlabel("step")
    axes[0].set_ylabel("average NEES")
    plt.tight_layout()
    save(fig, "ch03_kf_nees")


# ---------------------------------------------------------------- chapters 4-5
def arc_experiment():
    rng = np.random.default_rng(0)
    m0, P0 = np.zeros(3), np.diag([1.0, 1.0, 0.8**2])
    u = np.array([4.0, 0.0])

    def g(X):
        for _ in range(10):
            X = motion_model(X, u)
        return X

    samples = g(rng.multivariate_normal(m0, P0, 5000))
    m_e, P_e = m0.copy(), P0.copy()
    for _ in range(10):
        F, _ = motion_jacobians(m_e, u)
        m_e, P_e = motion_model(m_e, u), F @ P_e @ F.T
    lam, Wm, Wc = merwe_weights(3)
    chi = sigma_points(m0, P0, lam)
    m_u, P_u, _ = unscented_transform(g(chi), Wm, Wc, pose_mean, pose_residual)
    return samples, (m_e, P_e), (m_u, P_u, g(chi))


def fig_arc(with_ut, name):
    samples, (m_e, P_e), (m_u, P_u, chi) = arc_experiment()
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.48))
    ax.scatter(samples[:, 0], samples[:, 1], s=1.5, color=viz.MUTED, alpha=0.4, lw=0, label="samples")
    mc_m, mc_P = samples[:, :2].mean(0), np.cov(samples[:, :2].T)
    ax.scatter(*mc_m, s=30, c=viz.INK, zorder=6, label="true mean")
    viz.cov_ellipse(ax, mc_m, mc_P, ec=viz.INK, ls="--", label=r"true $2\sigma$")
    ax.scatter(*m_e[:2], marker="P", s=50, c=viz.COLORS["EKF"], zorder=6, label="EKF mean")
    viz.cov_ellipse(ax, m_e, P_e, name="EKF", lw=1.6, label=r"EKF $2\sigma$")
    if with_ut:
        ax.scatter(*m_u[:2], marker="X", s=50, c=viz.COLORS["UKF"], zorder=6, label="UKF mean")
        viz.cov_ellipse(ax, m_u, P_u, name="UKF", lw=1.6, label=r"UKF $2\sigma$")
        ax.scatter(*chi[:, :2].T, s=18, fc="white", ec=viz.COLORS["UKF"], lw=1.2, zorder=7, label="sigma points")
    ax.set_aspect("equal"); ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    save(fig, name)


def transient_experiment():
    world_lr = World.default(max_range=110)
    P0 = np.diag([30.0**2, 30.0**2, 0.8**2])
    N, T = 20, 120
    out = {"EKF": ([], []), "UKF": ([], [])}
    for k in range(N):
        r = simulate(world_lr, n_steps=T, seed=k)
        x0 = r.truth[0] + np.random.default_rng(100 + k).multivariate_normal(np.zeros(3), P0)
        for cls in (EKF, UKF):
            rr = run_filter(cls.from_world(world_lr, x0, P0), r)
            out[cls.name][0].append(nees(rr.means, rr.covs, r.truth))
            out[cls.name][1].append(position_error(rr.means, r.truth))
    return out, nees_bounds(3, N), T


def fig_transient(names, fname):
    out, (lo, hi), T = transient_experiment()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.34))
    a1.axhspan(lo, hi, color=viz.GRID, label="95% band")
    for name in names:
        ne, er = np.array(out[name][0]), np.array(out[name][1])
        a1.plot(np.arange(1, T + 1), ne.mean(0)[1:], color=viz.color(name), label=name)
        a2.plot(np.median(er, 0), color=viz.color(name), label=name)
    a1.set_yscale("log"); a1.set_xlabel("step"); a1.set_ylabel("average NEES (20 runs)"); a1.legend()
    a2.set_ylim(0, 20); a2.set_xlabel("step"); a2.set_ylabel("median position error")
    if len(names) > 1:
        a2.legend()
    plt.tight_layout()
    save(fig, fname)


# ---------------------------------------------------------------- chapter 6
def fig_ess():
    world = World.default()
    run = simulate(world, n_steps=400, seed=1)
    P0 = np.diag([10.0, 10.0, 0.05])
    sis = ParticleFilter(world, 1000, init=(run.truth[0], P0), resample_threshold=0.0, seed=0)
    pf = ParticleFilter(world, 1000, init=(run.truth[0], P0), resample_threshold=0.5, seed=0)
    r_sis, r_pf = run_filter(sis, run), run_filter(pf, run)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.34))
    a1.plot(np.arange(1, 401), sis.ess_history, color=viz.MUTED, lw=1.4, label="no resampling (SIS)")
    a1.plot(np.arange(1, 401), pf.ess_history, color=viz.COLORS["PF"], lw=1.0, label=r"resample if ESS $<N/2$")
    a1.set_yscale("log"); a1.set_xlabel("step"); a1.set_ylabel("ESS"); a1.legend(fontsize=7.5)
    a2.plot(position_error(r_sis.means, run.truth), color=viz.MUTED, lw=1.4, label="SIS")
    a2.plot(position_error(r_pf.means, run.truth), color=viz.COLORS["PF"], lw=1.0, label="particle filter")
    a2.set_ylim(0, 30); a2.set_xlabel("step"); a2.set_ylabel("position error"); a2.legend(fontsize=7.5)
    plt.tight_layout()
    save(fig, "ch06_ess")


def fig_resampling():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(50))
    N = len(w)
    cm, cs = [], []
    for _ in range(2000):
        cm.append(np.bincount(multinomial_resample(w, rng), minlength=N))
        cs.append(np.bincount(systematic_resample(w, rng), minlength=N))
    cm, cs = np.array(cm), np.array(cs)
    order = np.argsort(w)
    fig, ax = plt.subplots(figsize=(W * 0.6, W * 0.32))
    nw = N * w[order]
    xs = np.linspace(0, nw.max(), 200)
    ax.plot(xs, np.sqrt(xs * (1 - xs / N)), color=viz.MUTED, lw=0.8, ls="--")
    f = xs - np.floor(xs)
    ax.plot(xs, np.sqrt(f * (1 - f)), color=viz.COLORS["PF"], lw=0.8, ls="--")
    ax.plot(nw, cm.std(0)[order], "o", ms=3, color=viz.MUTED, label="multinomial")
    ax.plot(nw, cs.std(0)[order], "o", ms=3, color=viz.COLORS["PF"], label="systematic")
    ax.set_xlabel(r"expected number of copies $Nw^{(i)}$"); ax.set_ylabel("std of the copies"); ax.legend()
    save(fig, "ch06_resampling")


def fig_global():
    world = World.default()
    run = simulate(world, n_steps=400, seed=1)
    pf = ParticleFilter(world, 5000, init="global", seed=0)
    r = run_filter(pf, run, keep_snapshots=True)
    fig, axes = plt.subplots(1, 4, figsize=(W, W * 0.28))
    for ax, t in zip(axes, (0, 1, 5, 30)):
        viz.plot_world(ax, world, label_landmarks=False)
        parts, wts = r.snapshots[t]
        viz.plot_particles(ax, parts, None if t == 0 else wts, alpha=0.4)
        ax.scatter(*run.truth[t, :2], s=25, fc="lime", ec=viz.INK, lw=0.8, zorder=8)
        ax.set_title("prior" if t == 0 else f"after step {t}", fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "ch06_global")


def fig_modes():
    square = np.array([[-80, -80], [80, -80], [80, 80], [-80, 80]], float)
    wsq = World(square, sigma_range=15.0, sigma_bearing=0.25)
    rsq = simulate(wsq, n_steps=300, seed=1, x0=(60.0, 30.0, 1.0))
    pf = ParticleFilter(wsq, 20000, init="global", association="unknown", seed=1)
    r = run_filter(pf, rsq, keep_snapshots=True)
    fig, axes = plt.subplots(1, 4, figsize=(W, W * 0.3))
    for ax, t in zip(axes, (1, 10, 30, 300)):
        viz.plot_world(ax, wsq, label_landmarks=False)
        parts, wts = r.snapshots[t]
        viz.plot_particles(ax, parts, wts, alpha=0.5)
        ax.scatter(*rsq.truth[t, :2], s=25, fc="lime", ec=viz.INK, lw=0.8, zorder=8)
        masses = []
        for k in range(4):
            c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
            q = np.array([[c, -s], [s, c]]) @ rsq.truth[t, :2]
            masses.append(wts[np.hypot(*(parts[:, :2] - q).T) < 25].sum())
        ax.set_title(f"step {t}\n" + " ".join(f"{m:.2f}" for m in masses), fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "ch06_modes")


def fig_kidnap():
    world = World.default()
    rk = simulate(world, n_steps=400, seed=7, kidnap=(200, (100.0, -100.0, np.pi)))
    fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.33))
    for label, rec, col in [("plain", None, viz.MUTED), ("augmented MCL", (0.001, 0.1), viz.COLORS["PF"])]:
        rr = run_filter(ParticleFilter(world, 3000, init="global", association="unknown",
                                       roughening=0.5, recovery=rec, seed=7), rk)
        ax.plot(position_error(rr.means, rk.truth), color=col, lw=1.3, label=label)
    ax.axvline(200, color=viz.MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 260); ax.set_xlabel("step"); ax.set_ylabel("position error"); ax.legend()
    save(fig, "ch06_kidnap")


if __name__ == "__main__":
    fig_transition()
    fig_dead_reckoning()
    fig_grid()
    fig_kf_nees()
    fig_arc(False, "ch04_arc")
    fig_arc(True, "ch05_arc")
    fig_transient(["EKF"], "ch04_transient")
    fig_transient(["EKF", "UKF"], "ch05_transient")
    fig_ess()
    fig_resampling()
    fig_global()
    fig_modes()
    fig_kidnap()
