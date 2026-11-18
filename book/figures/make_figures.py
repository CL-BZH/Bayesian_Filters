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



# ---------------------------------------------------------------- explanatory figures
def _gauss(x, m, v):
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


def fig_bayes_cycle():
    """One predict/update cycle of the Bayes filter, exactly, on a 1-D grid."""
    x = np.linspace(-10, 20, 3001)
    dx = x[1] - x[0]
    bel = 0.6 * _gauss(x, 0.0, 1.0) + 0.4 * _gauss(x, 4.0, 0.6)
    d, q = 5.0, 1.5
    kern = _gauss(np.arange(-10, 10 + dx, dx), 0.0, q) * dx
    shifted = np.interp(x - d, x, bel, left=0, right=0)
    pred = np.convolve(shifted, kern, mode="same")
    lik = _gauss(x, 9.6, 0.8)
    post = pred * lik
    post /= post.sum() * dx
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.27), sharey=True)
    a1, a2, a3 = axes
    a1.fill_between(x, bel, color=viz.MUTED, alpha=0.35, lw=0)
    a1.plot(x, bel, color=viz.INK_2, lw=1.2, label=r"$\mathrm{bel}(x_t)$")
    a1.plot(x, pred, color=viz.COLORS["EKF"], lw=1.6, label=r"$\overline{\mathrm{bel}}(x_{t+1})$")
    a1.annotate("", xy=(9.0, 0.30), xytext=(4.5, 0.30), arrowprops=dict(arrowstyle="->", color=viz.INK_2))
    a1.text(6.75, 0.33, "shift $+$ blur", ha="center", color=viz.INK_2, fontsize=7.5)
    a1.set_title("predict", fontsize=9)
    a2.plot(x, pred, color=viz.COLORS["EKF"], lw=1.6, label=r"$\overline{\mathrm{bel}}(x_{t+1})$")
    a2.plot(x, lik / lik.max() * 0.3, color=viz.COLORS["UKF"], lw=1.4, ls="--", label=r"$p(z_{t+1}\mid x)$ (scaled)")
    a2.set_title("multiply by the likelihood", fontsize=9)
    a3.plot(x, pred, color=viz.COLORS["EKF"], lw=1.0, alpha=0.5)
    a3.fill_between(x, post, color=viz.COLORS["PF"], alpha=0.3, lw=0)
    a3.plot(x, post, color=viz.COLORS["PF"], lw=1.6, label=r"$\mathrm{bel}(x_{t+1})$")
    a3.set_title("renormalise", fontsize=9)
    for a in axes:
        a.set_xlim(-4, 14); a.set_xlabel("$x$"); a.set_yticks([]); a.set_ylim(0, 0.62)
        a.legend(fontsize=6.8, loc="upper left")
    plt.tight_layout()
    save(fig, "ch02_bayes_cycle")


def fig_gaussian():
    """Conditioning a joint Gaussian, and the scalar Kalman update."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.36), gridspec_kw=dict(width_ratios=[1, 1.15]))
    mx, my, sx, sy, rho = 0.0, 0.0, 1.0, 1.3, 0.75
    Sxy = rho * sx * sy
    X, Y = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-4, 4, 300))
    C = np.array([[sx**2, Sxy], [Sxy, sy**2]])
    Ci = np.linalg.inv(C)
    Q = Ci[0, 0] * X**2 + 2 * Ci[0, 1] * X * Y + Ci[1, 1] * Y**2
    a1.contour(X, Y, Q, levels=[1, 4, 9], colors=[viz.MUTED], linewidths=0.8)
    ys = np.linspace(-4, 4, 2)
    a1.plot(mx + Sxy / sy**2 * ys, ys, color=viz.COLORS["EKF"], lw=1.2, label=r"$\mathbb{E}[X\mid Y=y]$")
    y0 = 2.0
    a1.axhline(y0, color=viz.COLORS["UKF"], lw=1.0, ls="--")
    cm, cv = mx + Sxy / sy**2 * (y0 - my), sx**2 - Sxy**2 / sy**2
    xs = np.linspace(-3.2, 3.2, 300)
    a1.plot(xs, y0 + 1.2 * _gauss(xs, cm, cv), color=viz.COLORS["PF"], lw=1.4, label=r"density of $X\mid Y=y$")
    a1.text(-3.0, y0 + 0.12, "$Y=y$", color=viz.COLORS["UKF"], fontsize=8)
    a1.set_xlabel("$x$"); a1.set_ylabel("$y$"); a1.legend(fontsize=7, loc="lower right")
    a1.set_title("conditioning a joint Gaussian", fontsize=9)
    x = np.linspace(-5, 6, 600)
    a2.plot(x, _gauss(x, 0, 4), color=viz.INK_2, lw=1.4, label=r"prior $\mathcal{N}(0,4)$")
    a2.plot(x, _gauss(x, 2, 1), color=viz.COLORS["UKF"], lw=1.4, ls="--", label=r"measurement $z=2$, $R=1$")
    a2.fill_between(x, _gauss(x, 1.6, 0.8), color=viz.COLORS["KF"], alpha=0.25, lw=0)
    a2.plot(x, _gauss(x, 1.6, 0.8), color=viz.COLORS["KF"], lw=1.8, label=r"posterior $\mathcal{N}(1.6,0.8)$")
    a2.set_yticks([]); a2.set_xlabel("$x$"); a2.legend(fontsize=7, loc="upper left")
    a2.set_title("the scalar Kalman update ($K=0.8$)", fontsize=9)
    plt.tight_layout()
    save(fig, "ch03_gaussian")


def _cos_example():
    mu, sig = 0.5, 0.8
    rng = np.random.default_rng(0)
    th = rng.normal(mu, sig, 200000)
    return mu, sig, np.cos(th)


def fig_linearisation(with_ut, name):
    mu, sig, y = _cos_example()
    t = np.linspace(-2.6, 3.6, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.36), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a1.plot(t, np.cos(t), color=viz.INK, lw=1.4, label=r"$g(\theta)=\cos\theta$")
    slope = -np.sin(mu)
    a1.plot(t, np.cos(mu) + slope * (t - mu), color=viz.COLORS["EKF"], lw=1.2, label="tangent at the mean (EKF)")
    dens = _gauss(t, mu, sig**2)
    a1.fill_between(t, -1.6, -1.6 + 0.9 * dens / dens.max(), color=viz.MUTED, alpha=0.4, lw=0)
    a1.text(mu + 1.05, -1.45, r"$\theta\sim\mathcal{N}(0.5,\,0.8^2)$", fontsize=7.5, color=viz.INK_2)
    m_e, v_e = np.cos(mu), slope**2 * sig**2
    if with_ut:
        lam, Wm, Wc = merwe_weights(1)
        chi = mu + np.array([0.0, sig, -sig]) * np.sqrt(1 + lam)
        Z = np.cos(chi)
        zm = Wm @ Z
        Pzz = Wc @ (Z - zm) ** 2
        Pxz = Wc @ ((chi - mu) * (Z - zm))
        Ht = Pxz / sig**2
        a1.plot(t, zm + Ht * (t - mu), color=viz.COLORS["UKF"], lw=1.2, label="regression line (UKF)")
        a1.scatter(chi, Z, s=28, fc="white", ec=viz.COLORS["UKF"], lw=1.4, zorder=6, label="sigma points")
        for c in chi:
            a1.plot([c, c], [-1.6, np.cos(c)], color=viz.COLORS["UKF"], lw=0.6, ls=":")
        omega = Pzz - Ht**2 * sig**2
    a1.set_ylim(-1.65, 2.2); a1.set_xlim(t[0], t[-1]); a1.set_xlabel(r"$\theta$")
    a1.legend(fontsize=6.8, loc="upper right")
    ys = np.linspace(-1.3, 1.5, 500)
    a2.hist(y, bins=120, density=True, color=viz.MUTED, alpha=0.5, label=r"true density of $g(\theta)$")
    a2.axvline(y.mean(), color=viz.INK, lw=1.0, ls="--", label=f"true mean {y.mean():.2f}")
    a2.plot(ys, _gauss(ys, m_e, v_e), color=viz.COLORS["EKF"], lw=1.4, label=f"EKF: mean {m_e:.2f}, sd {np.sqrt(v_e):.2f}")
    if with_ut:
        a2.plot(ys, _gauss(ys, zm, Pzz), color=viz.COLORS["UKF"], lw=1.4, label=f"UKF: mean {zm:.2f}, sd {np.sqrt(Pzz):.2f}")
    a2.set_yticks([]); a2.set_xlabel(r"$y=\cos\theta$"); a2.set_xlim(-1.3, 1.5)
    a2.set_ylim(0, 2.6)
    a2.legend(fontsize=6.5, loc="upper left")
    plt.tight_layout()
    save(fig, name)
    out = dict(true_mean=y.mean(), true_sd=y.std(), ekf_mean=m_e, ekf_sd=np.sqrt(v_e))
    if with_ut:
        out.update(ukf_mean=zm, ukf_sd=np.sqrt(Pzz), H_tilde=Ht, slope=slope, omega=omega)
    print(name, {k: round(float(v), 3) for k, v in out.items()})


def fig_sigma_points():
    m = np.array([0.0, 0.0])
    P = np.array([[2.0, 1.1], [1.1, 1.2]])
    lam, Wm, Wc = merwe_weights(2)
    chi = sigma_points(m, P, lam)
    fig, ax = plt.subplots(figsize=(W * 0.45, W * 0.38))
    viz.cov_ellipse(ax, m, P, n_sigma=1, ec=viz.INK_2, lw=1.0)
    viz.cov_ellipse(ax, m, P, n_sigma=2, ec=viz.MUTED, lw=0.8, ls="--")
    L = np.linalg.cholesky(P)
    for i in range(2):
        ax.annotate("", xy=L[:, i] * np.sqrt(2 + lam), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=viz.COLORS["UKF"], lw=1.0))
    ax.scatter(chi[:, 0], chi[:, 1], s=36, fc="white", ec=viz.COLORS["UKF"], lw=1.5, zorder=5)
    for j, c in enumerate(chi):
        lab = r"$\chi_0$" if j == 0 else (rf"$\chi_{{+{j}}}$" if j <= 2 else rf"$\chi_{{-{j-2}}}$")
        ax.annotate(lab, c, xytext=(5, 4), textcoords="offset points", fontsize=8, color=viz.INK_2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-2.6, 2.6)
    save(fig, "ch05_sigma_points")


def fig_importance():
    x = np.linspace(-6, 8, 800)
    target = 0.55 * _gauss(x, 0.0, 0.5) + 0.45 * _gauss(x, 3.5, 0.8)
    prop = _gauss(x, 1.5, 6.0)
    rng = np.random.default_rng(3)
    xs = rng.normal(1.5, np.sqrt(6.0), 40)
    w = (0.55 * _gauss(xs, 0, 0.5) + 0.45 * _gauss(xs, 3.5, 0.8)) / _gauss(xs, 1.5, 6.0)
    w /= w.sum()
    fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.32))
    ax.plot(x, target, color=viz.INK, lw=1.4, label=r"target $\pi$")
    ax.plot(x, prop, color=viz.MUTED, lw=1.4, ls="--", label="proposal $q$")
    ax.vlines(xs, 0, w * 2.2, color=viz.COLORS["PF"], lw=1.6, label=r"samples from $q$, height $\propto$ weight")
    ax.plot(xs, np.full_like(xs, -0.012), "|", color=viz.INK_2, ms=6)
    ax.set_yticks([]); ax.set_xlabel("$x$"); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(-0.03, None)
    save(fig, "ch06_importance")


def fig_amcl():
    world = World.default()
    rk = simulate(world, n_steps=300, seed=7, kidnap=(200, (100.0, -100.0, np.pi)))
    pf = ParticleFilter(world, 3000, init="global", association="unknown", roughening=0.5,
                        recovery=(0.001, 0.1), seed=7)
    slow, fast, err = [], [], []
    for t in range(rk.n_steps):
        pf.predict(rk.controls[t])
        pf.update(rk.z_rb[t], rk.visible[t])
        slow.append(pf.w_slow); fast.append(pf.w_fast)
        err.append(np.hypot(*(pf.mean[:2] - rk.truth[t + 1, :2])))
    steps = np.arange(1, rk.n_steps + 1)
    inj = np.array(pf.injected_history)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W * 0.8, W * 0.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.3, 1]))
    a1.plot(steps, slow, color=viz.INK_2, lw=1.4, label=r"$w_{\mathrm{slow}}$")
    a1.plot(steps, fast, color=viz.COLORS["UKF"], lw=1.2, label=r"$w_{\mathrm{fast}}$")
    a1.set_ylabel("average likelihood\n(per landmark)"); a1.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.25))
    a2.bar(steps, inj, color=viz.COLORS["PF"], width=1.0, label="injected particles")
    a2b = a2.twinx()
    a2b.plot(steps, err, color=viz.INK, lw=1.0, ls="--")
    a2b.set_ylabel("position error (dashed)", fontsize=8)
    a2.set_ylabel("injected"); a2.set_xlabel("step")
    for a in (a1, a2):
        a.axvline(200, color=viz.MUTED, lw=0.8, ls=":")
        a.set_xlim(150, 260)
    plt.tight_layout()
    save(fig, "ch06_amcl")
    print("amcl injected before/after", inj[150:200].sum(), inj[200:230].sum())


def fig_helicoid():
    """Poses reachable in one step lie on a helicoid; their shadow on the lawn is an arc band."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    s0, d0 = 4.0, 0.0          # commanded step and turn, from the pose (0, 0, 0)
    sig_s, sig_d = 0.6, 0.35   # exaggerated noise, for visibility
    rng = np.random.default_rng(1)
    eps = rng.normal(0, [sig_s, sig_d], size=(500, 2))
    dist, phi = s0 + eps[:, 0], d0 + eps[:, 1]
    px, py = dist * np.cos(phi), dist * np.sin(phi)
    D, PHI = np.meshgrid(np.linspace(0, 6.2, 30), np.linspace(-1.1, 1.1, 40))
    X, Y = D * np.cos(PHI), D * np.sin(PHI)
    fig = plt.figure(figsize=(W, W * 0.52))
    ax = fig.add_axes([0.0, 0.0, 0.60, 1.0], projection="3d")
    ax2 = fig.add_axes([0.72, 0.14, 0.27, 0.74])
    ax.plot_surface(X, Y, PHI, color=viz.COLORS["EKF"], alpha=0.18, linewidth=0, antialiased=True)
    ax.plot_wireframe(X, Y, PHI, color=viz.COLORS["EKF"], linewidth=0.25, alpha=0.5, rstride=4, cstride=5)
    z0 = -1.1
    ax.scatter(px, py, np.full_like(px, z0), s=1.5, color=viz.MUTED, alpha=0.6, depthshade=False)
    # the "banana": image on the lawn of the 95% ellipse of (distance, heading)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    k = np.sqrt(5.991)                       # 95% quantile of chi-squared with 2 d.o.f.
    tt = np.linspace(0, 2 * np.pi, 200)
    bd, bphi = s0 + k * sig_s * np.cos(tt), d0 + k * sig_d * np.sin(tt)
    bx, by = bd * np.cos(bphi), bd * np.sin(bphi)
    banana = Poly3DCollection([list(zip(bx, by, np.full_like(bx, z0)))],
                              facecolor=viz.MUTED, alpha=0.15, edgecolor="none")
    ax.add_collection3d(banana)
    ax.plot(bx, by, np.full_like(bx, z0), color=viz.INK_2, lw=1.0)
    # the same shadow seen from above: the lawn
    arc = np.linspace(-1.25, 1.25, 100)
    ax2.plot(s0 * np.cos(arc), s0 * np.sin(arc), color=viz.INK_2, lw=0.7, ls="--")
    ax2.scatter(px, py, s=3, color=viz.MUTED, alpha=0.7, lw=0)
    ax2.fill(bx, by, color=viz.MUTED, alpha=0.15, lw=0)
    ax2.plot(bx, by, color=viz.INK_2, lw=1.0)
    ax2.annotate("", xy=(s0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=viz.INK, lw=1.0))
    ax2.scatter([0], [0], s=30, color=viz.INK, zorder=3)
    ax2.text(-0.1, -0.45, "start", fontsize=7.5, color=viz.INK, ha="center")
    ax2.text(1.9, 0.18, "intended step", fontsize=7.5, color=viz.INK, ha="center")
    ax2.text(-1.0, 3.9, "circle of radius $s$", fontsize=7.5, color=viz.INK_2, ha="left")
    ax2.set_aspect("equal"); ax2.set_xlim(-1.2, 6.5); ax2.set_ylim(-4.5, 4.5)
    ax2.set_xlabel("$p^x$"); ax2.set_ylabel("$p^y$")
    ax2.set_title("the lawn, seen from above", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax.scatter(px, py, phi, s=2.5, color=viz.COLORS["UKF"], alpha=0.8, depthshade=False)
    for f in (-0.7, 0.0, 0.7):  # straight horizontal rulings: fixed heading, varying distance
        ax.plot([0, 6.2 * np.cos(f)], [0, 6.2 * np.sin(f)], [f, f], color=viz.INK_2, lw=1.0)
    ax.scatter([0], [0], [0], s=30, color=viz.INK, depthshade=False)
    ax.text(0, 0, 0.12, "start pose", fontsize=7.5, color=viz.INK)
    ax.set_xlabel("$p^x$"); ax.set_ylabel("$p^y$"); ax.set_zlabel(r"$\theta$")
    ax.set_xlim(0, 6.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(z0, 1.1)
    ax.view_init(elev=18, azim=-35)
    ax.tick_params(labelsize=7)
    save(fig, "ch01_helicoid")


def fig_coupling():
    """Position and heading are coupled: one step (exaggerated noise) and 25 steps (real noise)."""
    world = World.default()
    rng = np.random.default_rng(2)
    # (a) one step, exaggerated noise, positions coloured by heading
    eps = rng.normal(0, [0.6, 0.35], size=(1500, 2))
    d, phi = 4.0 + eps[:, 0], eps[:, 1]
    # (b), (c) straight path, real noise
    n = 4000
    x = np.zeros((n, 3))
    u = np.array([4.0, 0.0])
    x1 = None
    for t in range(1, 26):
        x = motion_model(x, u + rng.normal(0, [world.sigma_step, world.sigma_turn], size=(n, 2)))
        if t == 1:
            x1 = x.copy()
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.33), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    a1, a2, a3 = axes
    sc = a1.scatter(d * np.cos(phi), d * np.sin(phi), c=phi, cmap="coolwarm", s=3, lw=0, vmin=-1, vmax=1)
    a1.scatter([0], [0], s=20, color=viz.INK)
    for f in (-0.6, 0.0, 0.6):
        a1.plot([0, 6 * np.cos(f)], [0, 6 * np.sin(f)], color=viz.INK_2, lw=0.6, ls=":")
    a1.set_aspect("equal"); a1.set_xlabel("$p^x$"); a1.set_ylabel("$p^y$")
    a1.set_title("one step: colour = heading", fontsize=8)
    cb = fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.03); cb.set_label(r"$\theta$", fontsize=8); cb.ax.tick_params(labelsize=6.5)
    sc2 = a2.scatter(x[:, 0], x[:, 1], c=x[:, 2], cmap="coolwarm", s=2, lw=0, vmin=-0.45, vmax=0.45)
    a2.set_xlabel("$p^x$"); a2.set_ylabel("$p^y$")
    a2.set_title("25 steps: colour = heading", fontsize=8)
    cb2 = fig.colorbar(sc2, ax=a2, fraction=0.046, pad=0.03); cb2.ax.tick_params(labelsize=6.5)
    z = lambda v: (v - v.mean()) / v.std()
    a3.scatter(z(x[:, 2]), z(x[:, 1]), s=2, lw=0, color=viz.COLORS["EKF"], alpha=0.4, label="after 25 steps")
    a3.scatter(z(x1[:, 2]), z(x1[:, 1]), s=2, lw=0, color=viz.COLORS["UKF"], alpha=0.8, label="after 1 step")
    c25 = np.corrcoef(x[:, 1], x[:, 2])[0, 1]
    c1 = np.corrcoef(x1[:, 1], x1[:, 2])[0, 1]
    a3.set_xlim(-4, 4); a3.set_ylim(-4, 4); a3.set_aspect("equal")
    a3.set_xlabel(r"heading error $e_\theta/\sigma$"); a3.set_ylabel(r"sideways error $e_\perp/\sigma$")
    a3.set_title(f"correlation {c1:.3f} (1 step), {c25:.2f} (25)", fontsize=8)
    a3.legend(fontsize=6.5, loc="upper left", markerscale=4)
    plt.tight_layout()
    save(fig, "ch01_coupling")
    print("coupling corr", round(c1, 3), round(c25, 3))

if __name__ == "__main__":
    fig_transition()
    fig_helicoid()
    fig_coupling()
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
    fig_bayes_cycle()
    fig_gaussian()
    fig_linearisation(False, "ch04_linearisation")
    fig_linearisation(True, "ch05_regression")
    fig_sigma_points()
    fig_importance()
    fig_amcl()
