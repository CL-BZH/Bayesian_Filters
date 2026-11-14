"""Regenerate the animations shown in the README (docs/media/*.gif).

    python scripts/make_media.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from bayesian_filters import EKF, ParticleFilter, World, run_filter, simulate, viz

OUT = ROOT / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
viz.style()


def global_localisation_gif(path, n_frames=90):
    """Particles starting uniform over the arena, converging on the turtle."""
    world = World.default()
    run = simulate(world, n_steps=n_frames, seed=1)
    pf = ParticleFilter(world, 3000, init="global", roughening=0.5, seed=0)
    res = run_filter(pf, run, keep_snapshots=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    viz.plot_world(ax, world)
    ax.set_xlabel(""); ax.set_ylabel("")
    scat = ax.scatter([], [], s=3, color=viz.color("PF"), alpha=0.4, lw=0)
    (path_line,) = ax.plot([], [], color=viz.INK, ls="--", lw=1.2)
    turtle_marker = ax.scatter([], [], marker="o", s=50, fc="lime", ec=viz.INK, zorder=8)
    title = ax.set_title("")

    def draw(t):
        parts, _ = res.snapshots[t]
        scat.set_offsets(parts[:, :2])
        path_line.set_data(run.truth[: t + 1, 0], run.truth[: t + 1, 1])
        turtle_marker.set_offsets(run.truth[t, :2])
        title.set_text(f"Particle filter, global localisation — step {t}")
        return scat, path_line, turtle_marker, title

    anim = FuncAnimation(fig, draw, frames=range(0, n_frames + 1), blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def kidnap_gif(path, start=190, stop=260):
    """EKF vs PF (unknown data association, augmented MCL) when the turtle is kidnapped."""
    world = World.default()
    run = simulate(world, n_steps=stop, seed=1, kidnap=(200, (100.0, -100.0, np.pi)))
    P0 = np.diag([10.0, 10.0, 0.05])
    r_ekf = run_filter(EKF.from_world(world, run.truth[0], P0, association="nearest"), run)
    pf = ParticleFilter(world, 3000, init="global", association="unknown", roughening=0.5, recovery=(0.001, 0.1), seed=1)
    r_pf = run_filter(pf, run, keep_snapshots=True)
    fig, ax = plt.subplots(figsize=(4.6, 5.1))
    viz.plot_world(ax, world)
    ax.set_xlabel(""); ax.set_ylabel("")
    scat = ax.scatter([], [], s=7, color=viz.color("PF"), alpha=0.5, lw=0)
    truth_m = ax.scatter([], [], marker="o", s=50, fc="lime", ec=viz.INK, zorder=8, label="turtle")
    ekf_m = ax.scatter([], [], marker="P", s=70, c=viz.color("EKF"), zorder=9, label="EKF estimate")
    pf_m = ax.scatter([], [], marker="X", s=70, c=viz.color("PF"), ec="white", lw=0.6, zorder=9, label="PF estimate")
    fig.legend(loc="lower center", ncol=3, fontsize=7.5)
    fig.subplots_adjust(bottom=0.14)
    title = ax.set_title("")

    def draw(t):
        parts, _ = r_pf.snapshots[t]
        scat.set_offsets(parts[:, :2])
        truth_m.set_offsets(run.truth[t, :2])
        ekf_m.set_offsets(r_ekf.means[t, :2])
        pf_m.set_offsets(r_pf.means[t, :2])
        msg = "  KIDNAPPED!" if 200 < t <= 206 else ""
        title.set_text(f"Unknown landmark identities — step {t}{msg}")
        return scat, truth_m, ekf_m, pf_m, title

    anim = FuncAnimation(fig, draw, frames=range(start, stop + 1), blit=False)
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


if __name__ == "__main__":
    global_localisation_gif(OUT / "pf_global_localisation.gif")
    kidnap_gif(OUT / "kidnapped_ekf_vs_pf.gif")
    for p in sorted(OUT.glob("*.gif")):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.1f} MB")
