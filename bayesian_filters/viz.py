"""Matplotlib figures shared by the notebooks (no turtle graphics needed).

Colour follows the filter, never its rank, and is the same in every notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9b9a94"
GRID = "#e6e5e0"

#: fixed colour per filter (categorical slots 1, 2, 3 validate together in all pairs)
COLORS = {
    "EKF": "#2a78d6",
    "UKF": "#eb6834",
    "PF": "#1baf7a",
    "KF": "#6250d6",
    "Grid": "#2a78d6",
    "truth": INK,
}
SEQ_CMAP = "Blues"


def style():
    """A quiet house style: recessive grid and axes, ink-coloured text."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.size": 9.5,
        }
    )


def color(name):
    return COLORS.get(name, INK_2)


def plot_world(ax, world, label_landmarks=True):
    """Lawn outline and numbered landmarks."""
    ax.add_patch(
        Rectangle(
            (-world.half_width, -world.half_height), 2 * world.half_width, 2 * world.half_height,
            fill=False, ec=MUTED, lw=1.0, ls="--",
        )
    )
    lm = world.landmarks
    ax.scatter(lm[:, 0], lm[:, 1], marker="*", s=170, c=INK_2, ec="white", lw=1.0, zorder=5, label="landmarks")
    if label_landmarks:
        for j, (x, y) in enumerate(lm):
            ax.annotate(f"L{j}", (x, y), xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=8.5)
    m = 12
    ax.set_xlim(-world.half_width - m, world.half_width + m)
    ax.set_ylim(-world.half_height - m, world.half_height + m)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_path(ax, states, name="truth", **kw):
    kw.setdefault("color", color(name))
    kw.setdefault("lw", 1.4 if name == "truth" else 1.8)
    if name == "truth":
        kw.setdefault("ls", "--")
    kw.setdefault("label", name)
    return ax.plot(states[:, 0], states[:, 1], **kw)


def plot_pose(ax, pose, name="truth", size=9.0, **kw):
    """A pose as a small arrow."""
    kw.setdefault("color", color(name))
    dx, dy = size * np.cos(pose[2]), size * np.sin(pose[2])
    return ax.annotate(
        "", xy=(pose[0] + dx, pose[1] + dy), xytext=(pose[0], pose[1]),
        arrowprops=dict(arrowstyle="-|>", color=kw["color"], lw=1.6, mutation_scale=12), zorder=6,
    )


def cov_ellipse(ax, mean, cov, n_sigma=2.0, name="EKF", **kw):
    """The ``n_sigma`` ellipse of the position part of ``cov``."""
    vals, vecs = np.linalg.eigh(np.asarray(cov)[:2, :2])
    vals = np.clip(vals, 0, None)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_sigma * np.sqrt(vals[::-1])
    kw.setdefault("ec", color(name))
    kw.setdefault("fc", "none")
    kw.setdefault("lw", 1.4)
    e = Ellipse(mean[:2], w, h, angle=angle, **kw)
    ax.add_patch(e)
    return e


def plot_particles(ax, particles, weights=None, name="PF", max_points=3000, **kw):
    p = particles
    if len(p) > max_points:
        p = p[:: len(p) // max_points]
        weights = None if weights is None else weights[:: len(weights) // max_points]
    s = 4.0 if weights is None else 2.0 + 60.0 * weights / max(weights.max(), 1e-300)
    kw.setdefault("color", color(name))
    kw.setdefault("alpha", 0.35)
    return ax.scatter(p[:, 0], p[:, 1], s=s, lw=0, **kw)


def plot_error_over_time(ax, results, truth, ylabel="position error"):
    from .metrics import position_error

    for r in results:
        ax.plot(position_error(r.means, truth), color=color(r.name), lw=1.6, label=r.name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if len(results) > 1:
        ax.legend()
    return ax


def label_line_end(ax, y_values, name, x=None, dx=4):
    """Direct label at the right end of a line (in text ink, not series colour)."""
    y = np.asarray(y_values, dtype=float)
    lo, hi = ax.get_ylim()
    ok = np.where(np.isfinite(y) & (y >= lo) & (y <= hi))[0]
    if len(ok) == 0:
        return
    i = ok[-1]
    ax.annotate(name, (i if x is None else x[i], y[i]), xytext=(dx, 0), textcoords="offset points", va="center", color=INK_2, fontsize=8.5)
