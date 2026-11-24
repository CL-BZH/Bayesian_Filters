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

from bayesian_filters import (EKF, UKF, CompassGpsKF, DeadReckoning, GridFilter, ParticleFilter,
                              PathFollower, World, distance_to_path, simulate_closed_loop,
                              stripes_path, position_rmse, merwe_weights, motion_jacobians, motion_model, multinomial_resample,
                              nees, nees_bounds, position_error, run_filter, sigma_points, simulate,
                              systematic_resample, unscented_transform, viz)
from bayesian_filters.ukf import pose_mean, pose_residual
from bayesian_filters.world import wrap_angle

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


def potato_radius(phi):
    """Distance from the centre to the edge of the potato-shaped lawn, in the direction phi."""
    return 118 + 22 * np.cos(2 * phi + 0.4) + 14 * np.sin(3 * phi) + 7 * np.cos(5 * phi - 1.0)


TREE = (np.array([35.0, -25.0]), 16.0)  # a tree in the lawn: centre and radius


def on_potato_lawn(x, y):
    """True where (x, y) is on the lawn: inside the potato, outside the tree."""
    inside = np.hypot(x, y) <= potato_radius(np.arctan2(y, x))
    return inside & (np.hypot(x - TREE[0][0], y - TREE[0][1]) > TREE[1])


def fig_potato():
    from matplotlib.colors import ListedColormap, PowerNorm
    from matplotlib.patches import Circle
    phi = np.linspace(-np.pi, np.pi, 400)
    edge = potato_radius(phi) * np.array([np.cos(phi), np.sin(phi)])
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.48))
    # (a) the mask: fraction of each 16-unit cell that is on the lawn (8 x 8 sample points per cell)
    cell, half = 16.0, 160.0
    edges = np.arange(-half, half + cell, cell)
    sub = (np.arange(8) + 0.5) / 8 * cell
    frac = np.zeros((len(edges) - 1, len(edges) - 1))
    for i, y0 in enumerate(edges[:-1]):
        for j, x0 in enumerate(edges[:-1]):
            X, Y = np.meshgrid(x0 + sub, y0 + sub)
            frac[i, j] = on_potato_lawn(X, Y).mean()
    kind = np.where(frac == 1, 2, np.where(frac > 0, 1, 0))
    ax = axes[0]
    ax.imshow(kind, origin="lower", extent=[-half, half, -half, half], interpolation="nearest",
              cmap=ListedColormap(["white", "#cdeadb", "#7fc8a4"]), vmin=0, vmax=2)
    for e in edges:
        ax.axhline(e, color=viz.GRID, lw=0.5); ax.axvline(e, color=viz.GRID, lw=0.5)
    ax.plot(*edge, color=viz.INK, lw=1.2)
    ax.add_patch(Circle(TREE[0], TREE[1], fc="none", ec=viz.INK, lw=1.0))
    ax.annotate("tree", TREE[0], color=viz.INK_2, ha="center", va="center", fontsize=8)
    ax.set_title("(a) the mask", fontsize=8)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    # (b) a predicted belief next to the edge, before and after masking (4-unit cells)
    c4 = 4.0
    centres = np.arange(-half + c4 / 2, half, c4)
    X, Y = np.meshgrid(centres, centres)
    mu, sd = np.array([136.0, 15.0]), 14.0
    pred = np.exp(-((X - mu[0]) ** 2 + (Y - mu[1]) ** 2) / (2 * sd**2))
    pred /= pred.sum()
    lawn = on_potato_lawn(X, Y)
    kept = pred * lawn
    lost = 1 - kept.sum()
    post = kept / kept.sum()
    ax = axes[1]
    ax.imshow(np.ma.masked_where(post == 0, post), origin="lower", extent=[-half, half, -half, half],
              cmap=viz.SEQ_CMAP, interpolation="nearest", norm=PowerNorm(0.6, vmin=0, vmax=post.max()))
    ax.contour(X, Y, pred, levels=[pred.max() * np.exp(-2), pred.max() * np.exp(-0.5)],
               colors=[viz.INK_2], linestyles="--", linewidths=0.8)
    ax.plot(*edge, color=viz.INK, lw=1.2)
    ax.add_patch(Circle(TREE[0], TREE[1], fc="none", ec=viz.INK, lw=1.0))
    ax.set_title("(b) a predicted belief at the edge, masked", fontsize=8)
    ax.set_xlim(40, 175); ax.set_ylim(-55, 80); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    plt.tight_layout()
    save(fig, "ch02_potato")
    print(f"potato lawn: {100 * lost:.1f}% of the predicted belief falls off the lawn")


# ---------------------------------------------------------------- appendix E
def in_potato(x, y):
    """Inside the outer edge of the potato lawn (the tree is ignored)."""
    return np.hypot(x, y) <= potato_radius(np.arctan2(y, x))


def classify_cell(x0, y0, s):
    """'in', 'out' or 'mixed' for the square [x0, x0+s] x [y0, y0+s] and the potato lawn.

    The outer edge is a closed curve around the origin, so the square lies inside it when its
    sides do, and outside it when its sides do and it does not contain the origin (65 points per
    side are tested). The test against the round tree is exact."""
    t = np.linspace(0, 1, 65)
    one = np.ones_like(t)
    xs = np.r_[x0 + s * t, (x0 + s) * one, x0 + s * t[::-1], x0 * one]
    ys = np.r_[y0 * one, y0 + s * t, (y0 + s) * one, y0 + s * t[::-1]]
    inside = in_potato(xs, ys)
    has_origin = x0 <= 0 <= x0 + s and y0 <= 0 <= y0 + s
    potato = "in" if inside.all() else ("out" if not inside.any() and not has_origin else "mixed")
    (cx, cy), R = TREE
    nearest = np.hypot(cx - np.clip(cx, x0, x0 + s), cy - np.clip(cy, y0, y0 + s))
    farthest = max(np.hypot(cx - a, cy - b) for a in (x0, x0 + s) for b in (y0, y0 + s))
    tree = "apart" if nearest > R else ("inside" if farthest <= R else "mixed")
    if potato == "out" or tree == "inside":
        return "out"
    return "in" if potato == "in" and tree == "apart" else "mixed"


def quadtree(x0, y0, s, h, leaves):
    """Split the square while it is mixed and larger than h; append its leaves (x0, y0, s, kind)."""
    kind = classify_cell(x0, y0, s)
    if kind == "mixed" and s > h * (1 + 1e-9):
        for dx in (0, s / 2):
            for dy in (0, s / 2):
                quadtree(x0 + dx, y0 + dy, s / 2, h, leaves)
    else:
        leaves.append((x0, y0, s, kind))
    return leaves


def fig_quadtree():
    from matplotlib.patches import Circle, Rectangle
    phi = np.linspace(-np.pi, np.pi, 200001)
    r = potato_radius(phi)
    x, y = r * np.cos(phi), r * np.sin(phi)
    edge_length = np.hypot(np.diff(x), np.diff(y)).sum() + 2 * np.pi * TREE[1]
    print(f"quadtree: edge length P = {edge_length:.1f} (outer edge and tree), "
          f"lawn area = {0.5 * np.trapezoid(r**2, phi) - np.pi * TREE[1]**2:.0f}")
    rows = []
    for m in range(3, 8):
        h = 320 / 2**m
        leaves = quadtree(-160.0, -160.0, 320.0, h, [])
        n = {k: sum(l[3] == k for l in leaves) for k in ("in", "mixed", "out")}
        lawn_cells = sum((l[2] / h) ** 2 for l in leaves if l[3] == "in") + n["mixed"]
        bound = 1 + 12 * edge_length / h + 12 * 2 * m
        rows.append((h, len(leaves), lawn_cells, 4**m, bound))
        print(f"quadtree h={h:g} (m={m}): {len(leaves)} leaves ({n['in']} in, {n['mixed']} mixed, "
              f"{n['out']} out); uniform grid: {4**m} cells, {lawn_cells:.0f} of them meet the lawn; "
              f"bound 1+12P/h+24m = {bound:.0f}; mixed leaves x h / P = {n['mixed'] * h / edge_length:.2f}")
        if m == 5:
            drawn = leaves
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.47), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    fill = {"in": "#7fc8a4", "mixed": "#cdeadb", "out": "white"}
    for x0, y0, s, kind in drawn:
        ax.add_patch(Rectangle((x0, y0), s, s, fc=fill[kind], ec=viz.INK_2, lw=0.35))
    ax.plot(x, y, color=viz.INK, lw=1.1)
    ax.add_patch(Circle(TREE[0], TREE[1], fc="none", ec=viz.INK, lw=0.9))
    ax.set_title(f"(a) the quadtree of the lawn, $h=10$ ({len(drawn)} leaves)", fontsize=8)
    ax.set_xlim(-160, 160); ax.set_ylim(-160, 160); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax = axes[1]
    hs = np.array([row[0] for row in rows])
    series = [("uniform grid, whole square", 3, viz.MUTED, "s"),
              ("uniform grid, cells meeting the lawn", 2, viz.COLORS["UKF"], "^"),
              ("quadtree leaves", 1, viz.COLORS["EKF"], "o")]
    for label, col, colour, marker in series:
        ax.plot(hs, [row[col] for row in rows], color=colour, lw=1.5, marker=marker, ms=4, label=label)
    ax.plot(hs, [row[4] for row in rows], color=viz.INK_2, lw=1.0, ls="--", label="upper bound $1+12P/h+24m$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(hs); ax.set_xticklabels([f"{v:g}" for v in hs])
    ax.invert_xaxis()
    ax.set_xlabel("smallest cell side $h$"); ax.set_ylabel("number of cells")
    ax.set_title("(b) cells needed, as $h$ shrinks", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    plt.tight_layout()
    save(fig, "appE_quadtree")


# ---------------------------------------------------------------- appendix F
def fig_shift_1d():
    """Sharing in one dimension: where the mass of one cell goes; then what one cell receives, k = 0 and k = 1."""
    from matplotlib.patches import Rectangle
    f = 0.7
    blue, orange, ink = viz.COLORS["EKF"], viz.COLORS["UKF"], viz.INK
    fig, ax1 = plt.subplots(figsize=(W * 0.8, W * 0.27))
    # (a) the mass of one cell
    ax = ax1
    for i in range(3):
        ax.add_patch(Rectangle((i, 0), 1, 1, fc="none", ec=ink, lw=1.0))
        ax.text(i + 0.5, -0.1, f"cell {i}", ha="center", va="top", fontsize=8)
    ax.add_patch(Rectangle((f, 0), 1 - f, 1, fc=blue, alpha=0.25, lw=0))
    ax.add_patch(Rectangle((1, 0), f, 1, fc=blue, alpha=0.55, lw=0))
    ax.add_patch(Rectangle((f, 0), 1, 1, fc="none", ec=blue, lw=1.6, ls="--"))
    ax.annotate("", (f, 1.12), (0, 1.12), arrowprops=dict(arrowstyle="->", color=ink, lw=0.9))
    ax.text(f / 2, 1.17, f"move by $f={f}$", ha="center", va="bottom", fontsize=8)
    ax.text(f + (1 - f) / 2, 0.5, f"{1 - f:.1f}", ha="center", va="center", fontsize=8)
    ax.text(1 + f / 2, 0.5, f"{f}", ha="center", va="center", fontsize=8)
    ax.text(f + (1 - f) / 2, -0.38, "$1-f$", ha="center", va="top", fontsize=8, color=blue)
    ax.text(1 + f / 2, -0.38, "$f$", ha="center", va="top", fontsize=8, color=blue)
    ax.text(3.15, 0.5, "the moved copy of cell 0 (dashed)\ncovers $1-f$ of cell 0 and $f$ of cell 1:\n"
            "cell 0 keeps $1-f$ of its mass,\ncell 1 gets $f$", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.1, 6.3); ax.set_ylim(-0.7, 1.45)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    save(fig, "appF_shift_1d")
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(W * 0.8, W * 0.72))
    # (b), (c) what one cell receives, for k = 0 and k = 1
    from matplotlib.patches import Patch
    from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
    b = [0.1, 0.5, 0.3, 0.1]
    names = ["$j-2$", "$j-1$", "$j$", "$j+1$"]
    sub = {0: ("j", "j-1"), 1: ("j-1", "j-2")}       # (from 1-f, from f) for each k
    ytop = 1.55
    prop = lambda c: dict(color=c, fontsize=8.5)
    for ax, k, title in ((ax2, 0, "(a) a move by $f=0.7$ ($k=0$)"),
                         (ax3, 1, "(b) a move by $1+f=1.7$ ($k=1$)")):
        for i in range(4):
            ax.add_patch(Rectangle((i, ytop), 1, 1, fc="none", ec=ink, lw=1.0))
            ax.text(i + 0.5, ytop + 0.5, f"$b={b[i]}$", ha="center", va="center", fontsize=8)
            ax.text(i + 0.5, ytop + 1.08, names[i], ha="center", va="bottom", fontsize=8)
            ax.add_patch(Rectangle((i, 0), 1, 1, fc="none", ec=ink, lw=1.0))
        ax.text(-0.1, ytop + 0.5, "before", ha="right", va="center", fontsize=8)
        ax.text(-0.1, 0.5, "after", ha="right", va="center", fontsize=8)
        # the two cells whose moved copies overlap cell j (index 2): j-k-1 (orange) and j-k (blue)
        io, ib = 1 - k, 2 - k
        bo, bb = b[io], b[ib]
        ax.add_patch(Rectangle((io + k + f, 0), 1, 1, fc="none", ec=orange, lw=1.6, ls="--"))
        ax.add_patch(Rectangle((ib + k + f, 0), 1, 1, fc="none", ec=blue, lw=1.6, ls="--"))
        ax.add_patch(Rectangle((2, 0), f, 1, fc=orange, alpha=0.45, lw=0))
        ax.add_patch(Rectangle((2 + f, 0), 1 - f, 1, fc=blue, alpha=0.45, lw=0))
        ax.annotate("", (io + k + f + 0.5, 1.02), (io + 0.5, ytop - 0.02),
                    arrowprops=dict(arrowstyle="->", color=orange, lw=1.1))
        ax.annotate("", (ib + k + f + 0.5, 1.02), (ib + 0.5, ytop - 0.02),
                    arrowprops=dict(arrowstyle="->", color=blue, lw=1.1))
        so, sb = sub[k][1], sub[k][0]
        ax.legend(handles=[Patch(fc=orange, alpha=0.45, ec="none",
                                 label=f"from cell ${so}$: $f\\,b_{{{so}}}={f}\\times{bo}={f * bo:.2f}$"),
                           Patch(fc=blue, alpha=0.45, ec="none",
                                 label=f"from cell ${sb}$: $(1-f)\\,b_{{{sb}}}={1 - f:.1f}\\times{bb}={(1 - f) * bb:.2f}$")],
                  loc="upper left", bbox_to_anchor=(0.08, 0.02), fontsize=8, frameon=False, handlelength=1.4)
        new = f * bo + (1 - f) * bb
        line1 = TextArea("new mass of cell $j$:", textprops=prop(ink))
        line2 = HPacker(children=[TextArea("$b'_j=$", textprops=prop(ink)),
                                  TextArea(f"$f\\,b_{{{so}}}$", textprops=prop(orange)),
                                  TextArea("$+$", textprops=prop(ink)),
                                  TextArea(f"$(1-f)\\,b_{{{sb}}}$", textprops=prop(blue))], pad=0, sep=2)
        line3 = HPacker(children=[TextArea("$=$", textprops=prop(ink)),
                                  TextArea(f"${f * bo:.2f}$", textprops=prop(orange)),
                                  TextArea("$+$", textprops=prop(ink)),
                                  TextArea(f"${(1 - f) * bb:.2f}$", textprops=prop(blue)),
                                  TextArea(f"$={new:.2f}$", textprops=prop(ink))], pad=0, sep=2)
        box = VPacker(children=[line1, line2, line3], pad=0, sep=3, align="left")
        ax.add_artist(AnnotationBbox(box, (4.25, 0.5), box_alignment=(0, 0.5), frameon=False))
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlim(-0.8, 7.2); ax.set_ylim(-0.15, ytop + 1.45)
    for ax in (ax2, ax3):
        ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    save(fig, "appF_shift_receive")


def fig_shift_2d():
    """Sharing in two dimensions: shares are areas; the left column keeps 1 - fx."""
    from matplotlib.patches import Rectangle
    fx, fy = 0.7, 0.4
    blue, ink, grey = viz.COLORS["EKF"], viz.INK, viz.MUTED
    fig, axes = plt.subplots(1, 2, figsize=(W * 0.75, W * 0.42))
    for ax, title in zip(axes, ("(a) the shares are areas", "(b) the left column keeps $1-f_x$")):
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((i, j), 1, 1, fc="none", ec=ink, lw=1.0))
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.1, 2.1); ax.set_ylim(-0.3, 2.2)
    ax = axes[0]
    parts = [((fx, fy), 1 - fx, 1 - fy, 0.20), ((1, fy), fx, 1 - fy, 0.40),
             ((fx, 1), 1 - fx, fy, 0.30), ((1, 1), fx, fy, 0.55)]
    for (x0, y0), w, h, a in parts:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=blue, alpha=a, lw=0))
        ax.text(x0 + w / 2, y0 + h / 2, f"{w * h:.2f}", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((fx, fy), 1, 1, fc="none", ec=blue, lw=1.6, ls="--"))
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=grey, lw=2.2))
    ax.text(0.35, 0.2, "start", ha="center", fontsize=8, color=grey)
    ax.text(0.0, -0.22, f"$f_x={fx}$, $f_y={fy}$", fontsize=8, ha="left")
    ax = axes[1]
    ax.add_patch(Rectangle((fx, fy), 1 - fx, 1, fc=blue, alpha=0.45, lw=0))
    ax.add_patch(Rectangle((fx, fy), 1, 1, fc="none", ec=blue, lw=1.6, ls="--"))
    ax.text(fx + (1 - fx) / 2, fy + 0.5, f"{1 - fx:.1f}", ha="center", va="center", fontsize=8)
    ax.annotate("", (1, fy - 0.08), (fx, fy - 0.08), arrowprops=dict(arrowstyle="<->", color=ink, lw=0.8))
    ax.text(fx + (1 - fx) / 2, fy - 0.15, "$1-f_x$", ha="center", va="top", fontsize=8)
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    save(fig, "appF_shift_2d")


def fig_shift_spread():
    """Repeated interpolated shifts: the belief spreads, the mean is right; rounding is sharp but biased."""
    from scipy import ndimage
    f = 0.87
    cells = np.arange(16)
    b = np.zeros(16); b[2] = 1.0
    hist_i, hist_r = [b], [b]
    for t in range(10):
        hist_i.append(ndimage.shift(hist_i[-1], f, order=1, mode="constant"))
        hist_r.append(ndimage.shift(hist_r[-1], f, order=0, mode="constant"))
    blue, orange = viz.COLORS["EKF"], viz.COLORS["UKF"]
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, t in zip(axes[:2], (1, 5)):
        ax.bar(cells, hist_i[t], width=0.8, color=blue)
        mean = np.sum(cells * hist_i[t]); var = np.sum((cells - mean) ** 2 * hist_i[t])
        ax.axvline(2 + t * f, ymax=0.68, color=viz.INK, lw=0.9, ls="--")
        ax.set_title(f"(%s) after {t} shift{'s' if t > 1 else ''} of {f} cell" % ("a" if t == 1 else "b"),
                     fontsize=8.5)
        ax.text(0.97, 0.93, f"variance {var:.4f}\n$t\\,f(1-f)={t * f * (1 - f):.4f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
        ax.set_xlim(-0.5, 12.5); ax.set_ylim(0, 1.05); ax.set_xlabel("cell", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("probability", fontsize=8)
    ax = axes[2]
    ts = np.arange(11)
    ax.plot(ts, 2 + ts * f, color=viz.INK, lw=1.1, ls=(0, (3, 2)), label="true position", zorder=5)
    ax.plot(ts, [np.sum(cells * h) for h in hist_r], color=orange, lw=1.5, marker="o", ms=3,
            label="rounding: mean")
    ax.plot(ts, [np.sum(cells * h) for h in hist_i], color=blue, lw=1.5, marker="s", ms=3,
            label="sharing: mean")
    ax.set_xlabel("number of shifts", fontsize=8); ax.set_ylabel("cell", fontsize=8)
    ax.set_title("(c) rounding drifts, sharing does not", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_ylim(0, 17); ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    save(fig, "appF_shift_spread")
    print(f"shift spread: after 10 shifts of {f}: true {2 + 10 * f:.2f}, rounding {np.sum(cells * hist_r[10]):.2f}, "
          f"sharing {np.sum(cells * hist_i[10]):.2f}")


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


def fig_kf_track():
    """The compass-GPS Kalman filter on the standard run: path and error."""
    world = World.default()
    run = simulate(world, seed=1)
    f = CompassGpsKF(run.truth[0], np.diag([10.0, 10.0, 0.05]), world.sigma_step, world.sigma_compass,
                     world.sigma_gps)
    rr = run_filter(f, run)
    T = run.n_steps
    e_gps = np.linalg.norm(run.z_gps[:T] - run.truth[1:, :2], axis=1)
    e_kf = np.linalg.norm(rr.means[1:, :2] - run.truth[1:, :2], axis=1)
    print("kf track: RMS gps %.2f  RMS kf %.2f" % (np.sqrt(np.mean(e_gps**2)), np.sqrt(np.mean(e_kf**2))))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.42), gridspec_kw=dict(width_ratios=[1, 1.25]))
    n = 120                                   # first part of the run, for legibility
    a1.scatter(run.z_gps[:n, 0], run.z_gps[:n, 1], s=5, color=viz.MUTED, alpha=0.7, lw=0, label="GPS fixes")
    a1.plot(run.truth[: n + 1, 0], run.truth[: n + 1, 1], color=viz.INK, lw=1.2, label="true path")
    a1.plot(rr.means[: n + 1, 0], rr.means[: n + 1, 1], color=viz.COLORS["KF"], lw=1.3, label="Kalman filter")
    a1.set_aspect("equal"); a1.set_xlabel("$x$"); a1.set_ylabel("$y$")
    a1.set_title("first %d steps" % n, fontsize=8.5)
    a1.legend(fontsize=7, loc="best")
    steps = np.arange(1, T + 1)
    a2.plot(steps, e_gps, color=viz.MUTED, lw=0.7, label="GPS fix alone")
    a2.plot(steps, e_kf, color=viz.COLORS["KF"], lw=1.2, label="Kalman filter")
    a2.set_xlabel("step"); a2.set_ylabel("position error")
    a2.set_title("whole run", fontsize=8.5)
    a2.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    save(fig, "ch03_kf_track")


def fig_riccati():
    """The scalar variance recursion converges to its fixed point from any start."""
    q, r = 0.16, 64.0
    P_inf = (q + np.sqrt(q * q + 4 * q * r)) / 2 - q
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.34))
    for P0, ls in ((40.0, "-"), (0.0, "--")):
        P, seq = P0, [P0]
        for _ in range(60):
            Pb = P + q
            P = Pb * r / (Pb + r)
            seq.append(P)
        ax.plot(seq, color=viz.COLORS["KF"], lw=1.3, ls=ls, label="$P_0=%g$" % P0)
    ax.axhline(P_inf, color=viz.INK_2, lw=0.8, ls=":")
    ax.text(60, P_inf + 0.8, r"$P_\infty=%.2f$" % P_inf, fontsize=7.5, ha="right", color=viz.INK_2)
    ax.set_xlabel("step"); ax.set_ylabel("updated variance $P_t$")
    ax.set_ylim(-1, 20)
    ax.legend(fontsize=7.5)
    plt.tight_layout()
    save(fig, "ch03_riccati")


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


def fig_ekf_track():
    """The EKF on the standard run: the lawn, and one error component inside its 2-sigma band."""
    world = World.default()
    run = simulate(world, seed=1)
    f = EKF.from_world(world, run.truth[0], np.diag([1.0, 1.0, 0.05]))
    rr = run_filter(f, run)
    err = rr.means - run.truth
    err[:, 2] = wrap_angle(err[:, 2])
    sd = np.sqrt(np.einsum("tii->ti", rr.covs))
    inside = np.mean(np.abs(err[1:]) <= 2 * sd[1:], axis=0)
    print("ekf track: RMSE %.2f, inside 2 sigma (x, y, theta): %s"
          % (np.sqrt(np.mean(np.sum(err[1:, :2] ** 2, axis=1))), np.round(inside, 3)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, W * 0.42), gridspec_kw=dict(width_ratios=[1, 1.3]))
    n = 150
    a1.plot(run.truth[: n + 1, 0], run.truth[: n + 1, 1], color=viz.INK, lw=1.2, label="true path")
    a1.plot(rr.means[: n + 1, 0], rr.means[: n + 1, 1], color=viz.COLORS["EKF"], lw=1.2, ls="--", label="EKF")
    lm = np.asarray(world.landmarks)
    a1.scatter(lm[:, 0], lm[:, 1], marker="*", s=70, color=viz.COLORS["UKF"], zorder=5, label="landmarks")
    a1.set_aspect("equal"); a1.set_xlabel("$x$"); a1.set_ylabel("$y$")
    a1.set_title("first %d steps" % n, fontsize=8.5)
    a1.legend(fontsize=7, loc="center")
    steps = np.arange(len(err))
    a2.fill_between(steps, -2 * sd[:, 0], 2 * sd[:, 0], color=viz.COLORS["EKF"], alpha=0.18, lw=0,
                    label=r"$\pm2\sigma$ claimed by the filter")
    a2.plot(steps, err[:, 0], color=viz.COLORS["EKF"], lw=0.9, label="error on $p^x$")
    a2.axhline(0, color=viz.INK_2, lw=0.5)
    a2.set_xlabel("step"); a2.set_ylabel("error on $p^x$")
    a2.set_title("whole run", fontsize=8.5)
    a2.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    save(fig, "ch04_ekf_track")


def fig_ekf_banana():
    """One step with the exaggerated noise of the helicoid figure: true 95% region and the EKF's ellipse."""
    s0, sig_s, sig_d = 4.0, 0.6, 0.35
    rng = np.random.default_rng(1)
    eps = rng.normal(0, [sig_s, sig_d], size=(1500, 2))
    dist, phi = s0 + eps[:, 0], eps[:, 1]
    px, py = dist * np.cos(phi), dist * np.sin(phi)
    k = np.sqrt(5.991)                       # 95% quantile of chi-squared with 2 d.o.f.
    tt = np.linspace(0, 2 * np.pi, 200)
    bd, bphi = s0 + k * sig_s * np.cos(tt), k * sig_d * np.sin(tt)
    m_e = np.array([s0, 0.0])                # EKF: mean through f, covariance V M V^T
    V = np.array([[1.0, 0.0], [0.0, s0]])    # position rows of V at heading 0
    P_e = V @ np.diag([sig_s**2, sig_d**2]) @ V.T
    m_true = np.array([s0 * np.exp(-sig_d**2 / 2), 0.0])
    print("banana: true mean", m_true, "sample mean", px.mean(), py.mean())
    fig, ax = plt.subplots(figsize=(W * 0.7, W * 0.52))
    arc = np.linspace(-1.25, 1.25, 100)
    ax.plot(s0 * np.cos(arc), s0 * np.sin(arc), color=viz.INK_2, lw=0.6, ls=":")
    ax.scatter(px, py, s=2.5, color=viz.MUTED, alpha=0.55, lw=0, label="samples")
    ax.plot(bd * np.cos(bphi), bd * np.sin(bphi), color=viz.INK, lw=1.2, ls="--", label="true 95% region")
    ax.scatter(*m_true, s=28, c=viz.INK, zorder=6, label="true mean")
    viz.cov_ellipse(ax, m_e, P_e, n_sigma=k, name="EKF", lw=1.6, label="EKF 95% ellipse")
    ax.scatter(*m_e, marker="P", s=45, c=viz.COLORS["EKF"], zorder=6, label="EKF mean")
    ax.scatter([0], [0], s=28, color=viz.INK, zorder=6)
    ax.text(0, -0.5, "start", fontsize=7.5, color=viz.INK, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.0, 6.6); ax.set_ylim(-4.6, 4.6)
    ax.set_xlabel("$p^x$"); ax.set_ylabel("$p^y$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5)
    save(fig, "ch04_banana")


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



def _conv_example():
    """The function a of the convolution figures (a peak and a plateau) and its blur by g = N(0, 0.7^2)."""
    y = np.linspace(-2, 12, 2801)
    dy = y[1] - y[0]
    a = 0.3 * _gauss(y, 2.0, 0.3**2) + 0.7 * ((y >= 5) & (y <= 9)) / 4.0
    sd = 0.7
    a_g = np.convolve(a, _gauss(np.arange(-3, 3 + dy / 2, dy), 0.0, sd**2) * dy, mode="same")
    return y, a, sd, a_g


def fig_conv_read():
    """Computing a * g: for three points p, the function, the weights around p and their product
    (whose area is the value at p); last row, the value at every p."""
    y, a, sd, a_g = _conv_example()
    dy = y[1] - y[0]
    blue = viz.COLORS["EKF"]
    ps = (2.0, 5.0, 7.0)
    fig, axes = plt.subplots(4, 3, figsize=(W, W * 0.48), sharex=True, sharey="row")
    for c, p in enumerate(ps):
        w = _gauss(y, p, sd**2)
        prod = a * w
        value = prod.sum() * dy
        axes[0, c].plot(y, a, color=viz.INK, lw=1.1)
        axes[1, c].plot(y, w, color=blue, lw=1.2)
        axes[2, c].plot(y, prod, color=blue, lw=1.0)
        axes[2, c].fill_between(y, 0, prod, color=blue, alpha=0.35, lw=0)
        axes[2, c].annotate(f"area {value:.3f}", (0.97, 0.75), xycoords="axes fraction", ha="right",
                            fontsize=8.5, color=viz.INK)
        axes[3, c].plot(y, a, color=viz.MUTED, lw=0.8)
        axes[3, c].plot(y, a_g, color=blue, lw=1.4)
        axes[3, c].plot(p, value, "o", color=blue, ms=4.5)
        for r in range(4):
            axes[r, c].axvline(p, color=viz.INK_2, ls="--", lw=0.7)
        axes[0, c].set_title(f"$p={p:g}$", fontsize=9)
        print(f"convolution: (a*g)({p:g}) = {value:.4f}")
    for r, label in enumerate(["$a(y)$", "$g(p-y)$", "product", "$(a\\ast g)(p)$"]):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for ax in axes.flat:
        ax.set_yticks([]); ax.set_xlim(-1, 11); ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(labelsize=7.5)
    axes[3, 1].set_xlabel("$y$ (rows 1 to 3), $p$ (row 4)", fontsize=8.5, labelpad=1)
    for r in range(4):
        top = max(ax.get_ylim()[1] for ax in axes[r])
        axes[r, 0].set_ylim(0, top * 1.1)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.11, hspace=0.12, wspace=0.05)
    save(fig, "ch02_conv_read")
    print(f"convolution: top of the peak {a.max():.4f}, plateau height {0.7 / 4:.4f}")


def fig_conv_result():
    """a * g is the density of X + W: draws of X, of W and of X + W, with their densities."""
    y = np.linspace(-4, 12, 3201)
    dy = y[1] - y[0]
    a = 0.3 * _gauss(y, 2.0, 0.3**2) + 0.7 * ((y >= 5) & (y <= 9)) / 4.0
    sd = 0.7
    g = _gauss(y, 0.0, sd**2)
    a_g = np.convolve(a, _gauss(np.arange(-3, 3 + dy / 2, dy), 0.0, sd**2) * dy, mode="same")
    rng = np.random.default_rng(0)
    n = 200_000
    X = np.where(rng.random(n) < 0.3, rng.normal(2.0, 0.3, n), rng.uniform(5, 9, n))
    Wn = rng.normal(0, sd, n)
    blue, light = viz.COLORS["EKF"], "#b9d4f2"
    bins = np.arange(-4, 12.01, 0.2)
    panels = [(X, a, "(a) $X$, density $a$", viz.INK), (Wn, g, "(b) $W$, density $g$", viz.INK),
              (X + Wn, a_g, "(c) $X+W$, density $a\\ast g$", blue)]
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.28), sharex=True, sharey=True)
    for ax, (draws, dens, title, colour) in zip(axes, panels):
        ax.hist(draws, bins=bins, density=True, color=light, lw=0)
        ax.plot(y, dens, color=colour, lw=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$y$", fontsize=8.5, labelpad=1)
        ax.tick_params(labelsize=7.5)
    axes[0].set_xlim(-3, 11); axes[0].set_ylim(0, 0.62)
    axes[0].set_xticks([-2, 0, 2, 4, 6, 8, 10])
    plt.tight_layout(w_pad=0.6)
    save(fig, "ch02_conv_result")


def fig_cov_sum():
    """Covariance ellipses of X, W and X + W: a round W makes a thin X rounder; a W shaped like X
    keeps the shape."""
    th = np.radians(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Px = R @ np.diag([9.0, 0.09]) @ R.T
    cases = [(np.eye(2), "(a) round $W$: the sum is rounder"),
             (0.5 * Px, "(b) $\\mathrm{Cov}\\,W=0.5\\,\\mathrm{Cov}\\,X$: same shape, larger")]
    rng = np.random.default_rng(0)
    n = 20_000
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.45))
    for ax, (Pw, title) in zip(axes, cases):
        X = rng.multivariate_normal([0, 0], Px, n)
        Wn = rng.multivariate_normal([0, 0], Pw, n)
        S = X + Wn
        ax.scatter(*S[::10].T, s=1.5, color=viz.COLORS["EKF"], alpha=0.25, lw=0)
        viz.cov_ellipse(ax, np.zeros(2), Px, ec=viz.INK, lw=1.4, label="$X$")
        viz.cov_ellipse(ax, np.zeros(2), Pw, ec=viz.COLORS["UKF"], lw=1.4, ls="--", label="$W$")
        viz.cov_ellipse(ax, np.zeros(2), Px + Pw, ec=viz.COLORS["EKF"], lw=1.8, label="$X+W$")
        ax.set_xlim(-8, 8); ax.set_ylim(-5.5, 5.5); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        vals = np.linalg.eigvalsh(Px + Pw)
        print(f"cov sum {title[:3]}: variances along the axes {vals[::-1].round(3)}, standard deviations "
              f"{np.sqrt(vals[::-1]).round(3)}, axis ratio {np.sqrt(vals[1] / vals[0]):.2f} "
              f"(X alone: {np.sqrt(9 / 0.09):.0f}); sample covariance of X+W {np.cov(S.T).round(2).tolist()}, "
              f"Cov X + Cov W {(Px + Pw).round(2).tolist()}")
    plt.tight_layout()
    save(fig, "ch02_cov_sum")


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


def fig_pf_cycle():
    """The cycle of fig_bayes_cycle done with 60 particles: move, weight, resample."""
    x = np.linspace(-10, 20, 3001)
    dx = x[1] - x[0]
    d, q = 5.0, 1.5
    bel = 0.6 * _gauss(x, 0.0, 1.0) + 0.4 * _gauss(x, 4.0, 0.6)
    kern = _gauss(np.arange(-10, 10 + dx, dx), 0.0, q) * dx
    pred = np.convolve(np.interp(x - d, x, bel, left=0, right=0), kern, mode="same")
    lik = _gauss(x, 9.6, 0.8)
    post = pred * lik
    post /= post.sum() * dx
    rng = np.random.default_rng(3)
    N = 60
    comp = rng.random(N) < 0.6
    p0 = np.where(comp, rng.normal(0.0, 1.0, N), rng.normal(4.0, 0.6, N))
    p1 = p0 + d + rng.normal(0.0, q, N)                    # predict: move every particle
    w = _gauss(p1, 9.6, 0.8)
    w /= w.sum()                                            # weight by the likelihood
    idx = systematic_resample(w, rng)                       # resample
    p2 = p1[idx]
    ess = 1.0 / np.sum(w**2)
    print("pf cycle: ESS %.1f of %d, distinct survivors %d" % (ess, N, len(np.unique(idx))))
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.4), sharey=True)
    a1, a2, a3 = axes
    y0, y1 = -0.035, -0.085
    a1.plot(x, bel, color=viz.INK_2, lw=1.0, alpha=0.7, label=r"$\mathrm{bel}(x_t)$")
    a1.plot(x, pred, color=viz.COLORS["EKF"], lw=1.2, alpha=0.8, label=r"$\overline{\mathrm{bel}}(x_{t+1})$")
    a1.scatter(p0, np.full(N, y0), s=9, color=viz.INK_2, lw=0, label="particles at $t$")
    a1.scatter(p1, np.full(N, y1), s=9, color=viz.COLORS["EKF"], lw=0, label="after the move")
    a1.set_title("predict: move every particle", fontsize=8.5)
    a2.plot(x, pred, color=viz.COLORS["EKF"], lw=1.2, alpha=0.8, label=r"$\overline{\mathrm{bel}}(x_{t+1})$")
    a2.plot(x, lik / lik.max() * 0.3, color=viz.COLORS["UKF"], lw=1.2, ls="--",
            label=r"$p(z_{t+1}\mid x)$ (scaled)")
    a2.scatter(p1, np.full(N, y1), s=4 + 900 * w, color=viz.COLORS["EKF"], lw=0, alpha=0.75,
               label="area = weight")
    a2.set_title("weight by the likelihood", fontsize=8.5)
    a3.plot(x, post, color=viz.COLORS["PF"], lw=1.4, label=r"$\mathrm{bel}(x_{t+1})$")
    jitter = np.zeros(N)                                    # stack the copies of one particle
    for v in np.unique(p2):
        m = np.flatnonzero(p2 == v)
        jitter[m] = -0.016 * np.arange(len(m))
    a3.scatter(p2, y0 + jitter, s=9, color=viz.COLORS["PF"], lw=0, label="after resampling")
    a3.set_title("resample: copy the heavy ones", fontsize=8.5)
    for a in axes:
        a.axhline(0, color=viz.INK_2, lw=0.4)
        a.set_xlim(-4, 14); a.set_xlabel("$x$"); a.set_yticks([]); a.set_ylim(-0.26, 0.95)
        a.legend(fontsize=7, loc="upper left", frameon=False)
    plt.tight_layout()
    save(fig, "ch06_pf_cycle")


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
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(W * 0.8, W * 0.62), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    a1.plot(steps, slow, color=viz.INK_2, lw=1.4, label=r"$w_{\mathrm{slow}}$")
    a1.plot(steps, fast, color=viz.COLORS["UKF"], lw=1.2, label=r"$w_{\mathrm{fast}}$")
    a1.set_ylabel("average likelihood\n(per landmark)"); a1.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.25))
    a2.bar(steps, inj, color=viz.COLORS["PF"], width=1.0, label="injected particles")
    a2.set_ylabel("injected\nparticles")
    a3.plot(steps, err, color=viz.INK, lw=1.2)
    a3.set_ylabel("position\nerror"); a3.set_xlabel("step")
    for a in (a1, a2, a3):
        a.axvline(200, color=viz.MUTED, lw=0.8, ls=":")
        a.set_xlim(150, 260)
    a1.set_title("step 200 (dotted): the robot is carried to another part of the lawn", fontsize=8)
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

# ---------------------------------------------------------------- chapter 7
def closed_loop_experiment(n_seeds=10, n_steps=700):
    """Each filter in the loop, steering the robot along the mowing stripes.
    Returns {name: (cross-track RMS, estimation RMSE, runs that left the lawn)}."""
    world = World.default()
    wp = stripes_path(world)
    x0 = (wp[0, 0], wp[0, 1], 0.0)
    P0 = np.diag([1.0, 1.0, 0.01])
    make = {
        "dead reckoning": lambda s: DeadReckoning.from_world(world, x0, P0),
        "KF": lambda s: CompassGpsKF.from_world(world, x0, P0),
        "EKF": lambda s: EKF.from_world(world, x0, P0),
        "UKF": lambda s: UKF.from_world(world, x0, P0),
        "PF": lambda s: ParticleFilter(world, 1000, init=(np.array(x0), P0), seed=s),
    }
    out = {}
    for name, mk in make.items():
        xt, est, left = [], [], 0
        for seed in range(n_seeds):
            run, res = simulate_closed_loop(world, mk(seed), PathFollower(wp), n_steps, x0=x0, seed=seed)
            xt.append(distance_to_path(run.truth, wp) ** 2)
            est.append(position_error(res.means, run.truth) ** 2)
            left += int(not world.inside(run.truth[:, :2]).all())
        out[name] = (np.sqrt(np.mean(np.concatenate(xt))), np.sqrt(np.mean(np.concatenate(est))), left)
    return out


def fig_closed_loop():
    """The filter in the loop on a mowing pattern: dead reckoning against the EKF."""
    world = World.default()
    wp = stripes_path(world)
    x0 = (wp[0, 0], wp[0, 1], 0.0)
    P0 = np.diag([1.0, 1.0, 0.01])
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.60), sharey=True)
    handles = {}
    cases = [("dead reckoning in the loop", DeadReckoning.from_world(world, x0, P0), viz.COLORS["UKF"]),
             ("EKF in the loop", EKF.from_world(world, x0, P0), viz.COLORS["EKF"])]
    for ax, (title, flt, col) in zip(axes, cases):
        run, res = simulate_closed_loop(world, flt, PathFollower(wp), 700, x0=x0, seed=1)
        viz.plot_world(ax, world, label_landmarks=False)
        handles["planned path"], = ax.plot(wp[:, 0], wp[:, 1], color=viz.MUTED, lw=3.0, alpha=0.45, solid_capstyle="round")
        handles["believed position (%s)" % flt.name], = ax.plot(res.means[:, 0], res.means[:, 1], color=col, lw=1.0)
        handles["true position"], = ax.plot(run.truth[:, 0], run.truth[:, 1], color=viz.INK, lw=1.0)
        xt = distance_to_path(run.truth, wp)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$x$")
        print("closed loop, seed 1, %s: cross-track RMS %.2f max %.1f, estimation RMSE %.2f, steps %d"
              % (title, np.sqrt(np.mean(xt**2)), xt.max(), position_rmse(res.means, run.truth), run.n_steps))
    axes[0].set_ylabel("$y$"); axes[1].set_ylabel("")
    axes[0].set_ylim(-world.half_height - 12, world.half_height + 60)
    order = ["planned path", "true position", "believed position (dead reckoning)", "believed position (EKF)"]
    fig.legend([handles[k] for k in order], order, fontsize=7, loc="lower center", ncol=2)
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "ch07_closed_loop")
    for name, (xt, est, left) in closed_loop_experiment().items():
        print("closed loop, 10 runs, %-15s cross-track RMS %6.2f  estimation RMSE %6.2f  left the lawn %d/10" % (name, xt, est, left))


if __name__ == "__main__":
    fig_closed_loop()
    fig_transition()
    fig_helicoid()
    fig_coupling()
    fig_dead_reckoning()
    fig_grid()
    fig_potato()
    fig_quadtree()
    fig_shift_1d()
    fig_shift_2d()
    fig_shift_spread()
    fig_kf_track()
    fig_riccati()
    fig_kf_nees()
    fig_ekf_track()
    fig_ekf_banana()
    fig_arc(False, "ch04_arc")
    fig_arc(True, "ch05_arc")
    fig_transient(["EKF"], "ch04_transient")
    fig_transient(["EKF", "UKF"], "ch05_transient")
    fig_pf_cycle()
    fig_ess()
    fig_resampling()
    fig_global()
    fig_modes()
    fig_kidnap()
    fig_bayes_cycle()
    fig_conv_read()
    fig_conv_result()
    fig_cov_sum()
    fig_gaussian()
    fig_linearisation(False, "ch04_linearisation")
    fig_linearisation(True, "ch05_regression")
    fig_sigma_points()
    fig_importance()
    fig_amcl()
