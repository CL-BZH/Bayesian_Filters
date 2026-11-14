"""Live demo of a robot lawnmower being localised, drawn with Python's ``turtle`` module.

    python -m bayesian_filters.demo --filter pf
    python -m bayesian_filters.demo --filter ekf --kidnap 240
    python -m bayesian_filters.demo --filter pf --association unknown --recovery

Keys: space = pause / resume, q = quit.

Needs a display and Tk (``python3-tk`` on Debian/Ubuntu).  Notebooks do not
need this module; they use :mod:`bayesian_filters.viz` instead.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .ekf import EKF
from .kf import CompassGpsKF
from .pf import ParticleFilter
from .ukf import UKF
from .world import World, fence_correction, figure_eight_control

HEX = {"EKF": "#2a78d6", "UKF": "#eb6834", "PF": "#1baf7a", "KF": "#6250d6"}


def build_filter(kind, world, x0, args):
    P0 = np.diag([10.0, 10.0, 0.05])
    if kind == "kf":
        return CompassGpsKF.from_world(world, x0, P0)
    if kind == "ekf":
        return EKF.from_world(world, x0, P0, association="nearest" if args.association == "unknown" else "known")
    if kind == "ukf":
        return UKF.from_world(world, x0, P0)
    init = "global" if args.global_init else (x0, P0)
    return ParticleFilter(
        world, args.particles, init=init, association=args.association, roughening=0.5,
        recovery=(0.001, 0.1) if args.recovery else None, seed=args.seed,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", choices=["kf", "ekf", "ukf", "pf"], default="pf")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--landmarks", type=int, default=0, help="0 = the 4 fixed landmarks, n = n random ones")
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--global-init", action="store_true", help="PF: start with no idea where the mower is")
    ap.add_argument("--association", choices=["known", "unknown"], default="known")
    ap.add_argument("--recovery", action="store_true", help="PF: augmented MCL random-particle injection")
    ap.add_argument("--kidnap", type=int, default=None, help="teleport the mower at this step")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep per step")
    args = ap.parse_args(argv)

    import turtle  # imported here so the package works without Tk

    rng = np.random.default_rng(args.seed)
    world = World.random(args.landmarks, seed=args.seed) if args.landmarks else World.default()
    x = np.zeros(3)
    flt = build_filter(args.filter, world, x, args)
    name = flt.name

    screen = turtle.Screen()
    screen.setup(520, 560)
    screen.title(f"Where is the mower? ({name})")
    screen.tracer(0)
    screen.bgcolor("#eef6e6")  # the lawn
    # a top-down mower: deck with a bevelled front (shape coordinates point along +y)
    screen.register_shape("mower", ((-7, -9), (7, -9), (7, 6), (4, 10), (-4, 10), (-7, 6)))

    def pen(color="black", width=1):
        t = turtle.Turtle(visible=False)
        t.penup()
        t.speed(0)
        t.color(color)
        t.width(width)
        return t

    # static scene
    t = pen("lightgrey")
    t.goto(-world.half_width, -world.half_height)
    t.pendown()
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        t.goto(t.xcor() + dx * 2 * world.half_width, t.ycor() + dy * 2 * world.half_height)
    t.penup()
    from .world import LANDMARK_COLORS

    for j, (lx, ly) in enumerate(world.landmarks):
        t.goto(lx, ly)
        t.color("lightgrey")
        t.dot(12, LANDMARK_COLORS[j % len(LANDMARK_COLORS)])
        t.goto(lx + 7, ly + 5)
        t.color("#52514e")
        t.write(f"L{j}", font=("Arial", 9, "normal"))
    info = pen("#52514e")
    info.goto(-world.half_width, world.half_height + 12)

    robot = turtle.Turtle(shape="mower")
    robot.color("#0b0b0b", "lime")
    robot.speed(0)
    robot.radians()
    robot.shapesize(0.8, 0.8)
    estimate = turtle.Turtle(shape="mower")
    estimate.color(HEX[name], "")
    estimate.radians()
    estimate.penup()
    estimate.shapesize(0.8, 0.8, 2)
    trail = pen(HEX[name])
    cloud = pen(HEX[name])
    ellipse = pen(HEX[name], 2)

    state = {"paused": False, "quit": False}
    screen.onkey(lambda: state.update(paused=not state["paused"]), "space")
    screen.onkey(lambda: state.update(quit=True), "q")
    screen.listen()

    for step in range(args.steps):
        while state["paused"] and not state["quit"]:
            screen.update()
            time.sleep(0.05)
        if state["quit"]:
            break
        if args.kidnap is not None and step == args.kidnap:
            x = world.sample_uniform_poses(1, rng)[0]
            robot.penup()
        u = figure_eight_control(step)
        u[1] += fence_correction(world, x)
        x = world.move(x, u, rng)
        robot.goto(x[0], x[1])
        robot.setheading(x[2])
        robot.pendown()
        z, visible = world.sense_range_bearing(x, rng)
        if name == "KF":
            flt.predict(u, heading=world.sense_compass(x, rng))
            flt.update(world.sense_gps(x, rng))
        else:
            flt.predict(u)
            flt.update(z, visible)
        m, P = flt.mean, flt.cov
        estimate.goto(m[0], m[1])
        if len(m) > 2:
            estimate.setheading(m[2])
        trail.goto(m[0], m[1])
        trail.dot(2)
        cloud.clear()
        if name == "PF":
            for p in flt.particles[:: max(1, flt.N // 600)]:
                cloud.goto(p[0], p[1])
                cloud.dot(3)
        ellipse.clear()
        vals, vecs = np.linalg.eigh(P[:2, :2])
        a = np.linspace(0, 2 * np.pi, 40)
        pts = m[:2, None] + 2 * vecs @ (np.sqrt(np.clip(vals, 0, None))[:, None] * np.vstack([np.cos(a), np.sin(a)]))
        ellipse.goto(*pts[:, 0])
        ellipse.pendown()
        for k in range(1, pts.shape[1]):
            ellipse.goto(*pts[:, k])
        ellipse.penup()
        info.clear()
        err = np.hypot(*(m[:2] - x[:2]))
        info.write(f"{name}   step {step}   error {err:6.1f}   (space: pause, q: quit)", font=("Arial", 10, "normal"))
        screen.update()
        if args.delay:
            time.sleep(args.delay)
    try:
        screen.bye()
    except turtle.Terminator:  # window already closed
        pass


if __name__ == "__main__":
    main()
