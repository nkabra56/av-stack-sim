"""Matplotlib animation of a parking run: a top-down view plus two live telemetry
panels (IMPLEMENTATION.md's M5 milestone).

Top-down (left, larger): the planned path (light gray), the true trajectory (solid
blue, vehicle drawn at its true pose -- since that's what actually happened), the
EKF's estimated trajectory (dashed orange), and a 1-sigma position-uncertainty ellipse
around the estimate. The gap between the solid and dashed lines *is* the estimation
error -- the whole point of having an estimator is visible directly in the animation,
not just in a metrics table.

Speed profile (top right): commanded speed vs. time, filling in as the run
progresses, with a marker at the current tick -- makes the speed governor's throttling
(KNOWN_BUGS.md entries 2/3) visible as a real dip in the trace, not just inferable
from the vehicle slowing down on screen.

Sensor readings (bottom right, polar): each ultrasonic beam's current range as a bar
at its angle (relative to the vehicle's own heading, so the fan rotates rigidly with
the vehicle exactly like the real body-frame sensor does), radius clamped to the
sensor's own max_range. A beam that's clear reads at the rim; an obstacle closing in
shows up as a bar shortening toward the center, in whichever direction it's actually
approaching from.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle

from core.environment import Environment
from core.harness import SimulationResult

VEHICLE_LENGTH = 4.5  # matches the ~2.7m wheelbase + overhangs used everywhere else (Vehicle,
VEHICLE_WIDTH = 1.8  # scenario spot dimensions) -- not an arbitrary small rendering size


def _axis_bounds(result: SimulationResult, environment: Environment, pad: float = 1.0):
    xs = [environment.spot.x]
    ys = [environment.spot.y]
    if len(result.true_history):
        xs += list(result.true_history[:, 0])
        ys += list(result.true_history[:, 1])
    if result.path is not None and len(result.path):
        # Otherwise the planned path can render outside the visible axes whenever the
        # vehicle stalls well short of the goal (exactly the scenarios meant to show the
        # gap between the planned path and what actually happened -- see DESIGN.md section 6).
        xs += list(result.path[:, 0])
        ys += list(result.path[:, 1])
    for obstacle in environment.obstacles:
        xs += [obstacle.x - obstacle.radius, obstacle.x + obstacle.radius]
        ys += [obstacle.y - obstacle.radius, obstacle.y + obstacle.radius]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def _ellipse_params(cov_xy: np.ndarray) -> tuple[float, float, float]:
    """1-sigma position-uncertainty ellipse: (width, height, angle_degrees)."""
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    eigvals = np.clip(eigvals, 0.0, None)
    width, height = 2 * np.sqrt(eigvals[1]), 2 * np.sqrt(eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    return width, height, angle


def render_animation(
    result: SimulationResult, environment: Environment, title: str = "", save_path: str | None = None
) -> None:
    if len(result.true_history) == 0:
        print(f"No trajectory to animate for '{title}' — controller halted before the first step.")
        return

    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_sensor = fig.add_subplot(grid[1, 1], projection="polar")

    fig.suptitle(title)
    ax.set_aspect("equal", "box")

    for obstacle in environment.obstacles:
        ax.add_patch(
            Circle((obstacle.x, obstacle.y), obstacle.radius, facecolor="lightcoral", edgecolor="darkred", linewidth=2)
        )

    spot = environment.spot
    gs = spot.size
    ax.add_patch(
        Rectangle(
            (spot.x - gs / 2, spot.y - gs / 2), gs, gs,
            facecolor="green", alpha=0.3, edgecolor="darkgreen", linewidth=2,
        )
    )

    if result.path is not None and len(result.path):
        ax.plot(result.path[:, 0], result.path[:, 1], "--", lw=1.5, color="gray", alpha=0.6, label="planned")

    xlim, ylim = _axis_bounds(result, environment)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    vehicle_patch = Rectangle(
        (-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2), VEHICLE_LENGTH, VEHICLE_WIDTH,
        facecolor="blue", edgecolor="navy", linewidth=2,
    )
    ax.add_patch(vehicle_patch)
    true_trail, = ax.plot([], [], "-", lw=2.5, color="tab:blue", label="true")
    est_trail, = ax.plot([], [], "--", lw=2, color="tab:orange", label="estimated")
    uncertainty_ellipse = Ellipse((0, 0), 0, 0, facecolor="tab:orange", alpha=0.2, edgecolor="none")
    ax.add_patch(uncertainty_ellipse)
    ax.legend(loc="upper right", fontsize=8)

    true_history = result.true_history
    est_history = result.estimated_history
    cov_history = result.covariance_history
    n = len(true_history)

    # Speed profile: commanded v vs. time, filling in as the run progresses.
    has_speed = result.controls is not None and len(result.controls)
    times = np.arange(n) * result.dt
    speeds = result.controls[:, 0] if has_speed else np.zeros(n)
    ax_speed.set_title("speed", fontsize=9)
    ax_speed.set_xlabel("t (s)", fontsize=8)
    ax_speed.set_ylabel("v (m/s)", fontsize=8)
    ax_speed.tick_params(labelsize=7)
    if n > 1:
        ax_speed.set_xlim(0, times[-1])
    v_pad = max(0.1, 0.1 * np.abs(speeds).max()) if len(speeds) else 0.1
    ax_speed.set_ylim(speeds.min() - v_pad if len(speeds) else -0.1, speeds.max() + v_pad if len(speeds) else 0.1)
    speed_trail, = ax_speed.plot([], [], "-", lw=1.5, color="tab:blue")
    speed_dot, = ax_speed.plot([], [], "o", ms=4, color="tab:blue")

    # Sensor readings: one bar per beam, at its angle relative to the vehicle's own
    # heading (the polar axes rotate with the vehicle each frame, not the world).
    has_sensor = result.sensor_angles is not None and len(result.sensor_angles)
    sensor_angles = result.sensor_angles if has_sensor else np.array([])
    sensor_ranges = result.sensor_ranges if has_sensor else np.zeros((n, 0))
    max_range = float(sensor_ranges.max()) if sensor_ranges.size else 1.0
    ax_sensor.set_title("ultrasonic ranges", fontsize=9, pad=12)
    ax_sensor.set_theta_zero_location("N")  # forward = up, matching the top-down view's "ahead"
    ax_sensor.set_ylim(0, max_range)
    ax_sensor.tick_params(labelsize=6)
    bar_width = (2 * np.pi / len(sensor_angles) * 0.8) if len(sensor_angles) else 0.1
    sensor_bars = ax_sensor.bar(
        sensor_angles, np.zeros(len(sensor_angles)), width=bar_width, color="tab:green", alpha=0.7,
    )

    def update(i):
        x, y, theta = true_history[i]
        true_trail.set_data(true_history[: i + 1, 0], true_history[: i + 1, 1])
        est_trail.set_data(est_history[: i + 1, 0], est_history[: i + 1, 1])
        trans = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        vehicle_patch.set_transform(trans)

        ex, ey = est_history[i, 0], est_history[i, 1]
        width, height, angle = _ellipse_params(cov_history[i, :2, :2])
        uncertainty_ellipse.set_center((ex, ey))
        uncertainty_ellipse.width = width
        uncertainty_ellipse.height = height
        uncertainty_ellipse.angle = angle

        speed_trail.set_data(times[: i + 1], speeds[: i + 1])
        speed_dot.set_data([times[i]], [speeds[i]])

        for bar, r in zip(sensor_bars, sensor_ranges[i] if len(sensor_ranges) else []):
            bar.set_height(r)

        return (vehicle_patch, true_trail, est_trail, uncertainty_ellipse, speed_trail, speed_dot, *sensor_bars)

    anim = FuncAnimation(fig, update, frames=n, interval=50, blit=False)
    fig.tight_layout()

    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=20))
        plt.close(fig)
    else:
        plt.show()
