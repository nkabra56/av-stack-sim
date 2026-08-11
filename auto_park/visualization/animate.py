"""Matplotlib top-down animation of a parking run. See DESIGN.md section 2 (Visualization).

Shows the planned path (light gray), the true trajectory (solid blue, vehicle drawn at
its true pose -- since that's what actually happened), the EKF's estimated trajectory
(dashed orange), and a 1-sigma position-uncertainty ellipse around the estimate. The
gap between the solid and dashed lines *is* the estimation error -- the whole point of
having an estimator is visible directly in the animation, not just in a metrics table.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle

from auto_park.environment import Environment
from auto_park.harness import SimulationResult

VEHICLE_LENGTH = 1.0
VEHICLE_WIDTH = 0.6


def _axis_bounds(result: SimulationResult, environment: Environment, pad: float = 1.0):
    xs = [environment.spot.x]
    ys = [environment.spot.y]
    if len(result.true_history):
        xs += list(result.true_history[:, 0])
        ys += list(result.true_history[:, 1])
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
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

        return vehicle_patch, true_trail, est_trail, uncertainty_ellipse

    anim = FuncAnimation(fig, update, frames=len(true_history), interval=50, blit=True)

    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=20))
        plt.close(fig)
    else:
        plt.show()
