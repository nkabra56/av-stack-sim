"""Matplotlib top-down animation of a parking run. See DESIGN.md section 2 (Visualization)."""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from auto_park.environment import Environment
from auto_park.simulation import SimulationResult

VEHICLE_LENGTH = 1.0
VEHICLE_WIDTH = 0.6


def _axis_bounds(result: SimulationResult, environment: Environment, pad: float = 1.0):
    xs = [environment.spot.x]
    ys = [environment.spot.y]
    if len(result.history):
        xs += list(result.history[:, 0])
        ys += list(result.history[:, 1])
    for obstacle in environment.obstacles:
        xs += [obstacle.x - obstacle.radius, obstacle.x + obstacle.radius]
        ys += [obstacle.y - obstacle.radius, obstacle.y + obstacle.radius]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def render_animation(
    result: SimulationResult, environment: Environment, title: str = "", save_path: str | None = None
) -> None:
    if len(result.history) == 0:
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

    xlim, ylim = _axis_bounds(result, environment)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    vehicle_patch = Rectangle(
        (-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2), VEHICLE_LENGTH, VEHICLE_WIDTH,
        facecolor="blue", edgecolor="navy", linewidth=2,
    )
    ax.add_patch(vehicle_patch)
    trail, = ax.plot([], [], "-", lw=3, color="orange")

    history = result.history

    def update(i):
        x, y, theta = history[i]
        trail.set_data(history[: i + 1, 0], history[: i + 1, 1])
        trans = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        vehicle_patch.set_transform(trans)
        return vehicle_patch, trail

    anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)

    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=20))
        plt.close(fig)
    else:
        plt.show()
