"""3D Foxglove scene export: an additive alternative to animate.py's Matplotlib
animation, for producing a showcase-quality recording (see FOXGLOVE_VIZ_PLAN.md).

Writes an MCAP file containing a `/scene` (SceneUpdate) topic driving a 3D view --
ground/lot markings, obstacles, the ego vehicle, its true and EKF-estimated
trajectories, the 1-sigma uncertainty ellipse, and a live ultrasonic sensor fan --
plus a `/tf` FrameTransform for the vehicle pose (so a Foxglove layout's 3D panel can
camera-follow the "vehicle" frame), and `/speed` / `/sensors` / `/status` JSON
channels for Foxglove's own Plot/Raw Messages panels. Open the result in the free
Foxglove desktop app; see README.md's "3D visualization (Foxglove)" section.
"""

import math
from pathlib import Path

import numpy as np

import foxglove
from foxglove import Channel
from foxglove.channels import FrameTransformChannel, SceneUpdateChannel
from foxglove.messages import (
    Color,
    CubePrimitive,
    CylinderPrimitive,
    FrameTransform,
    LinePrimitive,
    LinePrimitiveLineType,
    Point3,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    Vector3,
)

from core.environment import Environment
from core.harness import SimulationResult
from core.visualization.animate import VEHICLE_LENGTH, VEHICLE_WIDTH, _axis_bounds, _ellipse_params

OBSTACLE_HEIGHT = 1.5
VEHICLE_BODY_HEIGHT = 1.2
CABIN_HEIGHT = 0.6
SENSOR_Z = 0.3
_IDENTITY = Quaternion(w=1.0)


def _yaw_quaternion(theta: float) -> Quaternion:
    return Quaternion(z=math.sin(theta / 2.0), w=math.cos(theta / 2.0))


def _pose(x: float, y: float, z: float, orientation: Quaternion = _IDENTITY) -> Pose:
    return Pose(position=Vector3(x=x, y=y, z=z), orientation=orientation)


def _lerp_color(t: float, near: tuple[float, float, float], far: tuple[float, float, float], a: float = 1.0) -> Color:
    t = max(0.0, min(1.0, t))
    return Color(
        r=near[0] + (far[0] - near[0]) * t,
        g=near[1] + (far[1] - near[1]) * t,
        b=near[2] + (far[2] - near[2]) * t,
        a=a,
    )


def _range_color(r: float, max_range: float) -> Color:
    """Close obstacle -> red, clear beam -> green, matching the intuition of
    animate.py's polar sensor panel (a beam that's clear reads at the rim)."""
    t = r / max_range if max_range > 0 else 1.0
    return _lerp_color(t, (0.85, 0.15, 0.15), (0.15, 0.8, 0.3))


def _rotated_rect_ring(cx: float, cy: float, half_w: float, half_h: float, theta: float, z: float) -> list[Point3]:
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    ct, st = math.cos(theta), math.sin(theta)
    return [Point3(x=cx + x * ct - y * st, y=cy + x * st + y * ct, z=z) for x, y in [*corners, corners[0]]]


def _ellipse_ring(ex: float, ey: float, cov_xy: np.ndarray, n_pts: int = 24, z: float = 0.02) -> list[Point3]:
    """Closed ring tracing the 1-sigma position-uncertainty ellipse, reusing
    animate.py's _ellipse_params eigen-decomposition unmodified."""
    width, height, angle_deg = _ellipse_params(cov_xy)
    a, b = width / 2.0, height / 2.0
    angle = math.radians(angle_deg)
    ct, st = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(n_pts + 1):
        t = 2 * math.pi * i / n_pts
        x, y = a * math.cos(t), b * math.sin(t)
        pts.append(Point3(x=ex + x * ct - y * st, y=ey + x * st + y * ct, z=z))
    return pts


def _dash_segments(p0: tuple[float, float], p1: tuple[float, float], n_dashes: int = 8, z: float = 0.01) -> list[LinePrimitive]:
    x0, y0 = p0
    x1, y1 = p1
    dashes = []
    for i in range(n_dashes):
        t0 = i / n_dashes
        t1 = t0 + 0.5 / n_dashes
        dashes.append(
            LinePrimitive(
                type=LinePrimitiveLineType.LineStrip,
                thickness=0.04,
                color=Color(r=0.7, g=0.72, b=0.76, a=0.8),
                points=[
                    Point3(x=x0 + (x1 - x0) * t0, y=y0 + (y1 - y0) * t0, z=z),
                    Point3(x=x0 + (x1 - x0) * t1, y=y0 + (y1 - y0) * t1, z=z),
                ],
            )
        )
    return dashes


def _line_strip_entity(entity_id: str, points: list[Point3], color: Color, thickness: float = 0.06, frame_id: str = "world") -> SceneEntity:
    return SceneEntity(
        id=entity_id,
        frame_id=frame_id,
        lines=[LinePrimitive(type=LinePrimitiveLineType.LineStrip, points=points, color=color, thickness=thickness)],
    )


def _ground_entity(xlim: tuple[float, float], ylim: tuple[float, float]) -> SceneEntity:
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    sx, sy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    ground = CubePrimitive(
        pose=_pose(cx, cy, -0.05),
        size=Vector3(x=sx, y=sy, z=0.1),
        color=Color(r=0.82, g=0.83, b=0.85, a=1.0),
    )
    return SceneEntity(id="ground", frame_id="world", cubes=[ground])


def _lot_entity(environment: Environment, start_xy: tuple[float, float]) -> SceneEntity:
    spot = environment.spot
    half = spot.size / 2
    outline = LinePrimitive(
        type=LinePrimitiveLineType.LineStrip,
        points=_rotated_rect_ring(spot.x, spot.y, half, half, spot.theta, z=0.01),
        color=Color(r=0.2, g=0.75, b=0.35, a=1.0),
        thickness=0.08,
    )
    lane = _dash_segments(start_xy, (spot.x, spot.y))
    return SceneEntity(id="lot", frame_id="world", lines=[outline, *lane])


def _obstacles_entity(environment: Environment) -> SceneEntity:
    cylinders = [
        CylinderPrimitive(
            pose=_pose(o.x, o.y, OBSTACLE_HEIGHT / 2),
            size=Vector3(x=o.radius * 2, y=o.radius * 2, z=OBSTACLE_HEIGHT),
            bottom_scale=1.0,
            top_scale=1.0,
            color=Color(r=0.85, g=0.35, b=0.3, a=0.95),
        )
        for o in environment.obstacles
    ]
    return SceneEntity(id="obstacles", frame_id="world", cylinders=cylinders)


def _vehicle_entity() -> SceneEntity:
    """Static local geometry in the "vehicle" frame -- world-frame pose comes entirely
    from the per-tick FrameTransform, so this entity is logged unchanged every tick."""
    body = CubePrimitive(
        pose=_pose(0, 0, VEHICLE_BODY_HEIGHT / 2),
        size=Vector3(x=VEHICLE_LENGTH, y=VEHICLE_WIDTH, z=VEHICLE_BODY_HEIGHT),
        color=Color(r=0.15, g=0.35, b=0.85, a=1.0),
    )
    cabin = CubePrimitive(
        pose=_pose(VEHICLE_LENGTH * 0.08, 0, VEHICLE_BODY_HEIGHT + CABIN_HEIGHT / 2),
        size=Vector3(x=VEHICLE_LENGTH * 0.45, y=VEHICLE_WIDTH * 0.85, z=CABIN_HEIGHT),
        color=Color(r=0.08, g=0.18, b=0.45, a=0.9),
    )
    return SceneEntity(id="vehicle_body", frame_id="vehicle", cubes=[body, cabin])


def _sensor_rays_entity(x: float, y: float, theta: float, angles: np.ndarray, ranges: np.ndarray, max_range: float) -> SceneEntity:
    lines = []
    for angle, r in zip(angles, ranges):
        ray_theta = theta + angle
        lines.append(
            LinePrimitive(
                type=LinePrimitiveLineType.LineStrip,
                thickness=0.03,
                color=_range_color(float(r), max_range),
                points=[
                    Point3(x=x, y=y, z=SENSOR_Z),
                    Point3(x=x + r * math.cos(ray_theta), y=y + r * math.sin(ray_theta), z=SENSOR_Z),
                ],
            )
        )
    return SceneEntity(id="sensor_rays", frame_id="world", lines=lines)


def render_foxglove(
    result: SimulationResult, environment: Environment, title: str = "", save_path: str = "out/demo.mcap", live: bool = False
) -> None:
    if len(result.true_history) == 0:
        print(f"No trajectory to export for '{title}' — controller halted before the first step.")
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    xlim, ylim = _axis_bounds(result, environment)
    dt_ns = int(result.dt * 1e9)
    n = len(result.true_history)

    has_sensor = result.sensor_angles is not None and len(result.sensor_angles)
    sensor_angles = result.sensor_angles if has_sensor else np.array([])
    sensor_ranges = result.sensor_ranges if has_sensor else np.zeros((n, 0))
    max_range = float(sensor_ranges.max()) if sensor_ranges.size else 1.0
    has_speed = result.controls is not None and len(result.controls)
    controls = result.controls if has_speed else np.zeros((n, 2))

    ground_entity = _ground_entity(xlim, ylim)
    lot_entity = _lot_entity(environment, tuple(result.true_history[0, :2]))
    obstacles_entity = _obstacles_entity(environment)
    vehicle_entity = _vehicle_entity()

    scene_channel = SceneUpdateChannel("/scene")
    tf_channel = FrameTransformChannel("/tf")
    speed_channel = Channel("/speed", message_encoding="json")
    sensors_channel = Channel("/sensors", message_encoding="json")
    status_channel = Channel("/status", message_encoding="json")

    true_pts: list[Point3] = []
    est_pts: list[Point3] = []

    with foxglove.open_mcap(save_path, allow_overwrite=True):
        server = foxglove.start_server() if live else None
        try:
            for i in range(n):
                t_ns = i * dt_ns
                x, y, theta = (float(v) for v in result.true_history[i])
                ex, ey, _ = (float(v) for v in result.estimated_history[i])

                tf_channel.log(
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="vehicle",
                        translation=Vector3(x=x, y=y, z=0.0),
                        rotation=_yaw_quaternion(theta),
                    ),
                    log_time=t_ns,
                )

                true_pts.append(Point3(x=x, y=y, z=0.05))
                est_pts.append(Point3(x=ex, y=ey, z=0.05))

                entities = [
                    ground_entity,
                    lot_entity,
                    obstacles_entity,
                    vehicle_entity,
                    _line_strip_entity("trail_true", true_pts, Color(r=0.15, g=0.4, b=0.95, a=1.0)),
                    _line_strip_entity("trail_est", est_pts, Color(r=0.95, g=0.6, b=0.1, a=1.0)),
                    _line_strip_entity(
                        "uncertainty", _ellipse_ring(ex, ey, result.covariance_history[i, :2, :2]), Color(r=0.95, g=0.6, b=0.1, a=0.5)
                    ),
                ]
                if has_sensor:
                    entities.append(_sensor_rays_entity(x, y, theta, sensor_angles, sensor_ranges[i], max_range))

                scene_channel.log(SceneUpdate(entities=entities), log_time=t_ns)
                speed_channel.log({"v": float(controls[i, 0]), "delta": float(controls[i, 1])}, log_time=t_ns)
                if has_sensor:
                    sensors_channel.log({f"beam_{j}": float(r) for j, r in enumerate(sensor_ranges[i])}, log_time=t_ns)

                if i == 0:
                    status_channel.log(
                        {"title": title, "scenario": title, "tick": i, "phase": "start"}, log_time=t_ns
                    )

            status_channel.log(
                {
                    "title": title,
                    "success": bool(result.success),
                    "collision": bool(result.collision),
                    "steps": n,
                    "phase": "end",
                },
                log_time=(n - 1) * dt_ns,
            )
        finally:
            if server is not None:
                server.stop()

    print(f"Foxglove scene written to {save_path} ({n} ticks, {result.dt}s/tick)")
