"""Exports simulation runs as JSON-in-JS for the browser 3D viewer in docs/viewer/.
Each scene is data-driven: tracks (x, y, heading per tick), signals, and static geometry."""

import argparse
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from core.background_traffic import simulate_lane
from core.control.acc import IDMController, MpcAccController
from core.control.intersection import IntersectionNavigator
from core.control.intersection_geometry import EAST, NORTH, SOUTH, WEST
from core.control.lane_geometry import build_arc_length_table
from core.demo import CONTROLLERS, PLANNERS
from core.environment import VEHICLE_RADIUS
from core.full_highway_harness import FullHighwayHarness
from core.harness import ParkingHarness
from core.intersection2d_harness import VehicleSpec, run_multi_approach_scenario
from core.intersection_harness import other_vehicle_present_from
from core.scenario_loader import load_scenario
from core.signalized_intersection import demo_cars, run_signalized_scenario
from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH, validate
from core.validation.kitti_loader import load_kitti_poses
from core.validation.lane_centering_validation import REAL_LATERAL_STD_M
from core.validation.lane_centering_validation import validate as validate_lane_centering
from core.validation.ngsim_loader import load_following_pair, load_lane_centerline
from core.visualization import scenery
from core.visualization.animate import VEHICLE_LENGTH, VEHICLE_WIDTH, _axis_bounds, _ellipse_params

DEFAULT_OUT = Path(__file__).parents[2] / "docs" / "viewer" / "scenes.js"
LANE_WIDTH = 3.7
BLUE, ORANGE, RED, GRAY = "#2f6fe4", "#f59e0b", "#ef4444", "#8b95a5"
ASPHALT, ZONE = "#2a313c", "#323a46"
KITTI_SEEDS = 20
STOP_LINE = 400.0  # arc length along the NGSIM lane, as in tests/test_full_highway.py
INTERSECTION_HALF = 9.5  # conflict-zone half width, the harness default
INTERSECTION_ROAD_WIDTH = 9.0
PARKING_SCENES = [
    ("perpendicular_flanked", "mpc", "Perpendicular, flanked by parked cars",
     "The direct Dubins path clips the left neighbor, so Hybrid A* routes around it."),
    ("parallel_between_cars", "mpc", "Parallel parking between two cars",
     "The tightest case: a reverse-gear cusp between parked cars, tracked by MPC."),
    ("parallel_between_cars", "pure_pursuit", "Same scene with Pure Pursuit: fails safe",
     "Pure Pursuit cannot track the curvature-saturated cusp. The speed governor halts it before the collision circles touch."),
]
GRASS, URBAN_GROUND = "#16231d", "#1c2026"
PALETTE = ["#9ca3af", "#e5e7eb", "#b91c1c", "#1e3a8a", "#166534", "#78350f", "#374151", "#a1a1aa", "#0e7490", "#7c2d12"]
APPROACHES = {"N": NORTH, "E": EAST, "S": SOUTH, "W": WEST}
HIGHWAY_OFFSETS = [LANE_WIDTH, 0.0, -LANE_WIDTH]
# (lane offset, replay lag in seconds, front-bumper starts, car lengths): cars in the lanes beside the ego
BACKGROUND_LANES = [
    (LANE_WIDTH, 3.0, [110.0, 75.0, 42.0, 10.0, -25.0], [4.6, 4.4, 5.2, 4.5, 4.8]),
    (-LANE_WIDTH, -4.0, [135.0, 98.0, 62.0, 28.0, -8.0], [4.5, 5.0, 4.4, 4.7, 4.5]),
]
APPROACH_COLORS = {"N": BLUE, "E": "#10b981", "S": ORANGE, "W": "#a855f7"}
APPROACH_NAMES = {"N": "the north", "E": "the east", "S": "the south", "W": "the west"}
STATE_LABELS = ["approaching", "stopped", "proceeding"]


def _arr(values, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _track(x, y, theta, digits: int = 3) -> dict:
    return {"x": _arr(x, digits), "y": _arr(y, digits), "th": _arr(theta, digits + 1)}


def _prov(real: str, simulated: str) -> dict:
    return {"real": real, "simulated": simulated}


def _car(track: str, color: str, label: str, **extra) -> dict:
    return {"track": track, "style": "solid", "color": color, "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
            "label": label, **extra}


def _ghost(track: str, color: str, label: str) -> dict:
    return {**_car(track, color, label), "style": "ghost"}


def _parking_scene(scenario_name: str, controller_name: str, title: str, blurb: str) -> dict:
    scenario = load_scenario(scenario_name)
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    harness = ParkingHarness(
        scenario.vehicle, scenario.environment, PLANNERS["hybrid_astar"](), controller, seed=scenario.seed
    )
    result = harness.run(max_steps=1000)
    env = scenario.environment
    n = len(result.true_history)
    seconds = n * result.dt

    if result.success:
        outcome = {"label": "Parked", "tone": "ok", "detail": f"Reached the spot in {seconds:.1f} s."}
    elif result.collision:
        outcome = {"label": "Collision", "tone": "bad", "detail": f"Hit an obstacle after {seconds:.1f} s."}
    else:
        outcome = {"label": "Stopped short", "tone": "warn", "detail": "Never reached the spot; halted before a collision."}

    (xmin, xmax), (ymin, ymax) = _axis_bounds(result, env, pad=4.0)
    est_err = np.hypot(result.estimated_history[:, 0] - result.true_history[:, 0],
                       result.estimated_history[:, 1] - result.true_history[:, 1])
    ellipses = np.array([_ellipse_params(c[:2, :2]) for c in result.covariance_history])
    ellipses[:, 2] = np.radians(ellipses[:, 2])
    path = result.path[:, :2] if result.path is not None else np.zeros((0, 2))

    return {
        "id": f"{scenario_name}-{controller_name}".replace("_", "-"),
        "group": "Parking",
        "title": title,
        "subtitle": f"Hybrid A* + {'MPC' if controller_name == 'mpc' else 'Pure Pursuit'}",
        "blurb": blurb,
        "kind": "parking",
        "dt": result.dt,
        "n": n,
        "outcome": outcome,
        "bounds": [xmin, xmax, ymin, ymax],
        "spot": {"x": env.spot.x, "y": env.spot.y, "th": env.spot.theta, "w": env.spot.width, "l": env.spot.length},
        "obstacles": [{"x": o.x, "y": o.y, "r": o.radius} for o in env.obstacles],
        "collision_radius": VEHICLE_RADIUS,
        "path": _arr(path[:: max(1, len(path) // 300)], 2),
        "tracks": {
            "ego": _track(*result.true_history.T),
            "est": _track(*result.estimated_history.T),
        },
        "vehicles": [_car("ego", BLUE, "Vehicle (ground truth)", steer="delta"), _ghost("est", ORANGE, "EKF estimate")],
        "trails": [{"track": "ego", "color": BLUE}, {"track": "est", "color": ORANGE}],
        "signals": {"v": _arr(result.controls[:, 0]), "delta": _arr(result.controls[:, 1], 4), "err": _arr(est_err)},
        "hud": [
            {"key": "v", "label": "Speed", "unit": "m/s", "fmt": 2},
            {"key": "delta", "label": "Steering", "unit": "deg", "fmt": 1, "scale": 180 / np.pi},
            {"key": "err", "label": "Estimate error", "unit": "m", "fmt": 2},
        ],
        "rays": {"angles": _arr(result.sensor_angles, 3), "ranges": _arr(result.sensor_ranges, 2), "track": "ego"},
        "uncertainty": {"track": "est", "ellipse": _arr(ellipses, 3)},
        "watch": [
            "The cyan line is the Hybrid A* path; the car reverses through its cusp to reach the stall.",
            "The orange ghost and shaded ellipse are the EKF's pose estimate and 2-sigma uncertainty.",
            "The rays are the ultrasonic sensors that the speed governor brakes on.",
        ],
        "provenance": _prov("Nothing recorded: the lot is a hand-built scenario.",
                            "The vehicle, its sensors, the EKF, the planner and the controller."),
    }


@dataclass
class _FollowRun:
    pair: object
    centerline: np.ndarray
    table: np.ndarray
    result: object
    ego: tuple  # body center x, y and heading, front-bumper referenced
    lead: tuple
    gap: np.ndarray  # geometric bumper-to-bumper gap along the ego heading


def _pose_on_lane(s, centerline, table):
    return (np.interp(s, table, centerline[:, 0]), np.interp(s, table, centerline[:, 1]),
            np.interp(s, table, centerline[:, 2]))


@cache
def _follow_run(controller_name: str) -> _FollowRun:
    controller = {"mpc": MpcAccController, "idm": IDMController}[controller_name](v0=20.0)
    pair = load_following_pair()
    centerline = load_lane_centerline()
    table = build_arc_length_table(centerline)
    lead_len = pair.leader.length
    harness = FullHighwayHarness(
        centerline=centerline, lead_position=pair.leader.position, lead_speed=pair.leader.speed,
        lead_length=lead_len, acc_controller=controller, seed=1,
    )
    r = harness.run()

    # The harness measures the gap from the ego's front, so shift the ego body back by half
    # its length (center = front - L/2) and the drawn gap matches the geometry.
    half = VEHICLE_LENGTH / 2
    ego = (r.ego_x - half * np.cos(r.ego_theta), r.ego_y - half * np.sin(r.ego_theta), r.ego_theta)
    lead = _pose_on_lane(r.lead_position - lead_len / 2, centerline, table)
    rear_x = lead[0] - lead_len / 2 * np.cos(lead[2])
    rear_y = lead[1] - lead_len / 2 * np.sin(lead[2])
    gap = (rear_x - r.ego_x) * np.cos(r.ego_theta) + (rear_y - r.ego_y) * np.sin(r.ego_theta)
    return _FollowRun(pair, centerline, table, r, ego, lead, gap)


def _follow_outcome(run: _FollowRun) -> dict:
    r, low = run.result, float(run.gap.min())
    if low <= 0:
        return {"label": "Overlap", "tone": "bad", "detail": f"The drawn bodies overlap by {-low:.1f} m."}
    return {
        "label": "No collision",
        "tone": "ok",
        "detail": (f"Minimum gap {low:.1f} m over {r.times[-1]:.0f} s. The harness reports {r.min_gap:.1f} m, "
                   "because it locates the ego by nearest waypoint (2 m spacing)."),
    }


def _background_traffic(run: _FollowRun) -> tuple[dict, list, dict]:
    """IDM traffic in the lanes beside the ego, driven by the recorded flow. Returns tracks, vehicles, signals."""
    flow, table, centerline = run.pair.leader.speed, run.table, run.centerline
    tracks, vehicles, signals = {}, [], {}
    for lane, (offset, lag, fronts, lengths) in enumerate(BACKGROUND_LANES):
        pos, _ = simulate_lane(flow, fronts, lengths, lag_ticks=round(lag * 10))
        for car in range(len(fronts)):
            x, y, th = _pose_on_lane(np.clip(pos[car] - lengths[car] / 2, table[0], table[-1]), centerline, table)
            name = f"bg{lane}_{car}"
            tracks[name] = _track(x - offset * np.sin(th), y + offset * np.cos(th), th, 2)
            signals[f"on_{name}"] = ((pos[car] > table[0] + 6) & (pos[car] < table[-1] - 6)).astype(int).tolist()
            vehicles.append({**_car(name, PALETTE[(lane * 5 + car) % len(PALETTE)], "Background traffic"),
                             "length": lengths[car], "visible": f"on_{name}"})
    return tracks, vehicles, signals


def _highway_road(centerline, offsets, style: str = "highway") -> dict:
    return {"centerline": _arr(centerline[:, :2], 2), "width": LANE_WIDTH, "offsets": offsets, "style": style,
            "shoulder": 2.5}


def _follow_scene(scene_id: str, controller_name: str, title: str, subtitle: str, blurb: str) -> dict:
    run = _follow_run(controller_name)
    r = run.result
    bg_tracks, bg_vehicles, bg_signals = _background_traffic(run)
    speed = np.asarray(r.ego_speed)
    return {
        "id": scene_id,
        "group": "Highway and intersections",
        "title": title,
        "subtitle": subtitle,
        "blurb": blurb,
        "kind": "highway",
        "ground": GRASS,
        "grid": False,
        "dt": 0.1,
        "n": len(r.times),
        "outcome": _follow_outcome(run),
        "road": _highway_road(run.centerline, HIGHWAY_OFFSETS),
        "props": scenery.highway_props(run.centerline, run.table, HIGHWAY_OFFSETS, LANE_WIDTH, seed=3),
        "tracks": {"ego": _track(*run.ego), "lead": _track(*run.lead), **bg_tracks},
        "vehicles": [
            _car("ego", BLUE, "Ego vehicle", steer="delta", tag="Ego"),
            {**_car("lead", GRAY, "Lead vehicle (recorded human driver)", tag="Lead"), "length": run.pair.leader.length,
             "width": 1.9},
            *bg_vehicles,
        ],
        "gap_pair": {"front": "lead", "rear": "ego", "key": "gap"},
        "trails": [{"track": "ego", "color": BLUE}],
        "signals": {
            "v": _arr(speed), "delta": _arr(r.ego_delta, 4), "gap": _arr(run.gap, 2), "lead_v": _arr(r.lead_speed),
            "cte": _arr(r.cross_track_error, 3), "accel": _arr(r.ego_accel, 2),
            "time_gap": _arr(np.minimum(run.gap / np.maximum(speed, 0.5), 60.0), 1), **bg_signals,
        },
        "hud": [
            {"key": "v", "label": "Ego speed", "unit": "m/s", "fmt": 1},
            {"key": "lead_v", "label": "Lead speed", "unit": "m/s", "fmt": 1},
            {"key": "gap", "label": "Gap", "unit": "m", "fmt": 1},
            {"key": "time_gap", "label": "Time gap", "unit": "s", "fmt": 1},
            {"key": "accel", "label": "Acceleration", "unit": "m/s²", "fmt": 1},
            {"key": "cte", "label": "Lane offset", "unit": "m", "fmt": 2},
        ],
        "watch": [
            "The colored bar between the cars is the gap; it turns amber and then red as it closes.",
            "Brake lights come on as the cars slow, including the ego's, through the recorded full stop.",
            "The ego holds its lane while starting 1.5 m off center.",
        ],
        "provenance": _prov(
            "The lead car's motion, the lane geometry, and the flow that drives the other lanes.",
            "The ego (controller, sensors, EKF) and the cars in the other lanes (IDM followers, no lane changes). "
            "The road furniture is illustrative."),
    }


def _stop_sign_scene() -> dict:
    centerline = load_lane_centerline()
    table = build_arc_length_table(centerline)
    n, dt, arrival, clear = 3000, 0.1, 10.0, 40.0
    # A far, fast synthetic lead keeps ACC free to cruise, the stress case for the arbiter.
    harness = FullHighwayHarness(
        centerline=centerline,
        lead_position=500.0 + 20.0 * dt * np.arange(n),
        lead_speed=np.full(n, 20.0),
        lead_length=4.5,
        acc_controller=IDMController(v0=20.0),
        seed=1,
        ego_initial_gap=400.0,
        intersection_navigator=IntersectionNavigator(stop_line_position=STOP_LINE, v_cruise=15.0),
        other_vehicle_script=other_vehicle_present_from(arrival, clear),
    )
    r = harness.run(max_steps=n)
    half = VEHICLE_LENGTH / 2
    ego_x, ego_y = r.ego_x - half * np.cos(r.ego_theta), r.ego_y - half * np.sin(r.ego_theta)
    to_line = np.maximum(STOP_LINE - np.interp(r.ego_x, centerline[:, 0], table), 0.0)

    sx, sy, sth = _pose_on_lane(STOP_LINE, centerline, table)
    cx, cy, cth = _pose_on_lane(STOP_LINE + 8.0, centerline, table)
    wx, wy, _ = _pose_on_lane(STOP_LINE + 2.6, centerline, table)
    nx, ny = -np.sin(sth), np.cos(sth)
    ticks = np.arange(len(r.times)) * dt
    cross = (ticks >= arrival) & (ticks < clear)
    if r.ego_stop_time is not None and r.proceed_time is not None and not r.ran_stop_sign:
        outcome = {"label": "Yielded, then went", "tone": "ok", "detail": (
            f"Stopped at {r.ego_stop_time:.1f} s and waited for cross traffic to clear at {clear:.0f} s. "
            f"It proceeded at {r.proceed_time:.1f} s without running the stop line.")}
    else:
        outcome = {"label": "Stop-sign violation", "tone": "bad", "detail": "The ego did not stop and yield as required."}
    props = scenery.highway_props(centerline, table, [LANE_WIDTH, 0.0], LANE_WIDTH, seed=4, rails=False)
    props += [scenery.stop_sign(sx + (LANE_WIDTH / 2 + 1.3) * np.sin(sth), sy - (LANE_WIDTH / 2 + 1.3) * np.cos(sth), sth),
              scenery.crosswalk(wx + LANE_WIDTH / 2 * nx, wy + LANE_WIDTH / 2 * ny, sth, 2 * LANE_WIDTH)]

    return {
        "id": "stop-sign-yield",
        "group": "Highway and intersections",
        "title": "Stop sign: yielding to cross traffic",
        "subtitle": "ACC + intersection navigator + Stanley",
        "blurb": ("The ego cruises, stops at the line, and yields to scripted cross traffic that arrived first. "
                  "Acceleration is the more conservative of ACC and the navigator."),
        "kind": "highway",
        "ground": GRASS,
        "grid": False,
        "dt": dt,
        "n": len(r.times),
        "outcome": outcome,
        "road": _highway_road(centerline, [LANE_WIDTH, 0.0], "arterial"),
        "props": props,
        "decor": {
            "strips": [{"x1": cx - 30 * nx, "y1": cy - 30 * ny, "x2": cx + 30 * nx, "y2": cy + 30 * ny,
                        "width": 8.0, "color": ASPHALT}],
            "bars": [{"x1": sx - LANE_WIDTH / 2 * nx, "y1": sy - LANE_WIDTH / 2 * ny,
                      "x2": sx + LANE_WIDTH / 2 * nx, "y2": sy + LANE_WIDTH / 2 * ny, "w": 0.5, "color": "#f3f4f6"}],
        },
        "zones": [{"x": cx, "y": cy, "w": 8.0, "h": 60.0, "th": cth, "color": "#fbbf24", "opacity": 0.3,
                   "t0": arrival, "t1": clear, "label": "Cross traffic present (scripted)"}],
        "tracks": {"ego": _track(ego_x, ego_y, r.ego_theta)},
        "vehicles": [_car("ego", BLUE, "Ego vehicle", steer="delta", tag="Ego")],
        "trails": [{"track": "ego", "color": BLUE}],
        "signals": {
            "v": _arr(r.ego_speed), "delta": _arr(r.ego_delta, 4), "to_line": _arr(to_line, 1),
            "state": [STATE_LABELS.index(s.name.lower()) for s in r.states], "cross": cross.astype(int).tolist(),
        },
        "hud": [
            {"key": "v", "label": "Ego speed", "unit": "m/s", "fmt": 1},
            {"key": "to_line", "label": "To stop line", "unit": "m", "fmt": 1},
            {"key": "state", "label": "Navigator", "labels": STATE_LABELS},
            {"key": "cross", "label": "Cross traffic", "labels": ["clear", "present"]},
        ],
        "watch": [
            "The ego brakes to a stop short of the white line and the STOP sign, then waits.",
            "While the amber zone is lit, scripted cross traffic has right of way; the ego goes only when it clears.",
        ],
        "provenance": _prov(
            "The lane geometry.",
            "The ego, its ACC, Stanley and navigator. Cross traffic is a scripted presence window with no "
            "trajectory; the cross street, sign and crosswalk are illustrative overlays on a freeway lane."),
    }


def _intersection_decor(legs: str, reach: float) -> dict:
    half, rw = INTERSECTION_HALF, INTERSECTION_ROAD_WIDTH / 2
    x0, x1, y0, y1 = scenery.box_rect(legs, rw, half)
    ends = {"N": (0, 0, 0, reach), "S": (0, -reach, 0, 0), "E": (0, 0, reach, 0), "W": (-reach, 0, 0, 0)}
    strips = [{"x1": ends[g][0], "y1": ends[g][1], "x2": ends[g][2], "y2": ends[g][3], "width": 2 * rw, "color": ASPHALT}
              for g in legs]
    lines = {"N": (0, y1, 0, reach), "S": (0, -reach, 0, y0), "E": (x1, 0, reach, 0), "W": (-reach, 0, x0, 0)}
    bars = [{"x1": lines[g][0], "y1": lines[g][1], "x2": lines[g][2], "y2": lines[g][3], "w": 0.15,
             "color": "#d4a72c", "opacity": 0.8} for g in legs]
    for g in legs:
        a = APPROACHES[g]
        px, py = a.position(-(half + 1.5), 3.0)
        nx, ny = -np.sin(a.heading), np.cos(a.heading)
        bars.append({"x1": px - 1.5 * nx, "y1": py - 1.5 * ny, "x2": px + 1.5 * nx, "y2": py + 1.5 * ny,
                     "w": 0.4, "color": "#f3f4f6"})
    box = {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0, "th": 0, "color": ZONE, "opacity": 1}
    return {"strips": strips, "boxes": [box], "bars": bars}


def _intersection_scene(scene_id: str, title: str, blurb: str, specs: list[VehicleSpec], max_steps: int,
                        legs: str = "NESW", watch: list | None = None) -> dict:
    result = run_multi_approach_scenario(specs, max_steps=max_steps)
    reach = max(s.start_distance for s in specs) + 25.0

    names = [v.name for v in result.vehicles]
    proceed = {v.name: float(next(t for t, s in zip(result.times, v.states, strict=True) if s.name == "PROCEEDING"))
               for v in result.vehicles}
    order = ", ".join(f"{n} ({proceed[n]:.1f} s)" for n in sorted(proceed, key=proceed.get))
    if result.collided or result.ran_stop_sign:
        outcome = {"label": "Violation", "tone": "bad", "detail": "A collision or a stop-sign violation occurred."}
    else:
        outcome = {"label": "No collision", "tone": "ok", "detail": f"Proceed order: {order}."}

    return {
        "id": scene_id,
        "group": "Highway and intersections",
        "title": title,
        "subtitle": "Navigators on 2D intersection geometry",
        "blurb": blurb,
        "kind": "intersection",
        "ground": URBAN_GROUND,
        "grid": False,
        "dt": 0.1,
        "n": len(result.times),
        "outcome": outcome,
        "bounds": [-reach, reach, -reach, reach],
        "view_bounds": [-55, 55, -55, 55],
        "decor": _intersection_decor(legs, reach),
        "props": scenery.urban_props(legs, APPROACHES, INTERSECTION_HALF, INTERSECTION_ROAD_WIDTH, 75.0, seed=len(scene_id)),
        "tracks": {v.name: _track(v.x, v.y, v.theta) for v in result.vehicles},
        "vehicles": [_car(s.approach.name, APPROACH_COLORS[s.approach.name],
                          f"From {APPROACH_NAMES[s.approach.name]}, going {s.turn}", tag=s.approach.name) for s in specs],
        "trails": [{"track": s.approach.name, "color": APPROACH_COLORS[s.approach.name]} for s in specs],
        "signals": {f"state_{v.name}": [STATE_LABELS.index(s.name.lower()) for s in v.states] for v in result.vehicles},
        "hud": [{"key": f"state_{n}", "label": f"{n}, {s.turn}", "labels": STATE_LABELS} for n, s in zip(names, specs, strict=True)],
        "watch": watch or [],
        "provenance": _prov("Nothing recorded.",
                            "Every car runs its own navigator on the same 2D geometry, with a collision check between every pair."),
    }


def _four_way_scene() -> dict:
    specs = [VehicleSpec(NORTH, start_distance=70.0), VehicleSpec(EAST, start_distance=85.0),
             VehicleSpec(SOUTH, start_distance=100.0), VehicleSpec(WEST, start_distance=115.0)]
    return _intersection_scene(
        "four-way-stop", "Four-way stop, staggered arrivals",
        "Four cars reach a four-way stop at different times and proceed in arrival order. Each runs its own "
        "navigator on real 2D geometry, with a collision check between every pair.", specs, 5000,
        watch=["Each car stops behind its white line at the STOP sign, then goes once it has right of way.",
               "The panel tracks every car's state; the first to arrive is the first to proceed."])


def _three_way_scene() -> dict:
    specs = [VehicleSpec(NORTH, start_distance=70.0), VehicleSpec(EAST, start_distance=90.0),
             VehicleSpec(WEST, start_distance=110.0)]
    return _intersection_scene(
        "three-way-stop", "Three-way stop (T intersection)",
        "A T intersection with no south leg. Three cars arrive in turn and the navigators sort out the order.",
        specs, 5000, legs="NEW",
        watch=["There is no road to the south, so the paved area is smaller on that side.",
               "Arrival order decides who goes: north, then east, then west."])


def _left_turn_scene() -> dict:
    specs = [VehicleSpec(NORTH, start_distance=70.0, turn="left"), VehicleSpec(SOUTH, start_distance=130.0)]
    return _intersection_scene(
        "left-turn-yield", "Left turn yielding to oncoming traffic",
        "The north car arrives first but is turning left, so it waits for the oncoming southbound car "
        "to clear before crossing.", specs, 6000,
        watch=["The blue car stops first but yields: left turns give way to oncoming traffic.",
               "It follows a curved Dubins path through the box once the southbound car has cleared."])


def _signalized_scene() -> dict:
    cars, reach = demo_cars(), 155.0
    res = run_signalized_scenario(cars)
    n = len(res.times)
    tracks, vehicles, signals = {}, [], {"sig_ns": res.ns_state.tolist(), "sig_ew": res.ew_state.tolist()}
    waiting, through = np.zeros(n), np.zeros(n)
    for i, c in enumerate(res.cars):
        name = f"c{i}"
        tracks[name] = _track(c.x, c.y, c.theta, 2)
        signals[f"on_{name}"] = c.active.astype(int).tolist()
        vehicles.append({**_car(name, PALETTE[i % len(PALETTE)], f"Car {i}"), "visible": f"on_{name}"})
        crossed_at = c.line_time if c.line_time is not None else np.inf
        waiting += c.active & (c.speed < 0.5) & (res.times < crossed_at)
        through += res.times >= crossed_at
    signals["waiting"], signals["through"] = waiting.astype(int).tolist(), through.astype(int).tolist()
    keys = {"N": "sig_ns", "S": "sig_ns", "E": "sig_ew", "W": "sig_ew"}
    lights = ["green", "yellow", "red"]
    return {
        "id": "signalized-intersection",
        "group": "Highway and intersections",
        "title": "Signalized intersection",
        "subtitle": "Fixed-time signal with IDM cars",
        "blurb": ("Twenty-five cars approach a fixed-time signal: 14 s green, 3 s yellow, 2 s all-red for each "
                  "direction. Cross-street cars queue on red and discharge on green."),
        "kind": "intersection",
        "ground": URBAN_GROUND,
        "grid": False,
        "dt": 0.1,
        "n": n,
        "outcome": {"label": "No red-light entries" if not res.red_entries else "Red-light entry",
                    "tone": "ok" if not res.red_entries and not res.collided else "bad",
                    "detail": f"{int(through[-1])} of {len(cars)} cars cleared the stop line in {res.times[-1]:.0f} s "
                              f"with no collision; the smallest same-lane gap was {res.min_gap:.1f} m."},
        "bounds": [-reach, reach, -reach, reach],
        "view_bounds": [-55, 55, -55, 55],
        "decor": _intersection_decor("NESW", reach),
        "props": scenery.urban_props("NESW", APPROACHES, INTERSECTION_HALF, INTERSECTION_ROAD_WIDTH, 75.0, seed=11,
                                     signal_keys=keys),
        "tracks": tracks,
        "vehicles": vehicles,
        "legend": [{"color": BLUE, "text": "Cars (IDM, straight through)"},
                   {"color": "#22c55e", "text": "Signal heads: the lit lamp is the current phase"}],
        "signals": signals,
        "hud": [
            {"key": "sig_ns", "label": "North-south", "labels": lights},
            {"key": "sig_ew", "label": "East-west", "labels": lights},
            {"key": "waiting", "label": "Cars waiting", "unit": "", "fmt": 0},
            {"key": "through", "label": "Cars through", "unit": "", "fmt": 0},
        ],
        "watch": [
            "Cars slow for yellow only if they cannot clear the line, and stop behind the white bar on red.",
            "Queues build on the red approaches and release together when the light turns green.",
            "Every car uses the stack's IDM controller; there are no turns or pedestrians.",
        ],
        "provenance": _prov("Nothing recorded.",
                            "The signal plan and all 25 cars, including their spawn times. The scenery is illustrative."),
    }


@cache
def _kitti_seed_stats() -> dict:
    seq = load_kitti_poses(DEFAULT_POSES_PATH)
    runs = [validate(seq, seed=s) for s in range(KITTI_SEEDS)]
    ekf, dr = np.array([r.ekf_rmse for r in runs]), np.array([r.dr_rmse for r in runs])
    return {"ekf": float(np.median(ekf)), "dr": float(np.median(dr)), "wins": int((ekf < dr).sum())}


def _kitti_scene(scene_id: str = "kitti-ekf", title: str = "Real KITTI drive: EKF vs dead reckoning",
                 subtitle: str = "KITTI Odometry sequence 09, frames 840-1139", blurb: str | None = None,
                 **noise) -> dict:
    seq = load_kitti_poses(DEFAULT_POSES_PATH)
    res = validate(seq, seed=0, **noise)
    speed = np.append(seq.v, seq.v[-1])
    x0, y0 = res.true[0, :2]
    route = np.column_stack([res.true[:, 0] - x0, -(res.true[:, 1] - y0)])

    def flip(pose: np.ndarray) -> dict:
        # KITTI's ground-plane frame is left-handed (y to the right); negating y and heading
        # makes left turns render as left turns. Errors are unchanged (an isometry).
        return _track(pose[:, 0] - x0, -(pose[:, 1] - y0), -pose[:, 2])

    stats = _kitti_seed_stats()
    if noise:
        detail = f"Dead reckoning: {res.dr_rmse:.3f} m RMSE, the same odometry noise as the first KITTI scene."
    else:
        detail = (f"Dead reckoning: {res.dr_rmse:.3f} m RMSE on this noise draw. Over {KITTI_SEEDS} draws the medians are "
                  f"{stats['ekf']:.2f} m for the EKF and {stats['dr']:.2f} m for dead reckoning, and the EKF wins "
                  f"{stats['wins']} of {KITTI_SEEDS}.")
    return {
        "id": scene_id,
        "group": "Real data",
        "title": title,
        "subtitle": subtitle,
        "blurb": blurb or (
            "The path is a real 30 s urban drive from KITTI ground truth. Sensor noise is simulated on top, "
            "so this compares estimators, not sensors. Dead reckoning drifts, the EKF stays on the road."),
        "kind": "kitti",
        "ground": URBAN_GROUND,
        "grid": False,
        "dt": float(seq.times[1] - seq.times[0]),
        "n": len(seq.x),
        "outcome": {"label": f"EKF {res.ekf_rmse:.3f} m RMSE", "tone": "ok", "detail": detail},
        "road": {"centerline": _arr(route[::3], 2), "width": 6.5},
        "props": scenery.route_props(route[::3], 6.5, seed=5),
        "tracks": {"truth": flip(res.true), "ekf": flip(res.ekf), "dr": flip(res.dead_reckoning)},
        "vehicles": [_car("truth", BLUE, "Ground truth (KITTI)", tag="Truth"), _ghost("ekf", ORANGE, "EKF estimate"),
                     _ghost("dr", RED, "Dead reckoning")],
        "trails": [{"track": "truth", "color": BLUE}, {"track": "ekf", "color": ORANGE}, {"track": "dr", "color": RED}],
        "signals": {"v": _arr(speed, 2), "ekf_err": _arr(res.ekf_err, 2), "dr_err": _arr(res.dr_err, 2)},
        "hud": [
            {"key": "v", "label": "Speed", "unit": "m/s", "fmt": 1},
            {"key": "ekf_err", "label": "EKF error", "unit": "m", "fmt": 2},
            {"key": "dr_err", "label": "Dead-reckoning error", "unit": "m", "fmt": 2},
        ],
        "watch": [
            "The red ghost (dead reckoning) drifts away through the turns; the orange ghost (EKF) stays close.",
            "The EKF is pulled back by a position fix every second and a compass reading every tick.",
        ],
        "provenance": _prov(
            "The vehicle's path, speed and heading (KITTI Odometry ground truth).",
            "The odometry, compass and position-fix noise, and both estimators. The road and buildings are "
            "illustrative; only the path is from KITTI."),
    }


def _kitti_noisy_scene() -> dict:
    return _kitti_scene(
        "kitti-noisy-fixes", "Real KITTI drive with noisier position fixes", "Position noise 2.0 m instead of 0.3 m",
        "The same drive and odometry noise, with position fixes almost seven times noisier. The EKF degrades "
        "gracefully and still tracks far better than dead reckoning.", position_std=2.0)


def _follower_scene() -> dict:
    run = _follow_run("mpc")
    r, pair = run.result, run.pair
    human = pair.follower
    n = min(len(r.times), len(human.position))
    hx, hy, hth = _pose_on_lane(human.position[:n] - human.length / 2, run.centerline, run.table)
    hx, hy = hx + LANE_WIDTH * np.sin(hth), hy - LANE_WIDTH * np.cos(hth)  # drawn one lane to the right
    ours, theirs = run.gap[:n], pair.real_space_headway[:n]

    def cut(track):
        return _track(*(a[:n] for a in track))

    return {
        "id": "ngsim-follower",
        "group": "Real data",
        "title": "Ours vs a recorded human follower",
        "subtitle": "MPC-ACC against NGSIM US-101 traffic",
        "blurb": ("Behind the same recorded leader, our MPC-ACC car (blue) drives beside the recorded human "
                  "follower (orange, drawn one lane over). The human's gap is exact NGSIM data; ours is simulated."),
        "kind": "highway",
        "ground": GRASS,
        "grid": False,
        "dt": 0.1,
        "n": n,
        "outcome": {"label": "Ours vs recorded gap", "tone": "ok", "detail": (
            f"Mean gap {ours.mean():.1f} m for our controller and {theirs.mean():.1f} m for the human. "
            "Our car starts 15 m back and 1.5 m off center.")},
        "road": _highway_road(run.centerline, HIGHWAY_OFFSETS),
        "props": scenery.highway_props(run.centerline, run.table, HIGHWAY_OFFSETS, LANE_WIDTH, seed=6),
        "tracks": {"ego": cut(run.ego), "lead": cut(run.lead), "human": _track(hx, hy, hth)},
        "vehicles": [
            _car("ego", BLUE, "Our MPC-ACC car", steer="delta", tag="Ours"),
            {**_car("lead", GRAY, "Recorded leader", tag="Leader"), "length": pair.leader.length, "width": 1.9},
            {**_car("human", ORANGE, "Recorded human follower (next lane)", tag="Human"), "length": human.length,
             "width": 1.9},
        ],
        "gap_pair": {"front": "lead", "rear": "ego", "key": "gap"},
        "trails": [{"track": "ego", "color": BLUE}, {"track": "human", "color": ORANGE}],
        "signals": {
            "v": _arr(r.ego_speed[:n]), "human_v": _arr(human.speed[:n]), "delta": _arr(r.ego_delta[:n], 4),
            "gap": _arr(ours, 2), "human_gap": _arr(theirs, 2),
        },
        "hud": [
            {"key": "gap", "label": "Our gap", "unit": "m", "fmt": 1},
            {"key": "human_gap", "label": "Human gap", "unit": "m", "fmt": 1},
            {"key": "v", "label": "Our speed", "unit": "m/s", "fmt": 1},
            {"key": "human_v", "label": "Human speed", "unit": "m/s", "fmt": 1},
        ],
        "watch": [
            "Compare the two gaps in the panel: both cars follow the same leader through the same stop.",
            "The orange car is the real driver; it is shown one lane over so the two cars do not overlap.",
        ],
        "provenance": _prov(
            "The leader's and the human follower's motion, and the lane geometry (NGSIM US-101).",
            "Our controller, its sensors and the EKF. The human is placed in the next lane for display only."),
    }


def _lane_scene() -> dict:
    offset, speed = 3.0, 20.0
    res = validate_lane_centering(initial_offset=offset, speed=speed)
    n = len(res.vehicle_x)
    centerline = load_lane_centerline()
    table = build_arc_length_table(centerline)
    return {
        "id": "ngsim-lane",
        "group": "Real data",
        "title": "Lane centering on a real NGSIM lane",
        "subtitle": "Stanley, 3 m initial offset",
        "blurb": ("Stanley steering on a lane centerline derived from real NGSIM vehicle positions, "
                  f"starting {offset:.0f} m off center at {speed:.0f} m/s."),
        "kind": "lane",
        "ground": GRASS,
        "grid": False,
        "dt": 0.1,
        "n": n,
        "outcome": {"label": f"Settles to {res.max_cte_after_settling:.2f} m", "tone": "ok", "detail": (
            f"Largest offset after 150 m is {res.max_cte_after_settling:.2f} m. "
            f"Real drivers on this lane scatter {REAL_LATERAL_STD_M} m.")},
        "road": _highway_road(centerline, HIGHWAY_OFFSETS),
        "props": scenery.highway_props(centerline, table, HIGHWAY_OFFSETS, LANE_WIDTH, seed=7),
        "tracks": {"ego": _track(res.vehicle_x, res.vehicle_y, res.vehicle_theta)},
        "vehicles": [_car("ego", BLUE, "Ego vehicle", steer="delta", tag="Ego")],
        "trails": [{"track": "ego", "color": BLUE}],
        "signals": {"v": _arr(np.full(n, speed)), "delta": _arr(res.delta, 4), "cte": _arr(res.cross_track_error, 3)},
        "hud": [
            {"key": "v", "label": "Speed", "unit": "m/s", "fmt": 1},
            {"key": "delta", "label": "Steering", "unit": "deg", "fmt": 1, "scale": 180 / np.pi},
            {"key": "cte", "label": "Lane offset", "unit": "m", "fmt": 2},
        ],
        "watch": [
            "The car steers hard at first and eases as it converges on the lane center.",
            "The trail behind it shows the correction path back to center.",
        ],
        "provenance": _prov("The lane centerline, derived from real NGSIM vehicle positions.",
                            "The car, the Stanley controller and the drive along the lane. The scenery is illustrative."),
    }


def build_scenes() -> list[dict]:
    builders = [
        *[lambda a=a: _parking_scene(*a) for a in PARKING_SCENES],
        lambda: _follow_scene(
            "highway-ngsim", "mpc", "Highway following a real driver", "MPC-ACC + Stanley on NGSIM US-101",
            "The lead car replays a recorded human driver, including a full stop. The ego holds a safe gap "
            "with MPC adaptive cruise control and stays in lane with Stanley, starting 1.5 m off center."),
        lambda: _follow_scene(
            "highway-idm", "idm", "Highway following with IDM", "IDM-ACC + Stanley on NGSIM US-101",
            "The same recorded leader, lane and sensor noise, with the closed-form Intelligent Driver Model "
            "in place of MPC. Only the acceleration controller changes."),
        _stop_sign_scene,
        _four_way_scene,
        _three_way_scene,
        _left_turn_scene,
        _signalized_scene,
        _kitti_scene,
        _kitti_noisy_scene,
        _follower_scene,
        _lane_scene,
    ]
    scenes = []
    for build in builders:
        scene = build()
        print(f"{scene['id']}: {scene['outcome']['label']} ({scene['n']} ticks)")
        scenes.append(scene)
    return scenes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export simulation runs for the browser 3D viewer.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output scenes.js path")
    args = parser.parse_args(argv)

    scenes = build_scenes()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.AV_SCENES = " + json.dumps(scenes, separators=(",", ":")) + ";\n")
    print(f"Wrote {len(scenes)} scenes to {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
