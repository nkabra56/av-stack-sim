"""Exports simulation runs as JSON-in-JS for the browser 3D viewer in docs/viewer/.
Each scene is data-driven: tracks (x, y, heading per tick), signals, and static geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from core.control.acc import MpcAccController
from core.control.lane_geometry import build_arc_length_table
from core.demo import CONTROLLERS, PLANNERS
from core.environment import VEHICLE_RADIUS
from core.full_highway_harness import FullHighwayHarness
from core.harness import ParkingHarness
from core.scenario_loader import load_scenario
from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH, validate
from core.validation.kitti_loader import load_kitti_poses
from core.validation.ngsim_loader import load_following_pair, load_lane_centerline
from core.visualization.animate import VEHICLE_LENGTH, VEHICLE_WIDTH, _axis_bounds, _ellipse_params

DEFAULT_OUT = Path(__file__).parents[2] / "docs" / "viewer" / "scenes.js"
LANE_WIDTH = 3.7
PARKING_SCENES = [
    ("perpendicular_open", "mpc", "Perpendicular parking, open lane",
     "The baseline case: a clear approach and one stall. Hybrid A* plans, MPC tracks."),
    ("parallel_open", "mpc", "Parallel parking, open curb",
     "A 2.6 m x 6.7 m stall along a clear curb, entered in reverse."),
    ("perpendicular_flanked", "mpc", "Perpendicular, flanked by parked cars",
     "The direct Dubins path clips the left neighbor, so Hybrid A* routes around it."),
    ("perpendicular_obstructed_lane", "mpc", "Perpendicular, obstructed lane",
     "A car blocks the drive aisle; the planner detours before turning into the stall."),
    ("parallel_between_cars", "mpc", "Parallel parking between two cars",
     "The tightest case: a reverse-gear cusp between parked cars, tracked by MPC."),
    ("parallel_between_cars", "pure_pursuit", "Same scene with Pure Pursuit: fails safe",
     "Pure Pursuit cannot track the curvature-saturated cusp. The speed governor halts it before the collision circles touch."),
]


def _arr(values, digits: int = 3) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _track(x, y, theta) -> dict:
    return {"x": _arr(x), "y": _arr(y), "th": _arr(theta, 4)}


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
        "spot": {"x": env.spot.x, "y": env.spot.y, "th": env.spot.theta, "size": env.spot.size},
        "obstacles": [{"x": o.x, "y": o.y, "r": o.radius} for o in env.obstacles],
        "collision_radius": VEHICLE_RADIUS,
        "path": _arr(path[:: max(1, len(path) // 300)], 2),
        "tracks": {
            "ego": _track(*result.true_history.T),
            "est": _track(*result.estimated_history.T),
        },
        "vehicles": [
            {"track": "ego", "style": "solid", "color": "#2f6fe4", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "steer": "delta", "label": "Vehicle (ground truth)"},
            {"track": "est", "style": "ghost", "color": "#f59e0b", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "label": "EKF estimate"},
        ],
        "trails": [{"track": "ego", "color": "#2f6fe4"}, {"track": "est", "color": "#f59e0b"}],
        "signals": {"v": _arr(result.controls[:, 0]), "delta": _arr(result.controls[:, 1], 4), "err": _arr(est_err)},
        "hud": [
            {"key": "v", "label": "Speed", "unit": "m/s", "fmt": 2},
            {"key": "delta", "label": "Steering", "unit": "deg", "fmt": 1, "scale": 180 / np.pi},
            {"key": "err", "label": "Estimate error", "unit": "m", "fmt": 2},
        ],
        "rays": {"angles": _arr(result.sensor_angles, 3), "ranges": _arr(result.sensor_ranges, 2), "track": "ego"},
        "uncertainty": {"track": "est", "ellipse": _arr(ellipses, 3)},
    }


def _highway_scene() -> dict:
    pair = load_following_pair()
    centerline = load_lane_centerline()
    table = build_arc_length_table(centerline)
    lead_len = pair.leader.length
    harness = FullHighwayHarness(
        centerline=centerline,
        lead_position=pair.leader.position,
        lead_speed=pair.leader.speed,
        lead_length=lead_len,
        acc_controller=MpcAccController(v0=20.0),
        seed=1,
    )
    r = harness.run()

    # The harness measures the gap from the ego's front, so shift the ego body back by half
    # its length (center = front - L/2) and the drawn gap matches the measured one.
    half = VEHICLE_LENGTH / 2
    ego_cx = r.ego_x - half * np.cos(r.ego_theta)
    ego_cy = r.ego_y - half * np.sin(r.ego_theta)
    lead_s = r.lead_position - lead_len / 2
    lead_x = np.interp(lead_s, table, centerline[:, 0])
    lead_y = np.interp(lead_s, table, centerline[:, 1])
    lead_th = np.interp(lead_s, table, centerline[:, 2])

    return {
        "id": "highway-ngsim",
        "group": "Highway",
        "title": "Highway following a real driver",
        "subtitle": "MPC-ACC + Stanley on NGSIM US-101",
        "blurb": (
            "The lead car replays a recorded human driver, including a full stop. The ego holds a safe gap "
            "with MPC adaptive cruise control and stays in lane with Stanley, starting 1.5 m off center."
        ),
        "kind": "highway",
        "dt": harness.dt,
        "n": len(r.times),
        "outcome": {"label": "No collision", "tone": "ok", "detail": f"Minimum gap {r.min_gap:.1f} m over {r.times[-1]:.0f} s."},
        "road": {"centerline": _arr(centerline[:, :2], 2), "width": LANE_WIDTH},
        "tracks": {"ego": _track(ego_cx, ego_cy, r.ego_theta), "lead": _track(lead_x, lead_y, lead_th)},
        "vehicles": [
            {"track": "ego", "style": "solid", "color": "#2f6fe4", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "steer": "delta", "label": "Ego vehicle"},
            {"track": "lead", "style": "solid", "color": "#8b95a5", "length": lead_len, "width": 1.9,
             "label": "Lead vehicle (recorded human driver)"},
        ],
        "trails": [{"track": "ego", "color": "#2f6fe4"}],
        "signals": {
            "v": _arr(r.ego_speed), "delta": _arr(r.ego_delta, 4), "gap": _arr(r.gap, 2),
            "lead_v": _arr(r.lead_speed), "cte": _arr(r.cross_track_error, 3),
        },
        "hud": [
            {"key": "v", "label": "Ego speed", "unit": "m/s", "fmt": 1},
            {"key": "lead_v", "label": "Lead speed", "unit": "m/s", "fmt": 1},
            {"key": "gap", "label": "Gap", "unit": "m", "fmt": 1},
            {"key": "cte", "label": "Lane offset", "unit": "m", "fmt": 2},
        ],
    }


def _kitti_scene() -> dict:
    seq = load_kitti_poses(DEFAULT_POSES_PATH)
    res = validate(seq, seed=0)
    speed = np.append(seq.v, seq.v[-1])
    x0, y0 = res.true[0, :2]

    def flip(pose: np.ndarray) -> dict:
        # KITTI's ground-plane frame is left-handed (y to the right); negating y and heading
        # makes left turns render as left turns. Errors are unchanged (an isometry).
        return _track(pose[:, 0] - x0, -(pose[:, 1] - y0), -pose[:, 2])

    gain = (1 - res.ekf_rmse / res.dr_rmse) * 100
    return {
        "id": "kitti-ekf",
        "group": "Real data",
        "title": "Real KITTI drive: EKF vs dead reckoning",
        "subtitle": "KITTI Odometry sequence 09, frames 840-1139",
        "blurb": (
            "The path is a real 30 s urban drive from KITTI ground truth. Sensor noise is simulated on top, "
            "so this compares estimators, not sensors. Dead reckoning drifts, the EKF stays on the road."
        ),
        "kind": "kitti",
        "dt": float(seq.times[1] - seq.times[0]),
        "n": len(seq.x),
        "outcome": {
            "label": f"EKF {res.ekf_rmse:.3f} m RMSE",
            "tone": "ok",
            "detail": f"Dead reckoning: {res.dr_rmse:.3f} m RMSE. The EKF is {gain:.0f}% lower.",
        },
        "path": _arr(np.column_stack([res.true[:, 0] - x0, -(res.true[:, 1] - y0)]), 2),
        "tracks": {"truth": flip(res.true), "ekf": flip(res.ekf), "dr": flip(res.dead_reckoning)},
        "vehicles": [
            {"track": "truth", "style": "solid", "color": "#2f6fe4", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "label": "Ground truth (KITTI)"},
            {"track": "ekf", "style": "ghost", "color": "#f59e0b", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "label": "EKF estimate"},
            {"track": "dr", "style": "ghost", "color": "#ef4444", "length": VEHICLE_LENGTH, "width": VEHICLE_WIDTH,
             "label": "Dead reckoning"},
        ],
        "trails": [
            {"track": "truth", "color": "#2f6fe4"},
            {"track": "ekf", "color": "#f59e0b"},
            {"track": "dr", "color": "#ef4444"},
        ],
        "signals": {"v": _arr(speed, 2), "ekf_err": _arr(res.ekf_err, 2), "dr_err": _arr(res.dr_err, 2)},
        "hud": [
            {"key": "v", "label": "Speed", "unit": "m/s", "fmt": 1},
            {"key": "ekf_err", "label": "EKF error", "unit": "m", "fmt": 2},
            {"key": "dr_err", "label": "Dead-reckoning error", "unit": "m", "fmt": 2},
        ],
    }


def build_scenes() -> list[dict]:
    scenes = []
    for scenario_name, controller_name, title, blurb in PARKING_SCENES:
        scene = _parking_scene(scenario_name, controller_name, title, blurb)
        print(f"{scene['id']}: {scene['outcome']['label']} ({scene['n']} ticks)")
        scenes.append(scene)
    scenes.append(_highway_scene())
    print(f"highway-ngsim: {scenes[-1]['outcome']['label']} ({scenes[-1]['n']} ticks)")
    scenes.append(_kitti_scene())
    print(f"kitti-ekf: {scenes[-1]['outcome']['label']} ({scenes[-1]['n']} ticks)")
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
