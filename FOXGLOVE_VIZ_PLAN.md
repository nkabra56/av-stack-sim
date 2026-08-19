# 3D Foxglove visualization for a LinkedIn demo video

## Context

The only existing renderer, `core/visualization/animate.py`, produces a plain-Matplotlib
top-down 2D animation (white background, flat colors) plus two small telemetry subplots,
exported as a GIF via `--save`. It's functional for engineering validation but not visually
striking enough to showcase. The user wants a genuinely **3D** simulation with a **camera
that dynamically follows/zooms on the vehicle**, built with **Foxglove** (the robotics
visualization tool), so they can record a polished demo video for LinkedIn.

This is additive: `core/harness.py` and all planning/control/estimation code stay untouched
— it already produces a rendering-agnostic `SimulationResult`. `animate.py` and the default
`python -m core.demo ...` behavior (GIF/interactive matplotlib) are not modified or removed;
Foxglove export is a new, opt-in path alongside it.

Key caveat discovered during research (docs.foxglove.dev): Foxglove's built-in "Export
video" feature is **Enterprise-plan + desktop-app only**. On the free tier there is no
programmatic video export. The realistic path to an actual LinkedIn-ready video file is:
generate the `.mcap` → open it in the free Foxglove desktop app with a follow-camera layout
→ hit play → **screen-record the app window** (OBS Studio, Windows Game Bar, etc.) → trim if
needed. This plan builds everything up to and including "press play with the right layout
loaded"; the final screen-capture step is manual and out of scope for automation.

## Approach

### 1. New module: `core/visualization/foxglove_export.py`

Consumes the same `SimulationResult` + `Environment` that `animate.py` consumes (see
`core/harness.py:43-58` for `SimulationResult` fields, `core/environment.py` for
`Obstacle`/`Spot`/`Environment`). Reuses `animate.py`'s existing geometry helpers rather than
re-deriving them: import `VEHICLE_LENGTH`, `VEHICLE_WIDTH`, `_axis_bounds`, `_ellipse_params`
from `core/visualization/animate.py`.

Public entry point, same shape as `render_animation`:
```python
def render_foxglove(result, environment, title="", save_path="out/demo.mcap", live=False): ...
```

Scene content, all logged via `foxglove.open_mcap(save_path)` (optionally also
`foxglove.start_server()` for live viewing when `live=True`):
- **Ground/lot**: a flat thin box sized from `_axis_bounds()`, plus `LinePrimitive`s for the
  parking spot outline (rotated by `spot.theta`) and a few dashed lane-marking segments —
  logged once, static.
- **Obstacles**: one cylinder/box per `Obstacle(x, y, radius)` at z=0, ~1.5m tall (cosmetic
  "parked car" height, same license `animate.py` takes drawing them as flat circles).
- **Vehicle**: a body box (`VEHICLE_LENGTH × VEHICLE_WIDTH`) + smaller cabin box, published in
  a `"vehicle"` child frame; pose driven per-tick by a `FrameTransform` (`world` → `vehicle`,
  yaw-only quaternion from `true_history[i]`) — this is also the frame the follow-camera
  layout targets, so the car geometry itself never needs per-tick recomputation.
- **Trails**: true (`true_history`) and EKF-estimated (`estimated_history`) paths as two
  distinctly colored growing `LinePrimitive`s, re-logged in full each tick (Foxglove
  primitives aren't append-only).
- **Uncertainty ellipse**: reuse `_ellipse_params(cov_xy)` unmodified, sample it into a closed
  ring of points rendered as a `LinePrimitive` — avoids depending on an unverified
  ellipsoid-primitive schema.
- **Sensor rays**: one short `LinePrimitive` per ultrasonic beam, from the vehicle out to its
  current range reading at `theta + beam_angle`, color-graded red (close) → green (clear) —
  the 3D equivalent of the existing polar sensor panel's "fan shortens toward an obstacle."
- **Telemetry channels** for Foxglove's own Plot panels: `/speed` (json: `v`, `delta`) and
  `/sensors` (json, per-beam ranges) — same signal as `animate.py`'s speed and sensor
  subplots, now in Foxglove's native plots.
- **Status logs**: a `/status` message at tick 0 (scenario, planner, controller, seed) and one
  at the final tick (success/collision/steps), mirroring `demo.py`'s existing console print.

Keep each scene-piece builder a small pure function (geometry in, SDK object out) so the
geometry math (ellipse ring, color lerp, rotated-rectangle corners) is unit-testable without
touching the SDK or writing a file.

**Before writing real code**: `pip install foxglove-sdk` and check `help(foxglove...)` /
type stubs for the exact primitive field names (e.g. whether `SpherePrimitive` takes a scalar
`radius` or a `size: Vector3`, exact `SceneEntity`/`SceneUpdate` field names). Secondary docs
sources were used for the plan-level API sketch below and aren't guaranteed byte-accurate:

```python
import foxglove
from foxglove import Channel
from foxglove.channels import SceneUpdateChannel
from foxglove.messages import (Color, CubePrimitive, LinePrimitive, SceneEntity,
                                SceneUpdate, Vector3, FrameTransform, Quaternion)
```

### 2. `pyproject.toml` — new optional extra

Add a `viz` extra, following the exact precedent already set by `rl = ["gymnasium", "stable-baselines3"]`
(pyproject.toml:23) — a short comment explaining it's opt-in, not core, same rationale:
```toml
viz = ["foxglove-sdk"]
```
Do not add `foxglove-sdk` to core `dependencies`.

### 3. `core/demo.py` — CLI wiring

Add a new flag, independent of `--save` (keeps GIF-writer and Foxglove-writer as separate,
non-conflated backends):
```python
parser.add_argument("--foxglove", metavar="PATH.mcap", help="Export a 3D scene for Foxglove instead of the matplotlib animation (needs `pip install -e \".[viz]\"`)")
```
Branch after `harness.run(...)` (`core/demo.py:60-69`): if `--foxglove` is given, lazily
`import` and call `render_foxglove(...)` instead of `render_animation(...)` — the lazy import
means the default `python -m core.demo ...` path never imports `foxglove-sdk` even if it
isn't installed. Mutually exclusive with `--save`/default animation, matching the "instead of"
framing.

### 4. New file: `foxglove-layouts/parking_demo.json` — the follow-camera

Not hand-authored. Manual one-time step: generate an `.mcap`, open it in the free Foxglove
desktop app, add a 3D panel, set its follow mode to Position+Rotation targeting the `vehicle`
frame, add Plot panel(s) wired to `/speed.*` and `/sensors.*`, then export the layout JSON via
the app and commit it at this path. This is how "dynamic follow/zoom" gets satisfied — a
configured camera-follow setting, not custom camera-path code.

### 5. `README.md`

Add a Quickstart line (`pip install -e ".[viz]"` then `python -m core.demo perpendicular_open --foxglove out/demo.mcap`)
and a short new "3D visualization (Foxglove)" subsection: install the free Foxglove desktop
app (separate manual install, not automated here), open the `.mcap`, load the committed
layout, hit play, and screen-record the window for a LinkedIn video — explicitly noting
Foxglove's built-in video export is Enterprise-only and doesn't apply. No Docker or ffmpeg
changes needed (ffmpeg stays relevant only to the existing, untouched `--save out.gif` path).

### 6. Tests: `tests/test_foxglove_export.py`

Follow the existing `pytest.importorskip` precedent (`tests/test_rl_training.py:13`,
`tests/test_rl_comparison.py:12`) so this stays skipped, not failed, in the default
`dev`-only environment / current Docker image / CI:
```python
pytest.importorskip("foxglove")
```
Test body: run a short scenario (`DubinsPlanner`, small `max_steps`, matching the "fast smoke
test" style of the existing RL tests) through `ParkingHarness`, call `render_foxglove(...)`
into a `tmp_path` file, assert the file exists and is non-empty. Optionally add a second,
dependency-free unit test directly on the pure geometry helpers (ellipse ring, color lerp) if
they're factored out as plain functions returning tuples/floats.

**Not automatable** (document, don't test): whether the scene actually looks right (car
shape, smooth camera follow, ellipse growing/shrinking, sensor color grading) requires a
human opening the `.mcap` in the Foxglove desktop app — a separate manual install this plan
doesn't automate.

## Files touched

- `core/visualization/foxglove_export.py` — new
- `core/visualization/animate.py` — read-only reuse of `VEHICLE_LENGTH`, `VEHICLE_WIDTH`,
  `_axis_bounds`, `_ellipse_params`; no behavior change
- `core/demo.py` — add `--foxglove` flag
- `pyproject.toml` — add `viz` extra
- `tests/test_foxglove_export.py` — new
- `foxglove-layouts/parking_demo.json` — new, exported from the Foxglove app
- `README.md` — Quickstart line + new subsection

## Verification

1. `pip install -e ".[dev]"` (no `viz`) → `pytest -q` still fully green, new test shows as
   **skipped**, nothing else regresses.
2. `pip install -e ".[dev,viz]"` → `pytest -q` → the new Foxglove test now **passes**.
3. `python -m core.demo perpendicular_open --foxglove out/demo.mcap` runs cleanly, prints the
   same success/collision status line as today, and `out/demo.mcap` exists and is non-empty.
4. Manual: open `out/demo.mcap` in the free Foxglove desktop app, load
   `foxglove-layouts/parking_demo.json`, confirm the 3D scene (car, obstacles, trails,
   uncertainty ellipse, sensor fan) renders correctly and the camera follows the vehicle
   through the maneuver; play through once as a dry run for the eventual screen recording.
5. Confirm `python -m core.demo perpendicular_open --save out/demo.gif` (no `--foxglove`)
   still behaves exactly as before — default path unaffected.
