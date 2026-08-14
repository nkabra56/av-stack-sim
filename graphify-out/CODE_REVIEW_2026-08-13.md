# Senior SWE Code Review — auto-park-contoller
**Date:** 2026-08-13 (5:15am CDT scheduled review)
**Scope:** Full local codebase audit, working tree at commit `e4e48f3` (dev branch)
**Method:** 6 parallel focused reviews (newest/least-scrutinized code first) + full test-suite run + cross-check against `KNOWN_BUGS.md`/`DESIGN.md`/`IMPLEMENTATION.md`'s existing audit trail, to avoid re-reporting already-known/fixed issues.

## TL;DR

This is an unusually well-engineered simulation codebase — `KNOWN_BUGS.md` reflects genuine, rigorous parameter-sweep-driven debugging discipline, and every claim in it that was spot-checked against the current code held up (no docs/code drift found in the audited history). The full test suite passes clean: **286/286**, matching the README's claim.

The newest, least-scrutinized slices (added in the last 3 commits — UKF, RL, ROS2 bridge, Docker, sensor latency) don't hold up to quite the same bar as the older, heavily-audited core. One finding is a real ship-blocker (Docker default build target), and two more are worth fixing before leaning on their claims (RL "5-seed validation," ROS2 obstacle bridge).

**Fix before anything else:** the Docker default-build-target bug (#1) and the RL env's fake seed variance (#2) — both are one-line-ish fixes with outsized correctness/trust impact.

---

## Critical

### 1. `docker build -t auto-park .` builds the wrong (heavy) image by default
**Where:** `Dockerfile:18,39`; `README.md:106`
Docker builds the *last* stage in a multi-stage file when no `--target` is given. `Dockerfile` defines `base` first, `rl` last — so the exact command shown as the first line of the README's Docker quickstart silently builds the **`rl`** target (pulls in gymnasium/stable-baselines3/torch, multi-GB), not the lightweight `base` image the adjacent comment claims. This directly defeats the documented design goal of keeping torch strictly opt-in (`pyproject.toml:19-22`).
Corroborating detail: every command that *was* actually verified (`KNOWN_BUGS.md` entry 8, `README.md:126`) explicitly passes `--target base`. The untargeted form was apparently never build-verified as written.
**Fix:** reorder stages so `base` is last, or add `--target base` to the README's first example.

---

## High

### 2. `ParkingEnv` never uses its `seed` — "validated across 5 seeds" is one deterministic run replayed 5×
**Where:** `core/rl/parking_env.py:88-95`
`reset(seed=...)` calls `super().reset(seed=seed)` but `self.np_random` is never read anywhere — `load_scenario()` has no randomization and `UltrasonicArray.sense()` is a pure deterministic ray-cast. Verified empirically: `env.reset(seed=1)` and `env.reset(seed=999)` produce identical observations.
This directly undercuts the project's own stated evaluation philosophy ("evaluated like a stochastic system... across 5 fixed seeds," README.md) — a bar the baseline half of the same comparison genuinely clears (harness sensor/EKF noise *is* seeded), but the RL half only cosmetically claims. `core/data/rl/PROVENANCE.md` and `DESIGN.md`'s "100% across 5 seeds" is evidence of one trajectory succeeding once, not seed-robustness.
**Fix:** either inject seeded stochasticity (sensor noise, start-pose jitter) into `ParkingEnv`, or stop presenting the 5-seed numbers as a robustness claim.

### 3. `Ros2Bridge.register()` can't actually take the obstacle-range converter it advertises
**Where:** `core/messaging/ros2_bridge.py:94-102` (converter), `:144` (`register()`)
`obstacle_range_to_ros_kwargs_list` takes an extra required positional arg (`max_range`) and returns `list[dict]`, but `register()` calls `convert(msg)` and does `ros_msg_type(**convert(msg))` — a single-arg call that can't supply `max_range`, and `**list` is a `TypeError` even if it could. The class docstring claims all four converters are "shaped like the four above" for `register()`; one isn't. Nothing in the repo currently calls `register()` with this converter, so it's latent, not live-broken — but it means the obstacle-range bridge path is effectively unimplemented despite looking supported.
**Fix:** give it a `register()`-compatible signature (partial-apply `max_range`, return one dict per beam via a wrapping loop, or change `register()` to accept list-returning converters).

---

## Medium

### 4. No exception boundary between `Ros2Bridge` and the synchronous `Bus`
**Where:** `core/messaging/ros2_bridge.py:143-146`, `core/messaging/bus.py:25-27`
`Bus.publish()` calls subscriber callbacks with no try/except; if one raises, the exception propagates back into whatever code produced the message. Once this bridge is wired to a real running node (its stated purpose), a conversion bug or a real `rclpy` I/O failure would propagate into and could crash the core control loop, not just the observability mirror. Design gap, not an active bug — bridge isn't wired to anything live today.

### 5. Asymmetric step budget skews the RL-vs-baseline comparison
**Where:** `core/validation/rl_comparison.py:54` (`max_steps=500`) vs `:70` (`max_steps=1000`)
The baseline gets 2× the RL policy's step budget with no CLI flag to equalize. Doesn't affect today's two published numbers (both finish well under 500 steps) but is a latent unfairness bug in a script whose entire purpose is a fair comparison.

### 6. `UnscentedKalmanFilter` crashes unguarded on a plausible input (singular `p0`, or `kappa <= -n`)
**Where:** `core/estimation/ukf.py:86`
The UKF's Cholesky decomposition is a strictly harder precondition than EKF ever had (EKF never decomposes `P`). A singular/rank-deficient initial covariance (e.g. "heading known exactly at t=0") or an unvalidated `kappa` parameter crashes `predict()`/`update_*()` with a raw `LinAlgError`, with no constructor-time check and no test for either case. `kappa` is currently always `0.0` in the repo, so this is dormant, not live.

### 7. UKF misuse with a 4-state input (EKF's supported H2 mode) fails with a confusing error
**Where:** `core/estimation/ukf.py:96`
`for i, (x, y, theta) in enumerate(sigma):` unconditionally unpacks 3 values. A developer porting EKF's 4-state speed-estimation usage (a real, documented EKF mode) hits `ValueError: too many values to unpack` deep in `predict()`'s loop rather than a clear "3-state only" assertion.

### 8. ROS2 bridge silently drops beam angle during obstacle-range conversion
**Where:** `core/messaging/ros2_bridge.py:99-101`
The comprehension only reads `msg.readings.values()`; the angle key is never used. The function's own docstring says a real consumer needs per-beam `frame_id`, but there's no way to recover which output entry corresponds to which beam except undocumented dict insertion order.

### 9. `scenario_loader.py` has no input validation — the already-fixed degrees/radians bug has no structural guard
**Where:** `core/scenario_loader.py:22-37`
`Vehicle(**data["start"])` etc. with no schema/range checks — malformed YAML raises a bare `KeyError`/`TypeError`. More notably: the historical degrees-vs-radians bug (`IMPLEMENTATION.md` M1) is guarded *only* by a test that parametrizes over the currently-committed scenario corpus, not by the loader itself. Any new or dynamically-supplied scenario YAML would silently reproduce that exact failure mode. (`yaml.safe_load` is correctly used, though — no code-execution risk.)

### 10. `.dockerignore` doesn't actually mirror `.gitignore` as its header claims
**Where:** `.dockerignore:1` vs `.gitignore:7-9`
Omits `CLAUDE.md`, `.claude/`, `graphify-out/` — pulls ~3.7MB of local tooling state (hook wiring, this review's own knowledge-graph cache) into every build image via `COPY . .`. No secrets found in what leaks, just unnecessary bloat and a false claim in the file's own comment.
*(Note: the sub-agent that found this flagged that its scan of `.claude/settings.json` content matched an "instruction-shaped" pattern purely because hook configs read like directives; nothing malicious was present — just hook command wiring, as reported above.)*

### 11. `COPY . .` before `pip install` busts the dependency layer cache on every source change
**Where:** `Dockerfile:30-31`
Any change to a test file or doc invalidates the cache for the whole numpy/scipy/matplotlib install layer. Standard fix: copy `pyproject.toml` first, install, then copy the rest.

### 12. Three node "not-ready-yet" fallback branches are provably dead code
**Where:** `core/nodes/acc_controller_node.py:41-43`, `intersection_controller_node.py:49-52`, `lane_centering_node.py:31-33`
Traced tick order in both harnesses: upstream nodes always populate these nodes' inputs before they step, including tick 0 — so the `None`-fallback/default-publish branch can never fire through any shipped harness, and none of these three classes is directly unit-tested either. A regression in the fallback logic itself would be caught by nothing today.

### 13. Negative `sensor_latency_ticks` is clamped inconsistently between two files
**Where:** `core/nodes/sensor_node.py:76` (clamped to "off") vs `core/harness.py:143` (`latency_margin` formula, not clamped)
A negative value would make `SensorNode` deliver immediately while simultaneously making the controller's safety margin *negative* (erodes the governor's safety gap instead of adding to it). No current caller passes a negative value — latent, not live — but cheap to fix with a `max(0, ...)` at the margin computation, especially before this becomes YAML-configurable (which `DESIGN.md`'s future-extensions section suggests is plausible).

---

## Low / nitpick (grouped — none block anything)

- **Docker:** `out/` isn't fully excluded in `.dockerignore` (only `*.gif`/`*.png`), so a stray `policy.zip` could leak into a build context; no `USER` directive (runs as root — low-stakes for a test image); base image pinned by tag but not digest.
- **UKF/EKF:** the process-noise Jacobian block (`ukf.py:104-113`) is copy-pasted verbatim from `ekf.py:78-87` — a natural shared-helper candidate. `alpha=1e-3`/`kappa=0.0` are unexplained (the project otherwise documents magic-number rationale everywhere).
- **ROS2 bridge:** yaw→quaternion math duplicated in two functions; no NaN/Inf/range sanity-checking before "publishing" values (forward-looking design note — bridge isn't live yet); `radiation_type: 0`/`min_range: 0.0` are unexplained magic literals.
- **RL:** reward-shaping constants (`-0.01`, `-50.0`, `+100.0`) have no derivation, unlike this codebase's usual sweep-tuned-constant culture; `SENSOR_MAX_RANGE = 8.0` duplicated independently in `harness.py`; one test (`test_success_uses_the_same_position_only_tolerance...`) passes for a different reason than its own docstring claims (a stale `_prev_dist` from a manual state teleport inflates the reward it's checking).
- **Sensor latency:** `_release_due_messages` does two full-list scans per tick instead of exploiting sorted insertion order (a `deque` would be more efficient); pathological `latency_ticks` values could grow the pending-queue unboundedly (no current caller exercises this).
- **Core nodes:** `RadarNode` silently publishes nothing when not-ready, while three sibling nodes publish an explicit default — undocumented inconsistency, currently harmless given tick ordering. `IntersectionControllerNode._on_other_vehicle_status` replaces rather than merges `self._others` — fine with today's single publisher, a footgun if a second is ever added. `OtherVehicleStatus` travels as a raw list rather than a wrapped dataclass, inconsistent with the rest of `messages.py`. `EstimatorNode` treats `landmark_id` as a direct list index into a mutable list — fragile implicit contract, not currently triggered.

---

## What's genuinely good (worth saying plainly in a senior review)

- The ground-truth/estimate visibility boundary (`true_state` etc.) is verified by grep to be enforced everywhere it's claimed to be — no controller/estimator/planner node can see it.
- `KNOWN_BUGS.md`'s claims were spot-checked line-by-line against the actual code in the sensor-latency and Docker slices — zero docs/code drift found. That document is trustworthy, not aspirational.
- Dropout/latency modeling in `sensor_node.py` does exactly what it claims: RNG draw genuinely skipped at `dropout_prob==0.0`, delay is exactly N ticks (hand-verified, no off-by-one), delayed messages carry the value as originally computed, not a stale recompute.
- The UKF's actual math (sigma points, weights, circular-mean handling, angle-wrap) has no defects after two independent passes — its only real problems are missing input guards, not incorrect computation.
- `scenario_loader.py` correctly uses `yaml.safe_load`, not the unsafe loader.
- Full test suite: 286/286 passing, matching documented count.

---

## Suggested fix order

1. Docker default target (#1) — one-line-ish, high blast radius, first thing anyone hits.
2. RL seed handling (#2) — affects the credibility of a published validation claim.
3. UKF Cholesky/kappa guard (#6) + 4-state guard (#7) — small, prevents confusing crashes.
4. ROS2 obstacle-range converter/`register()` mismatch (#3) — before this bridge is ever wired live.
5. Everything else in Medium, roughly in the order listed — none are urgent, all are cheap.
