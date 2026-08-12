# Known bugs & unresolved limitations

This tracks defects and gaps that are **known but not yet fixed** — the reverse of
DESIGN.md/IMPLEMENTATION.md's "Known issues"/"Found while building X" sections, which document bugs
that were found *and* resolved. Everything below is either a reproducible, currently-collision-
causing behavior, or a documented scope limitation that would surface as a real problem outside the
conditions it's currently exercised under. Each entry links to where it's already discussed in more
depth and what would need to happen to close it.

There are currently no known reproducible safety bugs — the two that used to be tracked here
(Pure Pursuit colliding on `parallel_between_cars`, and MPC-ACC's gap constraint going infeasible in
standstill-recovery) are both fixed; see entry 2 below and `core/control/acc.py`'s
`MpcAccController._effective_min_gap` docstring, respectively, for what changed and what residual,
non-bug behavior remains.

## Scope limitations that would surface as real problems outside current usage

### 2. Pure Pursuit still can't complete `parallel_between_cars` under Hybrid A* (collision fixed; now fails safe)

**Where**: `core/control/pure_pursuit.py`'s `PurePursuitAdaptive`, exposed via
`core/planning/hybrid_astar.py`'s obstacle-aware planner; safety net in
`core/nodes/controller_node.py`.
**Status**: the reproducible **collision** this used to cause (5/5 seeds, every time) is fixed — see
`ControllerNode`'s docstring and `tests/test_simulation.py::test_never_collides`, which now covers
this combination unconditionally. What's left is a genuine, permanent scope limitation, not a safety
bug: this combination still never *succeeds* (`tests/test_simulation.py`'s `NEVER_SUCCEEDS`,
pinned by `test_parallel_between_cars_pure_pursuit_fails_safe_not_success`).
**Root cause found while fixing the collision**: the old brake mechanism
(`hybrid_astar.brake_distance_for`) computed a fixed trigger distance from the planner's own
clearance floor (`vehicle_radius + safety_margin - buffer`) — with its default `buffer` equal to
`safety_margin`, this collapsed to *exactly* `vehicle_radius`, i.e. the literal collision boundary,
so the brake fired at the moment of contact rather than before it, for any speed. It wasn't
sufficient on its own to fix, either: a real physics-based stopping-distance formula
(`v_allowed = sqrt(2*a_max*gap)`, now `ControllerNode._safe_speed`) still requires enough stopping
buffer that it can't fit inside Hybrid A*'s ~0.15m intentional-clearance band at anything close to
`v_max` without also throttling speed continuously as the vehicle approaches an obstacle — which is
what the fix now does.
**What happens now**: Pure Pursuit's already-documented "no margin once curvature is at the
vehicle's limit" weakness (DESIGN.md section 7) still means it can't track this scenario's
curvature-saturated, obstacle-hugging reverse-gear cusp — but instead of the tracking error running
away into a collision, the speed governor throttles the vehicle to a safe stop before contact. A
real parameter sweep (lookahead, v_max, and an adaptive-lookahead variant tried while investigating
this) confirmed it isn't a Pure-Pursuit-tuning problem: nothing in that space lets it both stay clear
of the obstacle and keep converging. MPC's constraint-respecting rollout never needed the governor
here — it stays collision-free and converges reliably on its own (5/5, up to ~880 steps).
**What would actually close it**: either give Pure Pursuit a fundamentally different (non-reactive)
fallback for curvature-saturated regimes, or accept this as a permanent, documented controller
limitation and steer users toward MPC for tight maneuvers — same choice KNOWN_BUGS previously
described as open, now with the collision risk removed either way.
**Full account**: DESIGN.md section 6's M2 entry, "second real finding" paragraph;
`core/nodes/controller_node.py`'s module docstring for the speed-governor fix.

### 3. Re-planning exists now, but a re-plan can still land on a route the speed governor won't drive

**Where**: `core/nodes/controller_node.py` (stall detection -> `replan_request`) /
`core/nodes/planner_node.py` (re-plans against the live obstacle list on that signal).
**Status**: the original bug — `PlannerNode` plans once and never again, so `ControllerNode`
braking on an unplanned obstacle just left the vehicle stopped indefinitely — is fixed. If
`ControllerNode`'s speed governor (KNOWN_BUGS.md entry 2's fix) stays binding for `STALL_TICKS`
consecutive ticks, it publishes `replan_request`; `PlannerNode` re-plans from the latest pose
estimate against `environment.obstacles` *read live*, so an obstacle added to that list mid-run
(invisible to the original plan) is fully visible to the re-plan. Verified directly
(`tests/test_replanning.py`): a re-plan triggered this way produces a genuinely different path that
clears a newly-appeared obstacle by a real margin, capped at `max_replans` attempts, and a planner
that raises (no route exists) is handled without crashing the simulation.
**What's still open**: in one closed-loop configuration built while testing this
(`test_replanning_produces_a_materially_different_obstacle_avoiding_path`), the vehicle stalls,
triggers a correct re-plan, gets a valid detour back — and immediately stalls again at the *start*
of that detour, because Hybrid A* only guarantees its own `safety_margin` (0.15m) of clearance,
tighter than `ControllerNode`'s `stopping_buffer` (0.5m, tuned for entry 2's slower-approach
scenario). The governor doesn't know the tighter clearance belongs to a deliberately-computed route
rather than raw unplanned proximity, so it throttles the same way either time. Not a safety bug (the
vehicle still never collides — see `test_never_collides_with_a_dynamically_appearing_obstacle`) and
not the re-planning gap this entry originally tracked, but a real, separate wrinkle: a fast,
sudden-appearance obstacle can leave no `stopping_buffer` value that both (a) allows enough room to
find a detour from the stall point and (b) doesn't also throttle progress along that detour once
found.
**What would close it**: let the governor distinguish "the planner deliberately routed this close"
from "raw unplanned proximity" -- e.g. relax `stopping_buffer` toward the planner's own
`safety_margin` while tracking a path that's already known to respect it, the same distinction
`hybrid_astar.brake_distance_for` used to try to make statically (see entry 1's history) before it
turned out to need to be dynamic, not fixed. Not attempted yet.

### 4. H4's intersection model is a single conflict point, not real 2D geometry

**Where**: `core/control/intersection.py`'s `IntersectionNavigator`.
**What happens**: models right-of-way as mutual exclusion + arrival-order priority at one point two
approaches share — correct for the stop-sign scenarios it's validated against, but has no notion of
actual crossing paths, turning movements, or more than two approaches at an intersection. A scenario
needing real lane-level intersection geometry isn't representable with the current model at all.
**What would close it**: a real 4-way intersection simulation with lane-level geometry per approach
— a substantial new milestone, not a small fix. Noted as future work in DESIGN.md section 12, not
started.

### 5. Reeds-Shepp/Hybrid A* have no CCC (3-point-turn) family — raises rather than degrades for very close poses

**Where**: `core/planning/reeds_shepp.py`.
**What happens**: `reeds_shepp_path`/`ReedsSheppPlanner` only implement the CSC family (8
candidates: 4 Dubins families × forward/backward). When start and goal turning circles are closer
together than ~4× `turning_radius`, no CSC candidate exists, `reeds_shepp_path` returns `None`, and
`ReedsSheppPlanner.plan()` raises `RuntimeError`. `HybridAStarPlanner` degrades gracefully in this
regime (its primitive-by-primitive search can still compose the same maneuver out of ordinary
forward/reverse steps), but the *standalone* Reeds-Shepp planner cannot. Verified this doesn't
currently trigger for any of the 5 shipped scenarios, so it's not live today — but it's a real crash
waiting for whichever future scenario needs a start/goal pair that close together and reaches for
`ReedsSheppPlanner` directly (e.g. via `demo.py --planner reeds_shepp`).
**What would close it**: implement the CCC (LRL/RLR) family — the classic "3-point-turn" curves,
deliberately scoped out of M2. Noted in DESIGN.md section 6's alternatives-considered list as
"genuine future work if a scenario is ever added where ... Hybrid A*'s primitive-composed fallback
isn't good enough."

### 6. H5's real leader and lane centerline are from different NGSIM lanes

**Where**: `core/full_highway_harness.py`.
**What happens**: the replayed lead vehicle (`core/data/ngsim/excerpt_trajectories.csv`) is recorded
in NGSIM `lane_id=1`; the Stanley-tracked centerline (`core/data/ngsim/lane_centerline.csv`) is
derived from NGSIM **lane 2**. Both are real data from the same road/location and share NGSIM's
along-road coordinate convention (verified the position ranges genuinely overlap), so the scenario
is coherent enough to exercise ACC+Stanley composition — but it is not lane-precise, and this was a
deliberate scope call, not a resolved question. Documented in the module's own docstring, not
hidden, but still open.
**What would close it**: re-extract a lane-2-specific leader/follower pair from the same public
Socrata source (no registration required) so the whole scenario is single-lane-coherent. Flagged as
a natural follow-up when this was built; not done.

## Testing coverage gaps (not bugs, but relevant context)

`test_sensors.py` (hand-computed ray/circle intersection cases) and `test_control.py` (controller
convergence from a straight-line path) are both still on the "planned, not built" list in
IMPLEMENTATION.md section 4 — current coverage for both areas is integration-level only, via
`test_simulation.py`. Nothing is known to be broken here; it just hasn't been unit-tested in
isolation, so a regression in either area would currently only be caught indirectly.
