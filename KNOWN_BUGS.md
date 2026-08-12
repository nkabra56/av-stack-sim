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

### 4. H4's intersection model has no notion of turning movements

**Where**: `core/control/intersection_geometry.py` / `core/intersection2d_harness.py`.
**Status**: the original gap — no real crossing paths, no way to check more than two approaches, no
geometric verification that right-of-way reasoning actually prevents a collision — is closed.
`intersection_geometry.py` gives the intersection a real 2D layout (perpendicular roads, right-hand-
traffic lane offsets, a genuine conflict-zone box, `is_to_the_right` derived from actual travel
headings instead of a hand-set flag); `intersection2d_harness.py` runs any number of real vehicles
— each still just an unmodified `IntersectionNavigator`, the same reuse principle H4 always used for
`IDMController` — through it, deriving every vehicle's view of the others from real simulated state
instead of a script, and checking actual circle-to-circle proximity, not just arrival-order
bookkeeping. Verified directly (`tests/test_intersection_geometry.py`): 2-, 3-, and 4-way scenarios
all resolve with zero real collisions and correct proceed order, parallel (non-crossing) approaches
are recognized as such, and — to confirm the check isn't vacuous — a deliberately non-compliant
navigator is caught colliding on a pairing the compliant version already proved safe. A real,
physical finding along the way: the geometric constants (conflict-zone size, lane offset, stop
margin) aren't independent of `VEHICLE_RADIUS` — an under-sized intersection can put a vehicle
waiting at its own stop line within collision range of the perpendicular through-lane, a spacing
requirement the old single-point model never had to reason about at all; the module docstring/
harness comments carry the derivation.
**What's still open**: turning movements (e.g. left-turn-yields-to-oncoming-through-traffic) —
every vehicle in this model goes straight through its own approach. Deliberately deferred rather
than attempted in the same pass (this project's own precedent: H1 longitudinal-only before H3, H3
uncombined before H5) since it needs real per-turn path geometry — a curved connector from one
approach's lane to another's — not just an additional straight approach.
**What would close it**: give turning vehicles a real curved path between approaches (reusing
Reeds-Shepp/Dubins-style curve generation from the parking side is a plausible starting point, since
the underlying geometry problem — connect two poses with a drivable curve — is the same one already
solved there) and extend the right-of-way check to "yield to oncoming through traffic while
turning left," the one real-world rule this model still can't express. Not started.

### 5. CCC (3-point-turn) family: closed — but the original premise was wrong

**Where**: `core/planning/reeds_shepp.py`, `core/planning/dubins.py`, `core/planning/hybrid_astar.py`.
**Status**: this used to be tracked as "`reeds_shepp_path`/`ReedsSheppPlanner` only implement CSC, so
`RuntimeError` when start/goal turning circles are closer than ~4× `turning_radius`." That premise
doesn't hold: a global optimization search over every `(alpha, beta, d)` combination, using the
actual `_lsl`/`_rsr`/`_lsr`/`_rsl` functions, found **no case** where all 4 CSC families are
simultaneously infeasible, even as `d -> 0` — confirmed directly with 20,000 random trials through
`ReedsSheppPlanner.plan()` targeting exactly that regime, producing zero `RuntimeError`s. CSC (using
all 4 families, not just 2) was apparently always sufficient for *feasibility* here; CCC's absence
was never actually a crash risk, and `dubins.py`'s identical "CCC only matters when circles are
close" scoping note carries the same mistaken premise (not fixed there — DubinsPlanner is a separate,
forward-only baseline outside this entry's original "Where," so its docstring is left as historical
context, not corrected).
**What was real, and is now fixed**: in that same close-pose regime, CCC is strictly *shorter* than
the best CSC candidate in ~30% of random trials, sometimes by close to 2× — a genuine path-quality
gap. `reeds_shepp.py` now implements CCC (LRL/RLR) via direct geometric construction (tangent-circle
centers, not a from-memory trig formula — verified by reconstructing thousands of random candidates
through the same arc-stepping machinery already used elsewhere and checking the endpoint, plus a
20,000-trial no-crash check and a length-vs-actual-path-length consistency check, all pinned in
`tests/test_planning.py`). A second, unrelated latent bug turned up while validating it:
`reeds_shepp_length` unconditionally included the Euclidean-distance lower bound in its `min()`,
which (since a curved path can never be shorter than the straight line between its endpoints) meant
it almost always just returned that lower bound outright, silently discarding the "shortest of real
candidates" its docstring claimed — fixed to only fall back to Euclidean distance when nothing else
is feasible.
**A real finding from wiring it in**: CCC is NOT enabled for `HybridAStarPlanner` (`include_ccc=False`
at all 4 of its call sites, and the Euclidean-fallback fix travels with that same flag). CCC's shorter
paths are more curvature-aggressive than CSC's, and since Hybrid A*'s analytic expansion is attempted
from every search node once it's near the goal — not just the one final connection — enabling CCC
there measurably reopened bug 1's Pure Pursuit collision on 3 scenarios, including 2
(`perpendicular_flanked`, `perpendicular_obstructed_lane`) that were previously perfectly safe.
`HybridAStarPlanner` already degrades gracefully without CCC (composes the same 3-point-turn shape
out of ordinary primitives when needed), so it doesn't need the family and isn't worth the risk.
Pinned by `tests/test_planning.py::test_hybrid_astar_does_not_use_ccc`.

### 6. H5's real leader and lane centerline are from different NGSIM lanes — closed

**Where**: `core/full_highway_harness.py`, `core/validation/ngsim_loader.py`, `core/data/ngsim/`.
**Status**: closed. The replayed lead vehicle used to be NGSIM `lane_id=1` (`vehicle_id` 9/12)
while the Stanley-tracked centerline was derived from **lane 2** — real data, genuinely overlapping
positions, but not the same lane. Re-extracted a lane-2-specific leader/follower pair (`vehicle_id`
2896 leading, 2903 following) from the same public Socrata source (`data.transportation.gov`,
dataset `8ect-6jqj`, no registration required), filtered to `location='us-101' AND lane_id='2'`
within a time window overlapping the original excerpt (real US-101 congestion is road-wide, not
lane-specific, so the same window reliably has comparable lane-2 traffic). The new pair has a
**100% pure** `preceding` link (783/783 frames, vs. the kind of majority-but-not-total purity most
other candidate pairs in the same window had) and a genuine recorded full stop, same character as
the original. `core/validation/ngsim_loader.py`'s `DEFAULT_LEADER_ID`/`DEFAULT_FOLLOWER_ID` now
point at it; full extraction account in `core/data/ngsim/ATTRIBUTION.md`.
**A real finding from re-validating against it**: this leader's genuine full stop measurably (if
temporarily) stresses the composed EKF/Stanley loop during the low-speed restart afterward —
Stanley's `atan2(k*cte, speed)` correction is deliberately weakest exactly when speed is lowest, so
a transient lateral-tracking degradation while pulling away from a dead stop is expected behavior.
Confirmed genuinely transient (not a failure to converge) by checking the full run, not just the
region right after it, across all 6 (controller, seed) pairs: peak cross-track error during the
~15s recovery window reached 0.62m on one otherwise-unremarkable run (seed 2/idm), but every pair
settled back under the 0.46m real-driver-scatter bar well before the run's midpoint regardless.
`tests/test_full_highway.py`'s convergence tests now check the settling window from t=50s rather
than t=30s to fairly account for this (documented in the test itself, not silently widened) — a
finding, not a hidden regression: the RMS/min-gap figures reported in DESIGN.md's H5 entry shifted
somewhat with the new (genuinely different) real leader, both noted there.

## Testing coverage gaps (not bugs, but relevant context)

`test_sensors.py` (hand-computed ray/circle intersection cases) and `test_control.py` (controller
convergence from a straight-line path) are both still on the "planned, not built" list in
IMPLEMENTATION.md section 4 — current coverage for both areas is integration-level only, via
`test_simulation.py`. Nothing is known to be broken here; it just hasn't been unit-tested in
isolation, so a regression in either area would currently only be caught indirectly.
