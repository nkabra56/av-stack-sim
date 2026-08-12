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

### 3. Re-planning exists now, and a re-plan no longer gets stuck at the start of its own detour

**Where**: `core/nodes/controller_node.py` (stall detection -> `replan_request`, tracking-aware
buffer) / `core/nodes/planner_node.py` (re-plans against the live obstacle list) / `core/harness.py`
(derives the tracked buffer from the active planner).
**Status**: closed. The original bug — `PlannerNode` plans once and never again, so `ControllerNode`
braking on an unplanned obstacle just left the vehicle stopped indefinitely — was fixed first (see
above); a real residual then showed up while testing that fix: in
`test_replanning_produces_a_materially_different_obstacle_avoiding_path`, the vehicle would stall,
trigger a correct re-plan, get a valid detour back — and immediately stall again at the *start* of
that detour, because Hybrid A* only guarantees its own `safety_margin` (0.15m) of clearance,
tighter than `ControllerNode`'s `stopping_buffer` (0.5m, tuned for entry 2's slower-approach
scenario). The governor had no way to tell "the planner deliberately routed this close" apart from
"raw unplanned proximity," so it throttled both the same way.
**What closed it**: `ControllerNode` now computes live cross-track distance to the current path
each tick and uses a much smaller `tracked_stopping_buffer` whenever that distance is below
`tracking_threshold` -- i.e. the vehicle is actually tracking the plan, not drifting off it. Above
the threshold, it falls back to the original, fully conservative `stopping_buffer`, so entry 2's
exact failure mode (Pure Pursuit's cross-track error growing past ~0.66m under curvature
saturation) still gets the full margin it needs. `ParkingHarness` derives `tracked_stopping_buffer`
from `getattr(planner, "safety_margin", None)` -- planners with no exposed clearance guarantee
(Dubins/ReedsShepp, both obstacle-blind) automatically get `None` and keep the fully conservative
buffer unconditionally, since they never earned a smaller one.
**Real finding from a real parameter sweep** (not picked by eye): the naive-sounding threshold
(0.3m, "well under entry 2's ~0.66m collision-point error") *reopened* entry 2's collision outright
-- by the time cross-track error reaches even 0.1-0.15m, the vehicle is already well into the
dangerous divergence, not still safely on-plan. The threshold had to be tight enough that the
tracking/not-tracking classification is essentially never wrong, not just usually right. Swept
`tracking_threshold` x `tracked_buffer_extra` (the margin added on top of the planner's raw
`safety_margin`) jointly against both entry 2's regression scenario and this entry's own detour
scenario, across 5 seeds each: `tracking_threshold=0.03` (3cm) paired with `tracked_buffer_extra=0.3`
was the smallest/safest combination found -- 0/5 collisions on both scenarios, 5/5 full recoveries
(the vehicle now actually reaches the goal, not just gets an unusable detour) on this entry's
scenario. Pinned by `tests/test_replanning.py::test_replanning_recovery_holds_across_seeds` and
direct unit coverage of the buffer-selection logic itself in `tests/test_controller_node.py`.
**`tracked_buffer_extra` revised to 0.4, from a later code-review finding**: `_effective_buffer`'s
cross-track measurement used to be nearest-*waypoint* distance, which over-estimates true
cross-track error by up to half Hybrid A*'s waypoint spacing (~0.05m against a 3cm threshold --
not a large margin). Fixing it to true perpendicular-distance-to-path-segment (strictly more
accurate, and always <= the old measurement) classifies more ticks as "tracking," using the
smaller buffer more often -- which shifted this constant's own tuned operating point enough that
a wider 25-seed re-sweep (the original tuning only checked 5) found `tracked_buffer_extra=0.3`
now producing 1/10 real collisions once the seed range widened, invisible at 5 seeds; 0.2 fails
outright (4/5 collisions), confirming this constant sits right at a real safety cliff rather than
a comfortable margin above one. `0.4` held 0/25 collisions with 23/25 full completions across
seeds 1-25 (the 2 non-completions are the max_replans residual described above, not unsafe) and
is the current default.

### 4. H4's intersection model has no notion of turning movements — closed

**Where**: `core/control/intersection_geometry.py` / `core/intersection2d_harness.py`.
**Status**: closed, including turning movements. The original gap — no real crossing paths, no way
to check more than two approaches, no geometric verification that right-of-way reasoning actually
prevents a collision — was closed first (real 2D layout, `is_to_the_right` derived from actual
travel headings, real circle-to-circle proximity checking instead of trusting arrival-order
bookkeeping alone; a real finding along the way: the conflict-zone/lane-offset/stop-margin constants
aren't independent of `VEHICLE_RADIUS`, since an under-sized intersection can put a vehicle waiting
at its own stop line within collision range of the perpendicular through-lane). Turning movements
were then added on top: a vehicle with `turn != "straight"` drives its entry approach's straight
lane, a real curved connector reusing `DubinsPlanner` (`build_turn_path`), then the exit approach's
straight lane, and left turns yield to oncoming (opposite-approach) straight-through traffic that
hasn't cleared — the one right-of-way rule `IntersectionNavigator`'s pure arrival-order model can't
express on its own, added at the harness level as an extra phantom `OtherVehicleStatus` rather than
by modifying `IntersectionNavigator` itself.
**Two real bugs found and fixed while adding turning movements**, both via a random mixed-turn sweep
(varying start distances and turn assignments across all four approaches, hundreds of trials) rather
than the hand-picked two-vehicle scenarios that had looked correct in isolation:
1. **Geometric inconsistency between the curve's start and the stop line.** `build_turn_path` starts
   the curved connector `turn_lead = TURN_LEAD_RATIO * turning_radius` before the conflict zone (a
   real finding on its own: an under-sized `turn_lead` made `DubinsPlanner`'s CSC solve find a long,
   looping connection instead of the short, direct one — the same "CCC would be shorter" regime
   entry 5 characterized, but `DubinsPlanner` has no CCC fallback; `TURN_LEAD_RATIO = 2.5` was swept
   as the smallest value with margin that stayed short and direct for both turn directions across
   `turning_radius` 4-8m). But `IntersectionNavigator`'s own 1D stop line sits at
   `conflict_half_width + stop_margin` before the zone — a value chosen independently, for the
   straight-through-only model, with no awareness of `turn_lead` at all. When `turn_lead` exceeded
   that, a turning vehicle's stop line fell *inside* the curve's span, so a genuinely stopped vehicle
   was already geometrically partway around the corner instead of cleanly on its straight entry lane
   — confirmed directly (a stopped vehicle's y-coordinate was neither its entry lane's nor its exit
   lane's), and it produced a real collision in the sweep. Fixed by making the two mutually
   consistent (`VehicleSpec.turning_radius` default 4.0, `conflict_half_width` default 9.5, chosen so
   `TURN_LEAD_RATIO * turning_radius <= conflict_half_width + stop_margin` with margin), plus a
   runtime guard in `run_multi_approach_scenario` that raises immediately if a caller supplies a
   combination that violates the inequality, so this can't silently regress.
2. **A circular deadlock between the new yield rule and ordinary arrival-order yielding.** The first
   version of the oncoming-traffic rule was unconditional but still let a straight vehicle's own real
   arrival-order status count against it from the opposing left-turner's perspective — so if the
   left-turner happened to arrive first, the straight vehicle would yield to *it* via normal
   arrival-order, while the left-turner simultaneously yielded to the straight vehicle via the new
   rule: a live two-vehicle cycle where neither ever proceeds. Fixed by making the relation strictly
   one-directional — a straight vehicle never perceives an opposing left-turner's real arrival status
   at all (so it can never be made to yield to one), while the left-turner's own yield is unconditional
   as before. A second, narrower version of this bug turned up when the exemption was (wrongly, at
   first) extended to right-turners too: since the phantom rule only ever fires for a *left*-turner
   yielding to a *straight* vehicle, exempting right-turners left them with no mutual-exclusion
   mechanism against an opposing left-turner at all, and the sweep caught real collisions on that
   specific pairing — reverted to only exempting straight vehicles. A third fix was needed even after
   the deadlock was gone: the phantom rule originally only engaged once the oncoming vehicle had left
   `APPROACHING` state (stopped or already proceeding) — but "still approaching" only means "hasn't
   reached its own stop line yet," not "far away and safe to ignore"; a vehicle at full cruise speed
   can still reach the conflict zone before a left-turner finishes crossing it. Two vehicles collided
   in the sweep while both were `PROCEEDING`, because the left-turner's own stop_time arrived and its
   state check passed before the oncoming vehicle had even stopped. Fixed by gating purely on live
   position (`d[b] < clear_distance[b]`) instead of navigator state — conservative (a left-turner now
   waits out an oncoming vehicle's entire approach, not just its time in the box) but that trade
   favors safety over throughput, consistent with the rest of this project.
**Verified**: a 400-trial random sweep (random start distances 60-140m, random turn assignment per
approach) after all three fixes: 0 collisions. `tests/test_intersection_geometry.py` adds direct
regression coverage — turn exit heading/position correctness, the left-yields-to-oncoming-despite-
earlier-arrival case and its clearing-early control case, confirmation right-turners don't get the
special rule, curvature-limit/path-length checks on all 8 (approach x direction) turn geometries, and
a 60-trial version of the sweep itself.
**Known, accepted residual — a liveness limitation, not a safety one**: the same 400-trial sweep
still hits the step budget (never resolves) in ~8% of trials. Confirmed genuinely permanent (still
stuck at 10x the step budget, not just slow) and confirmed safe (never a collision in any case
observed). Root cause: the left-turn yield rule is deliberately unconditional/arrival-order-
independent (that's the real right-of-way rule), and combined with ordinary arrival-order yielding
among the *other* vehicles present, this can form a genuine N-vehicle wait cycle (e.g. a 3-way cycle:
a left-turner waits on its unconditional opposite, which waits on a third vehicle that arrived
earlier, which itself waits on the left-turner for the same reason) — a purely local, pairwise
right-of-way model has no global cycle detection. This is the same class of limitation as the
pre-existing exact-simultaneous-arrival tie gridlock (mechanical yield-to-right with no tiebreaker),
just with a different trigger. Pinned, not hidden, by
`test_left_turn_yield_can_gridlock_but_never_collides`. Actually closing it would need a global
precedence graph (topologically order all vehicles' yield relations, detect and break cycles) rather
than the current per-pair local reasoning — a materially bigger architectural change than turning
movements themselves needed, and out of scope here.

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
there measurably reopened entry 2's Pure Pursuit collision on 3 scenarios, including 2
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
