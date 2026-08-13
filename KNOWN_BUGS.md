# Known bugs & unresolved limitations

This tracks defects and gaps that are **known but not yet fixed** — the reverse of
DESIGN.md/IMPLEMENTATION.md's "Known issues"/"Found while building X" sections, which document bugs
that were found *and* resolved. Everything below is either a reproducible, currently-collision-
causing behavior, or a documented scope limitation that would surface as a real problem outside the
conditions it's currently exercised under. Each entry links to where it's already discussed in more
depth and what would need to happen to close it.

There are no known reproducible safety bugs under normal usage — the two that used to be tracked
here (Pure Pursuit colliding on `parallel_between_cars`, and MPC-ACC's gap constraint going
infeasible in standstill-recovery) are both fixed; see entry 2 below and `core/control/acc.py`'s
`MpcAccController._effective_min_gap` docstring, respectively, for what changed and what residual,
non-bug behavior remains. Entry 7 is a real, reproducible collision, but only under an opt-in
testing feature (`sensor_latency_ticks`) pushed well beyond its verified-safe range — off by
default, and every existing scenario/test runs with it off.

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

### 7. Parking mode has no notion of sensor dropout/latency — partially closed

**Where**: `core/nodes/sensor_node.py` (dropout/latency modeling itself), `core/nodes/controller_node.py`
(`latency_margin`), `core/harness.py` (derives it).
**Status**: the capability itself is built (DESIGN.md section 10's future-extensions list) --
`SensorNode` can independently drop each tick's messages (`dropout_prob`, never arrives) or delay
them (`latency_ticks`, arrives late with the value it actually had when computed, not a stale
recompute). Both default to off, so every pre-existing caller is byte-for-byte unaffected --
verified directly, not just assumed: the RNG draw that gates dropout is skipped entirely (not just
guaranteed to never fire) when `dropout_prob == 0.0`, so the noise-sample sequence every other test
depends on is untouched.
**A real bug found and fixed while validating it**: a sweep across all 5 scenarios, both
controllers, 5 seeds each found dropout genuinely safe up to at least 0.4 (0 collisions), but
latency caused real collisions starting at just 5 ticks (0.5s) -- `ControllerNode`'s reactive speed
governor was trusting a delayed `obstacle_ranges` reading as if it reflected the *current* gap, when
it could be reporting a gap from `latency_ticks` ago, before the vehicle closed more distance toward
it. Fixed with `latency_margin` -- an extra, worst-case-derived subtraction from the governor's gap
calculation (`sensor_latency_ticks * dt * v_max`, the most distance closeable during the delay at
the vehicle's own top speed), following the exact same "physics-derived margin" pattern entry 2's
`stopping_buffer` already established for the one-tick sense-decide-act latency. Verified: 0
collisions at latency_ticks<=10 across the full sweep after the fix (previously 4/25 at
latency_ticks=5 alone), and a direct regression test confirms the specific fix is load-bearing (the
same case collides with `latency_margin` forced back to 0).
**What's still open**: latency beyond ~10-20 ticks (1-2s) can still cause real collisions, through a
*different* mechanism the margin fix can't reach -- delayed EKF corrections (compass/position_fix/
landmark_bearings are subject to the same `latency_ticks`) let dead-reckoning drift accumulate for
longer between fixes, and a reactive controller (Pure Pursuit most of all, consistent with its
already-documented curvature-saturation fragility -- entry 2) can steer the *true* vehicle into an
obstacle the *estimated* vehicle would have cleared. Confirmed directly, not just inferred: one such
run's true/estimated position error reached 2.7-4.0m in the ticks immediately before collision.
**What would close it**: proper out-of-sequence-measurement handling in the EKF (fuse a delayed
correction against the state *as it was* when the measurement was actually taken, not the current
state, then re-propagate forward -- a well-established estimation technique, but a materially bigger
feature than a stopping-buffer margin). `tests/test_sensor_robustness.py` pins the currently-verified-
safe range (dropout_prob<=0.2, latency_ticks<=10) as a real regression test, not just a claim.

### 8. Dockerization build-verified; one regression test was platform-float-sensitive -- closed

**Where**: `Dockerfile`, `docker-compose.yml`; `tests/test_sensor_robustness.py`'s
`test_latency_margin_is_what_actually_closes_the_gap`; solver in `core/control/mpc.py`'s
`MPCController.control` (`scipy.optimize.minimize(method="SLSQP")`).
**Status**: closed. The Docker image itself is genuinely build-verified now, closing the caveat in
README.md's Quickstart ("Docker itself wasn't available in the environment these files were written
in ... haven't been build-verified"): `docker build --target base .` succeeds cleanly on Python
3.12-slim, `docker run --rm auto-park` (the image's default `pytest -q`) runs the full suite to
completion in a Linux container, and `docker compose run --rm demo` builds, runs `core.demo`, and
correctly delivers the output GIF to the host through the `./out` volume mount. Verified 2026-08-13.
**What was actually wrong, and where**: two distinct things, found in sequence.
1. **A real platform-sensitivity finding.** The regression test failed in the Linux container but
   passed on a Windows host, with byte-identical numpy (2.5.2) and scipy (1.18.0) on both sides --
   ruling out a dependency-pinning mismatch. `scipy.show_config()` on both showed the identical
   scipy-openblas 0.3.31.dev build, `DYNAMIC_ARCH`, `Haswell` baseline, but a different compiler
   (gcc 15.2/Windows vs. 14.2.1/Linux) and `MAX_THREADS` (24 vs. 64); OpenBLAS's `DYNAMIC_ARCH`
   selects its actual compute kernel from the CPU it detects at runtime, which differs by
   host/container. Pinning `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` was tried and made no
   difference -- ruling out thread-order nondeterminism specifically, and pointing at the kernel
   selection itself. `MPCController`'s SLSQP solve (gradient-based, iterative) is float-order-
   sensitive across BLAS backends, and the original test ran on `parallel_between_cars` (already
   entry 2's tightest-margin scenario) with the safety margin forced to exactly zero and
   `sensor_latency_ticks=5`. Direct measurement (a signed vehicle-obstacle clearance metric, not
   just the boolean) found the true effect size at that exact configuration was only **2-7mm**
   across seeds 1-5 on the host -- smaller than the ~5mm swing the platform difference alone
   produced (-2.4mm on host vs. +2.3mm in-container for the identical seed=1 case), i.e. this
   specific config really was a coin flip, not a test-design boundary-rounding artifact.
   `sensor_latency_ticks=10` -- still inside this scenario's own already-documented verified-safe
   upper bound (entry 7) -- produces a consistent ~6cm penetration without the fix and >15cm
   clearance with it, confirmed matching between host and container to within ~3mm. The test now
   uses `latency_ticks=10` and asserts on that continuous clearance metric (with headroom on both
   sides of zero) rather than the boolean `result.collision`.
2. **A real bug in the rewritten test itself, caught by the fix above.** The first version of the
   rewrite loaded one `Scenario` and reused its `scenario.vehicle` object across both the
   "no-fix"/"with-fix" harness runs in the same test. `VehicleNode` stores the `Vehicle` it's given
   without copying and mutates it in place every tick (`core/nodes/vehicle_node.py`'s
   `self.vehicle.update(...)`), so the second run silently started from wherever the first run's
   vehicle physically ended up (already mid-collision) rather than the scenario's real start pose --
   both runs then measured the same deeply-negative clearance, masking the fix entirely. Caught
   immediately by the new continuous assertion (the "with-fix" run's clearance failed instead of
   trivially passing); fixed by reloading `load_scenario(...)` fresh inside each closure call.
**What would close it further**: nothing outstanding -- the underlying app behavior (the latency
margin fix) was never actually broken; only this one test's construction was fragile, on two
independent axes, both now fixed.

## Testing coverage gaps (not bugs, but relevant context)

Closed. `test_sensors.py` (hand-computed ray/circle intersection cases: dead ahead, out of range,
behind the beam, off to the side, tangent, nearest-of-several, beam-angle composition with vehicle
heading) and `test_control.py` (Pure Pursuit/MPC convergence from directly on a straight path and
from a lateral offset, decoupled from any planner or the estimation stack) both now exist.
`test_simulation.py`'s integration-level coverage remains, unchanged, as the end-to-end check on
top of them.
