# Known bugs & unresolved limitations

This tracks defects and gaps that are **known but not yet fixed** — the reverse of
DESIGN.md/IMPLEMENTATION.md's "Known issues"/"Found while building X" sections, which document bugs
that were found *and* resolved. Everything below is either a reproducible, currently-collision-
causing behavior, or a documented scope limitation that would surface as a real problem outside the
conditions it's currently exercised under. Each entry links to where it's already discussed in more
depth and what would need to happen to close it.

## Reproducible safety bugs

### 1. Pure Pursuit reliably collides on `parallel_between_cars` under Hybrid A*

**Where**: `core/control/pure_pursuit.py`'s `PurePursuitAdaptive`, exposed via
`core/planning/hybrid_astar.py`'s obstacle-aware planner.
**Reproduce**: `python -m core.demo parallel_between_cars --controller pure_pursuit` (the default
planner is Hybrid A*), or `tests/test_simulation.py::test_parallel_between_cars_pure_pursuit_is_the_documented_unsafe_case`.
**What happens**: Pure Pursuit's already-documented "no margin once curvature is at the vehicle's
limit" weakness (DESIGN.md section 7) becomes an actual, consistent collision — 5/5 seeds, every
time — once Hybrid A* plans a genuinely curvature-saturated, obstacle-hugging avoidance route (this
scenario needs a reverse-gear cusp between two parked cars). Confirmed via a real parameter sweep
that it isn't a tuning problem: widening `brake_distance` or the planner's `safety_margin` just
trades the collision for Pure Pursuit never converging at all.
**Current mitigation, not a fix**: pinned as a documented, tested exception
(`tests/test_simulation.py`'s `UNSAFE_COMBINATIONS`) rather than silently allowed — the harness and
test suite both know this combination is unsafe, but nothing prevents a caller from actually running
it. `demo.py --controller pure_pursuit` on this scenario will still crash the car.
**What would actually fix it**: either give Pure Pursuit a real safety net for this regime (e.g. a
lookahead distance that adapts to local path curvature instead of a fixed constant), or accept it as
a permanent, documented controller limitation and steer users toward MPC for tight maneuvers. Not
attempted yet either way.
**Full account**: DESIGN.md section 6's M2 entry, "second real finding" paragraph.

### 2. MPC-ACC's gap constraint is still violated in standstill-recovery, ~0.5m below target

**Where**: `core/control/acc.py`'s `MpcAccController`.
**Reproduce**: `python -m core.validation.acc_validation --controller mpc` — the NGSIM excerpt
includes a real traffic stop.
**What happens**: `MpcAccController`'s `gap(t) >= min_gap` is a hard constraint enforced by the
optimizer (`scipy.optimize.minimize`'s `constraints` argument) — but if the ego ever ends up closer
than `min_gap` while both vehicles are stopped, there is *no feasible acceleration sequence* that
satisfies it (moving apart from a standstill would require reversing, which the ego can't do). SLSQP
silently returns its best constraint-violating attempt instead of failing loudly. Measured: realized
minimum gap lands ~0.5m below whatever `min_gap` is configured to, consistently, across a range of
values tried.
**Current mitigation, not a fix**: `min_gap` defaults to 3.0m (not the more natural 2.0m)
specifically to keep the *realized* worst case comfortably positive, chosen from measured erosion —
but this only shrinks the practical impact, it doesn't make the underlying infeasibility go away.
**What would actually fix it**: robust/stochastic MPC — tighten the gap constraint by a margin
proportional to prediction uncertainty instead of a fixed empirically-chosen `min_gap`, so the
safety margin adapts to how much the lead vehicle's actual behavior deviates from the constant-
velocity assumption the optimizer rolls out. Not built.
**Full account**: DESIGN.md section 11, "A real finding from validating against NGSIM."

## Scope limitations that would surface as real problems outside current usage

### 3. No re-planning when a sensed obstacle isn't on the current path (parking, M4)

**Where**: `core/nodes/controller_node.py` (braking) / `core/nodes/planner_node.py` (plans once,
never again).
**What happens**: `ControllerNode` already checks all 5 ultrasonic beams and brakes on any close
reading, but `PlannerNode` never re-plans around what triggered the brake. In every scenario this
project currently ships, the environment is static and fully known to the planner up front, so this
never actually bites — but a scenario with a genuinely unplanned/dynamic obstacle would make the
vehicle brake and stay stopped indefinitely rather than routing around it.
**What would close it**: wire `PlannerNode` to re-plan (not just `ControllerNode` to brake) when
`SensorNode` reports an obstacle the current path didn't account for. Explicitly called out as open
in IMPLEMENTATION.md's M4 entry; not started.

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
