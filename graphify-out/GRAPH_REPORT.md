# Graph Report - C:\Users\Nishant Kabra\Documents\personal\projects\auto-park-contoller  (2026-08-13)

## Corpus Check
- 112 files · ~82,383 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 997 nodes · 2561 edges · 66 communities (54 shown, 12 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 320 edges (avg confidence: 0.52)
- Token cost: 154,504 input · 0 output

## Community Hubs (Navigation)
- Intersection Navigation & Geometry
- Dubins Path Planning
- Extended Kalman Filter
- Data Provenance & Design Docs
- Graphify Skill: Watch & Spec
- ROS2 Bridge Messaging
- Vehicle Messages & Longitudinal Node
- Unscented Kalman Filter
- Sensors & Obstacle Environment
- MPC Control & Vehicle Model
- Hybrid A* Replanning
- Lane Geometry & Full Highway Harness
- Controller Node & Path Messages
- ACC Highway Harness
- Controller Node Tests
- Message Bus
- Estimator Node
- RL Parking Env
- Full Highway Tests
- Full Highway Harness Core
- Intersection Harness Tests
- RL Comparison Validation
- IDM ACC Controller
- Demo & Simulation Tests
- Core Interfaces & Harness
- Parking Harness & Planner Node
- ACC Validation
- MPC & Lane Centering Control
- True State & Sensor Node Tests
- ACC Controller Node
- NGSIM Loader Validation
- MPC-ACC Controller
- Stanley Lane Centering
- RL Training
- Sensor Robustness Tests
- Lane Centering Validation
- Environment & Animation Visualization
- Control Convergence Tests
- Vehicle Node
- Pure Pursuit Control
- Graphify Exports (Neo4j/FalkorDB/MCP)
- Controller/Planner Interfaces
- Graphify GitHub & Merge
- Graphify Transcription
- Open Parking Scenarios
- Graphify Honesty Rules
- Graphify Install Step
- Graphify Detect Step
- Graphify Extraction Pipeline Step
- Graphify Health Check Step
- Graphify HTML/Obsidian Step
- Actuator Limit Enforcement
- Bus Synchronous Dispatch
- Kinematic Bicycle Model
- True State Boundary
- Package Metadata

## God Nodes (most connected - your core abstractions)
1. `Bus` - 103 edges
2. `Vehicle` - 59 edges
3. `wrap_angle()` - 44 edges
4. `ExtendedKalmanFilter` - 39 edges
5. `Obstacle` - 38 edges
6. `ParkingHarness` - 37 edges
7. `Environment` - 33 edges
8. `PoseEstimateMsg` - 33 edges
9. `FullHighwayHarness` - 31 edges
10. `ObstacleRangeMsg` - 30 edges

## Surprising Connections (you probably didn't know these)
- `test_mpc_accel_stays_within_bounds()` --calls--> `MpcAccController`  [EXTRACTED]
  tests/test_acc.py → core/control/acc.py
- `_AlwaysWantsToMove` --uses--> `MPCController`  [INFERRED]
  tests/test_replanning.py → core/control/mpc.py
- `_AlwaysWantsToMove` --uses--> `PurePursuitAdaptive`  [INFERRED]
  tests/test_replanning.py → core/control/pure_pursuit.py
- `_AlwaysWantsToMove` --uses--> `Obstacle`  [INFERRED]
  tests/test_replanning.py → core/environment.py
- `_AlwaysWantsToMove` --uses--> `Spot`  [INFERRED]
  tests/test_replanning.py → core/environment.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Step 3 extraction pipeline (AST + semantic + merge)** — claude_skills_graphify_skill_parta_structural_extraction, claude_skills_graphify_skill_partb_semantic_extraction, claude_skills_graphify_skill_partc_merge_ast_semantic [EXTRACTED 1.00]
- **root= relativization kept consistent across build/manifest/merge (#1361)** — claude_skills_graphify_skill_step4_build_cluster_analyze, claude_skills_graphify_skill_step9_manifest_cost_cleanup, claude_skills_graphify_references_update_build_merge_function [INFERRED 0.85]
- **Semantic extraction cache/manifest attribution safety net (#1939/#2015/#1948)** — claude_skills_graphify_skill_stepb0_cache_check, claude_skills_graphify_skill_stepb3_collect_merge, claude_skills_graphify_references_update_manifest_stamping [INFERRED 0.75]
- **"Verify before you build on it" discipline across findings** — design_verify_before_build_discipline, design_hybrid_astar_reeds_shepp_m2, design_kitti_validation, design_h3_lane_centering_stanley, design_idm_controller, design_h5_ekf_bugs [EXTRACTED 1.00]
- **Classical/reactive vs. optimization-based controller comparison device** — design_pure_pursuit_vs_mpc_comparison, design_idm_controller, design_mpc_acc_controller, design_h3_lane_centering_stanley [EXTRACTED 1.00]
- **Real-world dataset validation suite (KITTI + NGSIM)** — design_kitti_validation, design_ngsim_acc_validation, design_h3_lane_centering_stanley, design_h5_full_highway_drive [EXTRACTED 1.00]

## Communities (66 total, 12 thin omitted)

### Community 0 - "Intersection Navigation & Geometry"
Cohesion: 0.06
Nodes (68): Approach, build_turn_path(), in_conflict_zone(), is_opposite(), is_to_the_right(), ndarray, Real 2D geometry for a 4-way intersection: straight approaches on two…, True if two approaches are directly facing each other (e.g. NORTH and SOUTH) --… (+60 more)

### Community 1 - "Dubins Path Planning"
Cohesion: 0.06
Nodes (66): CLI entry point: run a named scenario and show (or save) the animation. Usage:…, _arc_points(), _csc_points(), _lsl(), _lsr(), _mod2pi(), ndarray, Pose (+58 more)

### Community 2 - "Extended Kalman Filter"
Cohesion: 0.06
Nodes (57): ExtendedKalmanFilter, ndarray, Extended Kalman Filter for [x, y, theta] pose estimation, with an optional 4th…, 4-state only (H2): fuse a noisy speedometer reading into the v state., 4-state predict (H2): control input is acceleration, not speed -- v is now an…, _delta_from_yaw_rate(), main(), plot_validation() (+49 more)

### Community 3 - "Data Provenance & Design Docs"
Cohesion: 0.05
Nodes (51): KITTI Data Attribution, Geiger, Lenz & Urtasun 2012 - KITTI Vision Benchmark Suite (CVPR), KITTI Excerpt Poses (seq 09, frames 840-1139), NGSIM Data Attribution, Trained RL Policy Provenance, Scenario: Parallel between two parked cars, Scenario: Perpendicular flanked by parked cars, Scenario: Perpendicular obstructed lane (+43 more)

### Community 4 - "Graphify Skill: Watch & Spec"
Cohesion: 0.05
Nodes (43): Debounce (default 3s), /graphify add <url>, add-watch.md reference doc, Supported /graphify add URL types, --watch background folder watcher, calls edge direction + single-language rule, Discrete confidence_score rubric, extraction-spec.md reference doc (+35 more)

### Community 5 - "ROS2 Bridge Messaging"
Cohesion: 0.09
Nodes (33): ControlCmdMsg, control_cmd_to_ros_kwargs(), obstacle_range_to_ros_kwargs_list(), path_to_ros_kwargs(), pose_estimate_to_ros_kwargs(), Any, Protocol, Reference adapter mapping `messaging/bus.py`'s in-process pub/sub onto real… (+25 more)

### Community 6 - "Vehicle Messages & Longitudinal Node"
Cohesion: 0.13
Nodes (18): AccelOdometryMsg, CompassMsg, EgoHighwayStateMsg, LateralCmdMsg, LongitudinalCmdMsg, Typed messages passed over the Bus. See DESIGN.md section 2. `TrueStateMsg` is…, SpeedometerMsg, SteeringOdometryMsg (+10 more)

### Community 7 - "Unscented Kalman Filter"
Cohesion: 0.13
Nodes (18): _circular_mean(), ndarray, Unscented Kalman Filter for [x, y, theta] pose estimation -- an alternative to…, Shared correction step for all three measurement types below: propagate the…, Weighted mean of angles, correct across the -pi/pi wrap -- a naive weighted…, UnscentedKalmanFilter, parametrize, A state estimate near +pi, corrected by a compass reading just past -pi (the… (+10 more)

### Community 8 - "Sensors & Obstacle Environment"
Cohesion: 0.13
Nodes (20): Obstacle, A circular obstacle (bounding-circle approximation, see DESIGN.md section 8)., A fixed set of beams, each cast against circular obstacles. Beam angles are…, UltrasonicArray, Hand-computed ray/circle intersection cases for UltrasonicArray.sense -- direct…, Vehicle at the origin facing +x, obstacle centered at (5, 0) r=1 -- the ray…, The near edge (9.0) is still farther than max_range (5.0), so the beam reports…, Obstacle at (-5, 0), beam pointing along +x -- both ray/circle intersection… (+12 more)

### Community 9 - "MPC Control & Vehicle Model"
Cohesion: 0.12
Nodes (15): Short-horizon nonlinear MPC path tracker. See DESIGN.md section 7. Each call…, Load scenario YAML files into Vehicle + Environment objects., Scenario, ndarray, Kinematic bicycle model. See DESIGN.md section 3., Kinematic bicycle model. All angles (theta, delta) are radians., Minimum turning radius at max_steer: L / tan(max_steer). See DESIGN.md section…, Advance state by speed v (m/s, signed) and steering delta (rad) over dt seconds. (+7 more)

### Community 10 - "Hybrid A* Replanning"
Cohesion: 0.15
Nodes (22): Target parking spot center, plus its heading for perpendicular vs. parallel…, Spot, Published by ControllerNode when the speed governor has been binding for long…, ReplanRequestMsg, HybridAStarPlanner, _pose(), KNOWN_BUGS.md entry 3 / IMPLEMENTATION.md's M4 entry: PlannerNode used to plan…, Found in a second code-review pass: PlannerNode marks itself `_planned = True`… (+14 more)

### Community 11 - "Lane Geometry & Full Highway Harness"
Cohesion: 0.16
Nodes (15): build_arc_length_table(), pose_at_arc_length(), project_to_arc_length(), ndarray, Arc-length bookkeeping for a curved lane centerline (full closed-loop highway…, Geometric arc length (m) at each waypoint of an (N,3) x/y/theta centerline,…, Nearest-waypoint projection of a point onto the centerline, returned as that…, Inverse of project_to_arc_length -- for scenario setup only (placing a… (+7 more)

### Community 12 - "Controller Node & Path Messages"
Cohesion: 0.16
Nodes (16): ObstacleRangeMsg, PathMsg, Ultrasonic beam readings, angle (rad, relative to heading) -> range (m)., ControllerNode, _distance_to_polyline(), ndarray, Wraps the existing Controller (Pure Pursuit or MPC) unchanged -- both already…, `tracked_stopping_buffer` while accurately tracking the current path, else the… (+8 more)

### Community 13 - "ACC Highway Harness"
Cohesion: 0.19
Nodes (11): AccHarness, AccSimulationResult, ndarray, Tick-based executor for H1/H2 (ACC + fused ego speed): owns the Bus, builds the…, EgoLongitudinalStateMsg, LeadVehicleStateMsg, LeadVehicleNode, Replays a real recorded lead-vehicle trajectory (e.g. from NGSIM) tick by tick.… (+3 more)

### Community 14 - "Controller Node Tests"
Cohesion: 0.14
Nodes (19): _FixedController, _node_with_pose_and_path(), ControllerNode's speed governor (see its module docstring, KNOWN_BUGS.md entry…, tracked_stopping_buffer=None (the ControllerNode default) must disable the…, Stub Controller: always returns the same (v, delta), regardless of path --…, Builds a fresh ControllerNode, feeds it one obstacle_ranges reading at `angle`,…, The actual bug: previously a rear-only obstacle was invisible to the governor…, Regression guard for harness.py's DEFAULT_SENSOR_ANGLES: at least one beam… (+11 more)

### Community 15 - "Message Bus"
Cohesion: 0.12
Nodes (11): Callback, Bus, Any, Minimal synchronous pub/sub bus. See DESIGN.md section 2 (pub/sub…, ndarray, ndarray, Publishes a scripted other-vehicle's status over the Bus each tick (H5 Phase…, test_multiple_publishes_delivered_in_order() (+3 more)

### Community 16 - "Estimator Node"
Cohesion: 0.20
Nodes (8): LandmarkBearingMsg, LandmarkReading, PositionFixMsg, Low-rate absolute (x, y) fix -- e.g. a garage RTLS/UWB-anchor system. Does not…, EstimatorNode, Wraps the EKF: predicts on odometry, corrects on compass/position_fix/landmark…, Perception node: the only node besides the harness allowed to see true_state.…, SensorNode

### Community 17 - "RL Parking Env"
Cohesion: 0.17
Nodes (14): ParkingEnv, ndarray, Fast, deterministic unit coverage for ParkingEnv's Gym contract and reward…, Standard Gym API-contract checker: correct reset()/step() signatures, spaces…, Places the vehicle directly facing an obstacle at close range, rather than…, An out-of-range action shouldn't silently drive the vehicle faster/harder than…, Regression for a specific design choice: success is position-only (matches…, test_action_is_clipped_to_the_declared_bounds() (+6 more)

### Community 18 - "Full Highway Tests"
Cohesion: 0.18
Nodes (18): parametrize, Both hard invariants asserted on ONE run, not just as separate parametrized…, The specific composition edge case DESIGN.md section 12's H5 Phase B entry…, Safety is a hard pass/fail, evaluated against a real recorded leader…, Plausibility bar reused from H3's own standalone validation…, Same 0.2x-3.0x band as test_acc_validation.py's plausibility check -- not a…, _run(), _run_with_intersection() (+10 more)

### Community 19 - "Full Highway Harness Core"
Cohesion: 0.14
Nodes (6): FullHighwayHarness, ndarray, OtherVehicleScript, LongitudinalArbiterNode, OtherVehicleScriptNode, OtherVehicleScript

### Community 20 - "Intersection Harness Tests"
Cohesion: 0.24
Nodes (16): no_other_vehicle(), other_vehicle_present_from(), OtherVehicleScript, Runs an IntersectionNavigator against a scripted other-vehicle scenario. See…, run_intersection_scenario(), Other vehicle stops well before ego does and hasn't cleared yet -- ego must…, Other vehicle doesn't stop until well after ego already has -- ego shouldn't…, Same timing as the yield-to-the-right case, but the other vehicle is on the… (+8 more)

### Community 21 - "RL Comparison Validation"
Cohesion: 0.19
Nodes (14): compare(), ComparisonSummary, evaluate_baseline(), evaluate_rl_policy(), main(), PPO, Evaluates a trained ParkingEnv policy against the planner+controller baseline…, Pins the real, measured RL-vs-baseline comparison… (+6 more)

### Community 22 - "IDM ACC Controller"
Cohesion: 0.16
Nodes (11): IDMController, Adaptive cruise control: two longitudinal controllers, the same…, Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000) -- the literature-…, parametrize, Raw IDM's (s*/gap)^2 term is unbounded as gap -> 0; a real car can't actually…, Code-review finding: MpcAccController used to default a_min=-3.0 while…, test_follows_a_braking_lead_without_collision(), test_idm_accelerates_toward_desired_speed_with_no_lead_constraint() (+3 more)

### Community 23 - "Demo & Simulation Tests"
Cohesion: 0.22
Nodes (13): main(), list_scenarios(), _combinations(), parametrize, Evaluated statistically, not as single-run determinism: with sensor/odometry…, Safety must hold regardless of estimation noise, on every seed, for every…, Pins down NEVER_SUCCEEDS's claim as an actual regression guard, not just a…, Regression guard for the prototype bug where some scenarios passed degrees… (+5 more)

### Community 24 - "Core Interfaces & Harness"
Cohesion: 0.23
Nodes (8): Parking lot geometry: spots and obstacles. See DESIGN.md section 1/8., Tick-based executor: owns the Bus, builds all 5 nodes, and drives them in a…, Controller, Planner, Protocol, Shared structural types for planners and controllers. See IMPLEMENTATION.md…, Wraps the existing Planner (e.g. DubinsPlanner) unchanged. Plans once, off the…, Simulated ultrasonic ray-cast sensor array. See DESIGN.md section 4.

### Community 25 - "Parking Harness & Planner Node"
Cohesion: 0.22
Nodes (4): ParkingHarness, `on_tick(tick)`, if given, runs before each tick's nodes step -- its only use…, PoseEstimateMsg, PlannerNode

### Community 26 - "ACC Validation"
Cohesion: 0.24
Nodes (11): AccValidationResult, main(), plot_validation(), Validates control/acc.py's controllers against a real recorded car-following…, validate(), parametrize, Safety is a hard pass/fail, evaluated against a real recorded car-following…, Not a strict match (the real driver isn't assumed optimal), but our controller… (+3 more)

### Community 27 - "MPC & Lane Centering Control"
Cohesion: 0.24
Nodes (6): ndarray, MPCController, ndarray, HasPose, Anything with .x/.y/.theta -- a real Vehicle, or (in the node architecture) a…, ndarray

### Community 28 - "True State & Sensor Node Tests"
Cohesion: 0.23
Nodes (10): TrueStateMsg, SensorNode's dropout/latency modeling (DESIGN.md section 10's future-extensions…, A middling dropout_prob should deliver *some* but not *all* compass readings…, The delayed message should carry the value computed *when the reading was…, _sensor_node(), test_dropout_is_probabilistic_not_all_or_nothing(), test_dropout_prob_one_never_delivers_anything(), test_latency_delays_delivery_by_exactly_latency_ticks() (+2 more)

### Community 29 - "ACC Controller Node"
Cohesion: 0.21
Nodes (6): RadarMsg, Noisy forward-radar reading: bumper-to-bumper range and closing range-rate., AccController, AccControllerNode, Protocol, Wraps an ACC controller (IDM or MPC, control/acc.py). Consumes radar (range,…

### Community 30 - "NGSIM Loader Validation"
Cohesion: 0.26
Nodes (12): load_following_pair(), load_lane_centerline(), NgsimFollowingPair, NgsimTrajectory, ndarray, Path, Parses NGSIM vehicle-trajectory excerpts into leader/follower trajectory pairs…, Returns an (N, 3) x/y/theta path along a real, NGSIM-derived lane centerline… (+4 more)

### Community 31 - "MPC-ACC Controller"
Cohesion: 0.29
Nodes (6): MpcAccController, ndarray, A per-horizon-step floor that's *always* achievable, fixing the actual defect…, Short-horizon MPC for ACC. Unlike control/mpc.py's parking MPC, which only uses…, Code-review finding: neither result.success nor the gap constraint itself used…, test_mpc_falls_back_to_safe_braking_if_solver_returns_a_constraint_violating_point()

### Community 32 - "Stanley Lane Centering"
Cohesion: 0.24
Nodes (9): Stanley lane-centering controller -- the classical lateral-control law, playing…, StanleyController, Same check, offset to the other side -- catches a sign bug that only happens to…, Stanley's correction term divides by speed; without a floor this would produce…, Regression guard for a real sign-convention bug found while building this: the…, test_converges_from_a_lateral_offset_on_a_straight_lane(), test_converges_from_the_opposite_lateral_offset_too(), test_near_zero_speed_does_not_blow_up() (+1 more)

### Community 33 - "RL Training"
Cohesion: 0.24
Nodes (9): Gymnasium environment wrapping the parking simulation for a learned end-to-end…, main(), make_env(), PPO, Trains a PPO policy on ParkingEnv. See core/rl/parking_env.py for the…, train(), Monitor, Smoke test for the training pipeline itself -- confirms PPO can actually train… (+1 more)

### Community 34 - "Sensor Robustness Tests"
Cohesion: 0.26
Nodes (11): load_scenario(), _min_clearance(), ndarray, parametrize, End-to-end robustness of the closed loop under sensor dropout/latency…, Minimum signed vehicle-to-obstacle clearance across a run: negative means the…, Regression for the fix itself: on this exact case, a controller that doesn't…, _run() (+3 more)

### Community 35 - "Lane Centering Validation"
Cohesion: 0.27
Nodes (9): LaneCenteringResult, main(), plot_validation(), Validates control/lane_centering.py's Stanley controller against a real, NGSIM-…, validate(), parametrize, The plausibility bar: after settling, tracking error should stay within the…, test_converges_and_stays_within_real_driver_scatter() (+1 more)

### Community 36 - "Environment & Animation Visualization"
Cohesion: 0.33
Nodes (8): Environment, SimulationResult, _axis_bounds(), _ellipse_params(), ndarray, Matplotlib animation of a parking run: a top-down view plus two live telemetry…, 1-sigma position-uncertainty ellipse: (width, height, angle_degrees)., render_animation()

### Community 37 - "Control Convergence Tests"
Cohesion: 0.33
Nodes (9): ndarray, parametrize, Direct convergence unit coverage for the two path-tracking controllers,…, A real tracking test, not just "drive straight ahead": start 1.5m off the…, _run(), _straight_path(), test_converges_to_the_goal_from_a_lateral_offset(), test_converges_to_the_goal_from_directly_on_the_path() (+1 more)

### Community 38 - "Vehicle Node"
Cohesion: 0.28
Nodes (4): OdometryMsg, Noisy wheel/steering-encoder reading of the *actually applied* v, delta., Ground-truth plant node: owns the real Vehicle, applies the last commanded…, VehicleNode

### Community 39 - "Pure Pursuit Control"
Cohesion: 0.29
Nodes (4): PurePursuitAdaptive, ndarray, Geometric Pure Pursuit path tracker, with automatic forward/reverse selection.…, Pure Pursuit that reverses automatically when the lookahead point lies behind…

### Community 40 - "Graphify Exports (Neo4j/FalkorDB/MCP)"
Cohesion: 0.33
Nodes (6): FalkorDB export / --falkordb-push, exports.md reference doc, --mcp stdio server (graphify.serve), Neo4j export / --neo4j-push, Token reduction benchmark (>5000 words), Steps 6b-8: Wiki, Neo4j, FalkorDB, SVG, GraphML, MCP, benchmark

### Community 41 - "Controller/Planner Interfaces"
Cohesion: 0.33
Nodes (4): ndarray, Pose, Return an (N, 3) array of x, y, theta waypoints from start to goal., Return (v_desired, delta) given the current pose (estimate) and a path to track.

### Community 42 - "Graphify GitHub & Merge"
Cohesion: 0.50
Nodes (5): graphify clone <url>, github-and-merge.md reference doc, graphify merge-graphs, Multi-subfolder / monorepo merge flow, graphify Skill (SKILL.md)

### Community 43 - "Graphify Transcription"
Cohesion: 0.50
Nodes (4): transcribe.md reference doc, Step 2.5 transcription flow, Whisper domain-hint prompt (from god node labels), Step 2.5: Video and audio (conditional)

## Knowledge Gaps
- **47 isolated node(s):** `auto-park`, `graphify Skill (SKILL.md)`, `Step 1: Ensure graphify is installed`, `Step 2: Detect files`, `Step 3: Extract entities and relationships` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Bus` connect `Message Bus` to `Extended Kalman Filter`, `Environment & Animation Visualization`, `ROS2 Bridge Messaging`, `Vehicle Node`, `Vehicle Messages & Longitudinal Node`, `Sensors & Obstacle Environment`, `Hybrid A* Replanning`, `Lane Geometry & Full Highway Harness`, `Controller Node & Path Messages`, `ACC Highway Harness`, `Controller Node Tests`, `Estimator Node`, `Full Highway Harness Core`, `Core Interfaces & Harness`, `Parking Harness & Planner Node`, `True State & Sensor Node Tests`, `ACC Controller Node`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `Vehicle` connect `MPC Control & Vehicle Model` to `Stanley Lane Centering`, `RL Training`, `Sensor Robustness Tests`, `Lane Centering Validation`, `Environment & Animation Visualization`, `Extended Kalman Filter`, `Vehicle Node`, `Vehicle Messages & Longitudinal Node`, `Sensors & Obstacle Environment`, `Control Convergence Tests`, `Hybrid A* Replanning`, `Lane Geometry & Full Highway Harness`, `Controller Node & Path Messages`, `RL Parking Env`, `Full Highway Harness Core`, `Core Interfaces & Harness`, `Parking Harness & Planner Node`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `wrap_angle()` connect `Extended Kalman Filter` to `Stanley Lane Centering`, `Dubins Path Planning`, `RL Training`, `Intersection Navigation & Geometry`, `Vehicle Messages & Longitudinal Node`, `Unscented Kalman Filter`, `Pure Pursuit Control`, `MPC Control & Vehicle Model`, `Controller Node & Path Messages`, `Controller Node Tests`, `Estimator Node`, `RL Parking Env`, `MPC & Lane Centering Control`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `Bus` (e.g. with `FullHighwayHarness` and `FullHighwaySimulationResult`) actually correct?**
  _`Bus` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `Vehicle` (e.g. with `FullHighwayHarness` and `FullHighwaySimulationResult`) actually correct?**
  _`Vehicle` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ExtendedKalmanFilter` (e.g. with `FullHighwayHarness` and `FullHighwaySimulationResult`) actually correct?**
  _`ExtendedKalmanFilter` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Obstacle` (e.g. with `Controller` and `HasPose`) actually correct?**
  _`Obstacle` has 11 INFERRED edges - model-reasoned connections that need verification._