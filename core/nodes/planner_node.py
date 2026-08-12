"""Wraps the existing Planner (e.g. DubinsPlanner) unchanged. Plans once, off the
first pose_estimate it receives, and publishes path. Critically, it plans from the
vehicle's ESTIMATED pose, not ground truth: a real planner never gets to see the true
state either.

**Re-planning** (see KNOWN_BUGS.md bug 3 / IMPLEMENTATION.md's M4 entry, "still open:
wiring PlannerNode to re-plan"): `ControllerNode` publishes `replan_request` once its
speed governor has been binding long enough to look like a genuine stall. On that
signal, this node re-plans from the *latest* pose estimate against the environment's
*current* obstacle list -- re-reading `self.environment.obstacles` live (not the
snapshot from the first plan) is what actually closes the bug: a scenario that adds an
obstacle to that list mid-run (see `tests/test_replanning.py`) is invisible to the
original plan but fully visible to a re-plan. Capped at `max_replans` attempts so a
combination that's stuck for a structural reason (e.g. no route exists, or a controller
that can't track any route through this geometry -- see KNOWN_BUGS.md entry 2) doesn't
re-run an expensive Hybrid A* search forever; past the cap, the vehicle just stays
governed to a safe stop, exactly like before this fix, rather than looping. A planner
that raises (no route currently exists -- every shipped `Planner` fails with some
`RuntimeError` subclass: `PlanningFailure`, or a plain `RuntimeError` for
Dubins/ReedsShepp) leaves the old path in place rather than crashing the simulation.
"""

from core.environment import Environment
from core.interfaces import Planner
from core.messaging.bus import Bus
from core.messaging.messages import PathMsg, PoseEstimateMsg, ReplanRequestMsg


class PlannerNode:
    def __init__(
        self, bus: Bus, planner: Planner, environment: Environment, turning_radius: float, max_replans: int = 3
    ):
        self.bus = bus
        self.planner = planner
        self.environment = environment
        self.turning_radius = turning_radius
        self.max_replans = max_replans
        self._planned = False
        self._replans = 0
        self._latest_pose: PoseEstimateMsg | None = None
        bus.subscribe("pose_estimate", self._on_pose_estimate)
        bus.subscribe("replan_request", self._on_replan_request)

    def _on_pose_estimate(self, msg: PoseEstimateMsg) -> None:
        self._latest_pose = msg
        if self._planned:
            return
        self._planned = True
        self._plan_from(msg)

    def _on_replan_request(self, _msg: ReplanRequestMsg) -> None:
        if self._replans >= self.max_replans or self._latest_pose is None:
            return
        self._replans += 1
        self._plan_from(self._latest_pose)

    def _plan_from(self, pose: PoseEstimateMsg) -> None:
        start = (pose.x, pose.y, pose.theta)
        goal = (self.environment.spot.x, self.environment.spot.y, self.environment.spot.theta)
        try:
            path = self.planner.plan(start, goal, self.environment.obstacles, self.turning_radius)
        except RuntimeError:
            return  # no route currently exists; keep tracking the last known-good path
        self.bus.publish("path", PathMsg(path))
