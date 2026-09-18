"""Wraps the existing Planner unchanged: plans once from the first pose_estimate, off the
ESTIMATED pose not ground truth. Re-plans on `replan_request`, capped at `max_replans`
(see KNOWN_BUGS.md entry 3). A planner that raises leaves the old path in place."""

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
