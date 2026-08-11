"""Wraps the existing Planner (e.g. DubinsPlanner) unchanged. Plans once, off the
first pose_estimate it receives, and publishes path -- matching the M1 baseline's
"plan once up front" behavior (no replanning yet; that's M2/M4 future work). Critically,
it plans from the vehicle's ESTIMATED pose, not ground truth: a real planner never gets
to see the true state either.
"""

from core.environment import Environment
from core.interfaces import Planner
from core.messaging.bus import Bus
from core.messaging.messages import PathMsg, PoseEstimateMsg


class PlannerNode:
    def __init__(self, bus: Bus, planner: Planner, environment: Environment, turning_radius: float):
        self.bus = bus
        self.planner = planner
        self.environment = environment
        self.turning_radius = turning_radius
        self._planned = False
        bus.subscribe("pose_estimate", self._on_pose_estimate)

    def _on_pose_estimate(self, msg: PoseEstimateMsg) -> None:
        if self._planned:
            return
        self._planned = True
        start = (msg.x, msg.y, msg.theta)
        goal = (self.environment.spot.x, self.environment.spot.y, self.environment.spot.theta)
        path = self.planner.plan(start, goal, self.environment.obstacles, self.turning_radius)
        self.bus.publish("path", PathMsg(path))
