"""Ground-truth plant node: owns the real Vehicle, applies the last commanded control,
and publishes true_state + noisy odometry each tick. See DESIGN.md's architecture
diagram: true_state is subscribed only by SensorNode and the harness's own evaluation
logic, never by the estimator/planner/controller.

Acceleration limiting (bounding how fast commanded speed can actually change) and
steering-angle clamping live here, not in the controller: they're physical actuator
limits of the plant, not part of any controller's control law. ControllerNode is free
to command an unreachable v_desired; this node is what enforces what the vehicle can
actually do about it.
"""

import numpy as np

from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import ControlCmdMsg, OdometryMsg, TrueStateMsg
from auto_park.vehicle import Vehicle


class VehicleNode:
    def __init__(
        self,
        bus: Bus,
        vehicle: Vehicle,
        dt: float,
        rng: np.random.Generator,
        v_max: float = 1.5,
        a_max: float = 0.8,
        k_acc: float = 2.0,
        odom_v_std: float = 0.03,
        odom_delta_std: float = 0.01,
    ):
        self.bus = bus
        self.vehicle = vehicle
        self.dt = dt
        self.rng = rng
        self.v_max = v_max
        self.a_max = a_max
        self.k_acc = k_acc
        self.odom_v_std = odom_v_std
        self.odom_delta_std = odom_delta_std

        self._v = 0.0  # actual current speed (after accel limiting), not a control input
        self._last_cmd = ControlCmdMsg(0.0, 0.0)
        bus.subscribe("control_cmd", self._on_control_cmd)

    def _on_control_cmd(self, msg: ControlCmdMsg) -> None:
        self._last_cmd = msg

    def step(self) -> None:
        v_desired = self._last_cmd.v
        delta = np.clip(self._last_cmd.delta, -self.vehicle.max_steer, self.vehicle.max_steer)

        a = np.clip(self.k_acc * (v_desired - self._v), -self.a_max, self.a_max)
        self._v = float(np.clip(self._v + a * self.dt, -self.v_max, self.v_max))

        self.vehicle.update(self._v, delta, self.dt)
        self.bus.publish(
            "true_state", TrueStateMsg(self.vehicle.x, self.vehicle.y, self.vehicle.theta, self._v, delta)
        )

        v_meas = self._v + self.rng.normal(0.0, self.odom_v_std)
        delta_meas = delta + self.rng.normal(0.0, self.odom_delta_std)
        self.bus.publish("odometry", OdometryMsg(v_meas, delta_meas, self.dt))
