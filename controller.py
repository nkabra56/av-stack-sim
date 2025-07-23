# controller.py

import numpy as np

def wrap_angle(a):
    """Normalize angle to [–π, π]."""
    return (a + np.pi) % (2*np.pi) - np.pi

class Vehicle:
    """Simple kinematic bicycle model."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheelbase = wheelbase

    def update(self, v, delta, dt):
        """Advance state by speed v (m/s), steering δ (rad), over dt seconds."""
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += (v / self.wheelbase) * np.tan(delta) * dt


class UltrasonicSensor:
    """Simulate ultrasonic beams measuring distance to circular obstacles."""
    def __init__(self, angles, max_range=5.0):
        self.angles = angles
        self.max_range = max_range

    def sense(self, veh, obstacles):
        """Return dict {angle: distance} for each beam."""
        readings = {}
        for ang in self.angles:
            ray = veh.theta + ang
            dmin = self.max_range
            for cx, cy, r in obstacles:
                dx, dy = np.cos(ray), np.sin(ray)
                ex, ey = veh.x - cx, veh.y - cy
                A = dx*dx + dy*dy
                B = 2*(dx*ex + dy*ey)
                C = ex*ex + ey*ey - r*r
                disc = B*B - 4*A*C
                if disc < 0:
                    continue
                for t in [(-B + np.sqrt(disc)) / (2*A), (-B - np.sqrt(disc)) / (2*A)]:
                    if 0 <= t < dmin:
                        dmin = t
            readings[ang] = dmin
        return readings


def generate_perpendicular_parking_path(spot, approach=5.0, npts=120):
    """Quadratic Bézier P0→P1→P2 into a perpendicular spot."""
    P0 = np.array([spot['x'] - approach, spot['y']])
    P1 = np.array([spot['x'] - approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'],              spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    return (1-t)**2*P0 + 2*(1-t)*t*P1 + t**2*P2


def generate_parallel_parking_path(spot, approach=5.0, npts=120):
    """Quadratic Bézier P0→P1→P2 into a parallel spot."""
    P0 = np.array([spot['x'],              spot['y'] + approach])
    P1 = np.array([spot['x'] + approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'],              spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    return (1-t)**2*P0 + 2*(1-t)*t*P1 + t**2*P2


class PurePursuitAdaptive:
    """Pure Pursuit that automatically reverses if the target lies behind."""
    def __init__(self, wheelbase, lookahead=2.0, v_max=1.0):
        self.L = wheelbase
        self.ld = lookahead
        self.v_max = v_max

    def control(self, veh, path):
        # compute distances to all waypoints
        d = np.hypot(path[:,0] - veh.x, path[:,1] - veh.y)
        # pick lookahead point
        idx = np.where(d >= self.ld)[0]
        if len(idx):
            tx, ty = path[idx[0]]
        else:
            tx, ty = path[-1]
        # steering
        dx, dy = tx - veh.x, ty - veh.y
        alpha = wrap_angle(np.arctan2(dy, dx) - veh.theta)
        delta = np.arctan2(2*self.L*np.sin(alpha), self.ld)
        # forward/reverse decision
        direction = -1.0 if abs(alpha) > np.pi/2 else 1.0
        # desired speed
        rho = np.hypot(dx, dy)
        v_des = direction * self.v_max * np.tanh(rho)
        return v_des, delta, rho


class AdaptiveParkingController:
    """
    Autonomous parking:
      • Plans a smooth Bezier path into the spot
      • Uses ultrasonic front‐beam to brake on obstacles
      • Follows path with PurePursuitAdaptive
      • Generates acceleration/braking commands
    """
    def __init__(self,
                 vehicle,
                 obstacles,
                 sensor_angles=None,
                 sensor_range=5.0,
                 dt=0.1,
                 v_max=1.0,
                 a_max=0.5,
                 K_acc=2.0,
                 tol=0.1):
        self.veh       = vehicle
        self.obstacles = obstacles
        self.dt        = dt
        self.v_max     = v_max
        self.a_max     = a_max
        self.K_acc     = K_acc
        self.tol       = tol
        self.v         = 0.0
        self.sensor    = UltrasonicSensor(sensor_angles or [0], max_range=sensor_range)
        self.pp        = PurePursuitAdaptive(vehicle.wheelbase, lookahead=2.0, v_max=v_max)
        self.history   = []
        self.controls  = []  # list of (v, a, δ)

    def run(self, spot, mode='perpendicular'):
        # pick path
        if mode == 'perpendicular':
            path = generate_perpendicular_parking_path(spot)
        else:
            path = generate_parallel_parking_path(spot)

        for _ in range(2000):
            # 1) obstacle check (front beam only)
            d_front = self.sensor.sense(self.veh, self.obstacles)[0]
            # if too close, command v_des=0
            if d_front < 0.5:
                v_des = 0.0
            else:
                # 2) Pure Pursuit adaptive for v_des & δ
                v_des, δ, rho = self.pp.control(self.veh, path)
            # 3) compute acceleration to reach v_des
            a = self.K_acc * (v_des - self.v)
            a = np.clip(a, -self.a_max, self.a_max)
            # 4) update speed
            self.v = np.clip(self.v + a*self.dt, -self.v_max, self.v_max)
            # 5) apply motion
            self.veh.update(self.v, δ, self.dt)
            # 6) record
            self.history.append((self.veh.x, self.veh.y, self.veh.theta))
            self.controls.append((self.v, a, δ))
            # 7) stop if within tolerance of the spot center
            dx = spot['x'] - self.veh.x
            dy = spot['y'] - self.veh.y
            if np.hypot(dx, dy) < self.tol:
                break

        return np.array(self.history), np.array(self.controls)
