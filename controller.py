import numpy as np

class Vehicle:
    """Kinematic bicycle model."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheelbase = wheelbase

    def update(self, v, delta, dt):
        """Advance state by speed v (m/s), steering δ (rad) over dt (s)."""
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += (v / self.wheelbase) * np.tan(delta) * dt


class UltrasonicSensor:
    """Ray–circle intersection for simple obstacle detection."""
    def __init__(self, angles, max_range=5.0):
        self.angles = angles
        self.max_range = max_range

    def sense(self, vehicle, obstacles):
        readings = {}
        for ang in self.angles:
            ray = vehicle.theta + ang
            min_d = self.max_range
            for cx, cy, r in obstacles:
                d = self._ray_circle_dist(vehicle.x, vehicle.y, ray, cx, cy, r)
                if d is not None and d < min_d:
                    min_d = d
            readings[ang] = min_d
        return readings

    def _ray_circle_dist(self, x0, y0, ray, cx, cy, r):
        dx, dy = np.cos(ray), np.sin(ray)
        ex, ey = x0 - cx, y0 - cy
        A = dx*dx + dy*dy
        B = 2*(dx*ex + dy*ey)
        C = ex*ex + ey*ey - r*r
        disc = B*B - 4*A*C
        if disc < 0:
            return None
        t1 = (-B + np.sqrt(disc)) / (2*A)
        t2 = (-B - np.sqrt(disc)) / (2*A)
        ts = [t for t in (t1, t2) if t >= 0]
        return min(ts) if ts else None


def generate_perpendicular_parking_path(spot, approach=5.0, npts=100):
    """
    Smooth quadratic‐Bezier path for perpendicular parking:
      P0 = start, P1 = mid, P2 = goal
    """
    P0 = np.array([spot['x'] - approach, spot['y']])
    P1 = np.array([spot['x'] - approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'], spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2


def generate_parallel_parking_path(spot, approach=5.0, npts=100):
    """Smooth quadratic‐Bezier path for parallel parking."""
    P0 = np.array([spot['x'], spot['y'] + approach])
    P1 = np.array([spot['x'] + approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'], spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2


class PurePursuitController:
    """Pure Pursuit with deadband to reduce oscillation."""
    def __init__(self, wheelbase, lookahead=1.5, max_speed=1.0,
                 deadband_deg=5.0):
        self.L = wheelbase
        self.ld = lookahead
        self.max_speed = max_speed
        # steering deadband (radians)
        self.deadband = np.radians(deadband_deg)

    def control(self, vehicle, path, reverse=False):
        # compute distances to waypoints
        d = np.hypot(path[:,0] - vehicle.x,
                     path[:,1] - vehicle.y)
        # pick first beyond lookahead, else last
        idxs = np.where(d >= self.ld)[0]
        if len(idxs):
            tx, ty = path[idxs[0]]
        else:
            # end of path => stop
            return 0.0, 0.0
        # steering error
        dx, dy = tx - vehicle.x, ty - vehicle.y
        alpha = np.arctan2(dy, dx) - vehicle.theta
        # wrap to [-π,π]
        alpha = (alpha + np.pi) % (2*np.pi) - np.pi
        # apply deadband
        if abs(alpha) < self.deadband:
            delta = 0.0
        else:
            delta = np.arctan2(2*self.L*np.sin(alpha), self.ld)
        # speed scaling (smooth decel)
        rho = np.hypot(dx, dy)
        speed = self.max_speed * np.tanh(rho)
        v = -speed if reverse else speed
        return v, delta


class ParkingController:
    """High‑level parking using Pure Pursuit + obstacle check."""
    def __init__(self, vehicle, obstacles,
                 sensor_angles=None, sensor_range=2.0,
                 dt=0.1, tol=0.1):
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.dt = dt
        self.tol = tol
        self.sensor = UltrasonicSensor(
            angles=sensor_angles or [0, np.pi/4, -np.pi/4],
            max_range=sensor_range
        )
        self.pp = PurePursuitController(
            wheelbase=vehicle.wheelbase,
            lookahead=1.5,
            max_speed=1.0,
            deadband_deg=5.0
        )
        self.history = []
        self.controls = []

    def run(self, spot, mode='perpendicular', max_steps=500):
        # build dense path
        if mode == 'perpendicular':
            path = generate_perpendicular_parking_path(spot)
        else:
            path = generate_parallel_parking_path(spot)

        reverse = True
        for _ in range(max_steps):
            # obstacle proximity stop
            if min(self.sensor.sense(self.vehicle, self.obstacles).values()) < 0.3:
                break
            # compute control
            v, delta = self.pp.control(self.vehicle, path, reverse)
            # stop if at goal
            dx = self.vehicle.x - path[-1,0]
            dy = self.vehicle.y - path[-1,1]
            if np.hypot(dx, dy) < self.tol or v == 0.0:
                break
            # apply
            self.vehicle.update(v, delta, self.dt)
            self.history.append((self.vehicle.x,
                                 self.vehicle.y,
                                 self.vehicle.theta))
            self.controls.append((v, delta))

        return np.array(self.history), np.array(self.controls)
