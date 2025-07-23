import numpy as np

class Vehicle:
    """Simple kinematic bicycle model."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheelbase = wheelbase

    def update(self, v, delta, dt):
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += (v / self.wheelbase) * np.tan(delta) * dt


class UltrasonicSensor:
    """Simulates ultrasonic beams measuring distance to circular obstacles."""
    def __init__(self, angles, max_range=5.0):
        self.angles = angles
        self.max_range = max_range

    def sense(self, vehicle, obstacles):
        readings = {}
        for ang in self.angles:
            ray = vehicle.theta + ang
            min_d = self.max_range
            for (cx, cy, r) in obstacles:
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


class PurePursuitController:
    """Pure Pursuit path follower for forward or reverse driving."""
    def __init__(self, wheelbase, lookahead=2.0, speed=1.0):
        self.L = wheelbase
        self.ld = lookahead
        self.speed = speed

    def control(self, vehicle, path, reverse=False):
        dists = np.hypot(path[:,0] - vehicle.x, path[:,1] - vehicle.y)
        idxs = np.where(dists >= self.ld)[0]
        if len(idxs):
            tx, ty = path[idxs[0]]
        else:
            tx, ty = path[-1]
        dx = tx - vehicle.x
        dy = ty - vehicle.y
        alpha = np.arctan2(dy, dx) - vehicle.theta
        delta = np.arctan2(2*self.L*np.sin(alpha), self.ld)
        v = -self.speed if reverse else self.speed
        return v, delta


def generate_perpendicular_parking_path(spot, approach=5.0):
    """
    Curved path into a perpendicular spot:
      1) start behind spot on lane
      2) intermediate turn point
      3) spot center
    """
    x0 = spot['x'] - approach
    y0 = spot['y']
    xm = spot['x'] - approach/2
    ym = spot['y'] + approach/2
    return np.array([[x0, y0],
                     [xm, ym],
                     [spot['x'], spot['y']]])


def generate_parallel_parking_path(spot, approach=5.0):
    """
    Curved path into a parallel spot:
      1) start beside spot on lane
      2) intermediate turn point
      3) spot corner
    """
    x0 = spot['x']
    y0 = spot['y'] + approach
    xm = spot['x'] + approach/2
    ym = spot['y'] + approach/2
    return np.array([[x0, y0],
                     [xm, ym],
                     [spot['x'], spot['y']]])


class ParkingController:
    """High‑level parking maneuvers + simple obstacle checking."""
    def __init__(self, vehicle, obstacles, sensor_angles=None, sensor_range=2.0, dt=0.1):
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.dt = dt
        self.sensor = UltrasonicSensor(
            angles=sensor_angles or [0, np.pi/4, -np.pi/4],
            max_range=sensor_range
        )
        self.pure_pursuit = PurePursuitController(
            wheelbase=vehicle.wheelbase,
            lookahead=1.5,
            speed=1.0
        )
        self.history = []
        self.controls = []

    def run(self, spot, mode='perpendicular'):
        path = (generate_perpendicular_parking_path(spot)
                if mode=='perpendicular'
                else generate_parallel_parking_path(spot))
        reverse = True
        for _ in range(500):
            # stop if close enough
            if np.hypot(self.vehicle.x - path[-1,0],
                        self.vehicle.y - path[-1,1]) < 0.2:
                break

            # obstacle check
            reads = self.sensor.sense(self.vehicle, self.obstacles)
            if min(reads.values()) < 0.3:
                v, delta = 0.0, 0.0
            else:
                v, delta = self.pure_pursuit.control(self.vehicle, path, reverse)

            self.vehicle.update(v, delta, self.dt)
            self.history.append((self.vehicle.x, self.vehicle.y, self.vehicle.theta))
            self.controls.append((v, delta))

        return np.array(self.history), np.array(self.controls)
