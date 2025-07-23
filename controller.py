import numpy as np

class Vehicle:
    """Simple kinematic bicycle model."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.5):
        self.x = x
        self.y = y
        self.theta = theta  # heading in radians
        self.wheelbase = wheelbase

    def update(self, v, delta, dt):
        """Advance by speed v, steering δ, over dt seconds."""
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += (v / self.wheelbase) * np.tan(delta) * dt


class UltrasonicSensor:
    """Ray–circle intersection distance for simple obstacle avoidance."""
    def __init__(self, angles, max_range=5.0):
        self.angles = angles      # beam angles relative to vehicle heading
        self.max_range = max_range

    def sense(self, vehicle, obstacles):
        readings = {}
        for ang in self.angles:
            ray = vehicle.theta + ang
            min_d = self.max_range
            for cx, cy, r in obstacles:
                d = self._ray_circle_dist(vehicle.x, vehicle.y, ray, cx, cy, r)
                if d and d < min_d:
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


def generate_perpendicular_parking_path(spot, approach=5.0, npts=60):
    """
    Returns a smooth, dense list of waypoints for perpendicular parking.
    spot: {'x','y'} center of spot
    """
    P0 = np.array([spot['x'] - approach, spot['y']])
    P1 = np.array([spot['x'] - approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'], spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    # simple quadratic Bezier: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
    return (1- t)**2 * P0 + 2*(1- t)*t * P1 + t**2 * P2


def generate_parallel_parking_path(spot, approach=5.0, npts=60):
    """
    Returns a smooth, dense list of waypoints for parallel parking.
    spot: {'x','y'} rear corner of spot
    """
    P0 = np.array([spot['x'], spot['y'] + approach])
    P1 = np.array([spot['x'] + approach/2, spot['y'] + approach/2])
    P2 = np.array([spot['x'], spot['y']])
    t = np.linspace(0,1,npts)[:,None]
    return (1- t)**2 * P0 + 2*(1- t)*t * P1 + t**2 * P2


class WaypointFollower:
    """Simple P‐controller steering & proportional speed to follow a single waypoint."""
    def __init__(self, k_v=1.0, k_α=2.0, max_δ=np.radians(30)):
        self.k_v = k_v        # speed gain
        self.k_α = k_α        # heading gain
        self.max_δ = max_δ    # steering limit

    def control(self, vehicle, wx, wy):
        dx = wx - vehicle.x
        dy = wy - vehicle.y
        ρ = np.hypot(dx, dy)
        # desired heading
        ψ = np.arctan2(dy, dx)
        # heading error normalized to [-π,π]
        α = (ψ - vehicle.theta + np.pi) % (2*np.pi) - np.pi
        # steering = k_α * α, clamped
        δ = np.clip(self.k_α * α, -self.max_δ, self.max_δ)
        # speed = k_v * distance (clamped if desired)
        v = self.k_v * ρ
        return v, δ, ρ


class ParkingController:
    """Follows the dense waypoint list with obstacle checking."""
    def __init__(self, vehicle, obstacles,
                 sensor_angles=None, sensor_range=2.0,
                 dt=0.1, tol=0.1):
        self.vehicle = vehicle
        self.obstacles = obstacles
        self.dt = dt
        self.tol = tol  # waypoint tolerance
        self.sensor = UltrasonicSensor(
            angles=sensor_angles or [0, np.pi/4, -np.pi/4],
            max_range=sensor_range
        )
        self.follower = WaypointFollower()
        self.history = []
        self.controls = []

    def run(self, spot, mode='perpendicular'):
        # choose and build dense path
        if mode == 'perpendicular':
            path = generate_perpendicular_parking_path(spot)
        else:
            path = generate_parallel_parking_path(spot)

        idx = 0
        max_steps = 1000

        for _ in range(max_steps):
            # obstacle proximity stop
            if min(self.sensor.sense(self.vehicle, self.obstacles).values()) < 0.3:
                break

            # if we've reached final waypoint, we're done
            if idx >= len(path):
                break

            wx, wy = path[idx]
            v, δ, ρ = self.follower.control(self.vehicle, wx, wy)

            # if close enough to this waypoint, advance
            if ρ < self.tol:
                idx += 1
                continue

            # apply control
            self.vehicle.update(v, δ, self.dt)
            self.history.append((self.vehicle.x, self.vehicle.y, self.vehicle.theta))
            self.controls.append((v, δ))

        return np.array(self.history), np.array(self.controls)
