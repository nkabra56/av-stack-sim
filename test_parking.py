import sys
sys.path.append('/mnt/data')

import matplotlib.pyplot as plt
import numpy as np
from controller import Vehicle, ParkingController
from controller import generate_perpendicular_parking_path, generate_parallel_parking_path
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as transforms

scenarios = [
    {
        'name': 'Perpendicular Parking',
        'mode': 'perpendicular',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -5.0, 'y': 0.0, 'theta': 0.0},
        'obstacles': []
    },
    {
        'name': 'Perpendicular with Obstacles',
        'mode': 'perpendicular',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -6.0, 'y': 0.0, 'theta': 0.0},
        'obstacles': [(-4.0, 0.5, 0.4), (-3.0, -0.5, 0.3), (-2.0, 0.0, 0.5)]
    },
    {
        'name': 'Parallel Parking',
        'mode': 'parallel',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': 0.0, 'y': 5.0, 'theta': -np.pi/2},
        'obstacles': []
    },
    {
        'name': 'Parallel with Obstacles',
        'mode': 'parallel',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': 0.0, 'y': 6.0, 'theta': -np.pi/2},
        'obstacles': [(0.5, 4.0, 0.4), (-0.5, 3.0, 0.3), (0.0, 2.0, 0.5)]
    }
]

for sc in scenarios:
    # compute trajectory
    veh = Vehicle(x=sc['start']['x'], y=sc['start']['y'], theta=sc['start']['theta'])
    pc = ParkingController(vehicle=veh, obstacles=sc['obstacles'], dt=0.1)
    history, _ = pc.run(sc['spot'], mode=sc['mode'])

    # define goal box centered on spot (1m×1m)
    gs = 1.0
    gx = sc['spot']['x'] - gs/2
    gy = sc['spot']['y'] - gs/2

    fig, ax = plt.subplots()
    ax.set_title(sc['name'])
    ax.set_aspect('equal', 'box')

    # draw goal spot
    goal = Rectangle((gx, gy), gs, gs, facecolor='green', alpha=0.3)
    ax.add_patch(goal)

    # draw obstacles as blocks
    for cx, cy, r in sc['obstacles']:
        obs = Rectangle((cx - r, cy - r), 2*r, 2*r,
                        facecolor='lightcoral', edgecolor='darkred', alpha=0.6)
        ax.add_patch(obs)

    # compute axis limits to include history, obstacles, and goal
    xs = history[:, 0].copy()
    ys = history[:, 1].copy()
    # include obstacles extents
    for cx, cy, r in sc['obstacles']:
        xs = np.append(xs, [cx - r, cx + r])
        ys = np.append(ys, [cy - r, cy + r])
    # include goal extents
    xs = np.append(xs, [gx, gx + gs])
    ys = np.append(ys, [gy, gy + gs])

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    pad = 1.0
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # vehicle patch at origin, will be transformed
    length, width = 0.5, 0.3
    vehicle_patch = Rectangle((-length/2, -width/2), length, width,
                              facecolor='blue', edgecolor='none')
    ax.add_patch(vehicle_patch)

    # trail line
    trail, = ax.plot([], [], '-', linewidth=2, color='orange')

    # update function for animation
    def update(i):
        x, y, theta = history[i]
        trail.set_data(history[:i+1, 0], history[:i+1, 1])
        trans = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
        vehicle_patch.set_transform(trans)
        return vehicle_patch, trail

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=100, blit=True)
    plt.show()
