import sys
sys.path.append('/mnt/data')

import matplotlib.pyplot as plt
import numpy as np
from controller import Vehicle, AdaptiveParkingController
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as transforms

# Define multiple parking scenarios
scenarios = [
    {
        'name': 'Perpendicular: clear',
        'mode': 'perpendicular',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -8.0, 'y': 0.0, 'theta': 90.0},
        'obstacles': []
    },
    {
        'name': 'Perpendicular: single obstacle',
        'mode': 'perpendicular',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -8.0, 'y': 1.0, 'theta': np.pi/6},
        'obstacles': [(-4.0, 0.5, 0.5)]
    },
    {
        'name': 'Perpendicular: multiple obstacles',
        'mode': 'perpendicular',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -8.0, 'y': -1.0, 'theta': -np.pi/6},
        'obstacles': [(-6.0, 0.5, 0.5), (-5.0, -0.5, 0.4), (-3.0, 1.0, 0.3)]
    },
    {
        'name': 'Parallel: clear',
        'mode': 'parallel',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': 0.0, 'y': 8.0, 'theta': -np.pi/2},
        'obstacles': []
    },
    {
        'name': 'Parallel: obstacle',
        'mode': 'parallel',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': 1.0, 'y': 8.0, 'theta': -np.pi/2},
        'obstacles': [(0.5, 4.0, 0.5)]
    },
    {
        'name': 'Parallel: multiple obstacles',
        'mode': 'parallel',
        'spot': {'x': 0.0, 'y': 0.0},
        'start': {'x': -1.0, 'y': 8.0, 'theta': -np.pi/2},
        'obstacles': [(-0.5, 5.0, 0.4), (0.5, 3.0, 0.3), (0.0, 6.0, 0.5)]
    }
]

for sc in scenarios:
    print(f"\n=== Scenario: {sc['name']} ===")

    # 1) Initialize vehicle and adaptive parking controller
    veh = Vehicle(**sc['start'])
    apc = AdaptiveParkingController(
        vehicle=veh,
        obstacles=sc['obstacles'],
        dt=0.1,
        v_max=1.0,
        a_max=0.5,
        K_acc=2.0,
        tol=0.1
    )

    # 2) Run autonomous parking
    history, controls = apc.run(sc['spot'], mode=sc['mode'])
    print(f"Generated {len(history)} steps")

    if len(history) == 0:
        print("No trajectory — controller may have halted early.")
        continue

    # 3) Plot setup
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(sc['name'])
    ax.set_aspect('equal', 'box')

    # 4) Draw obstacles as blocks
    for cx, cy, r in sc['obstacles']:
        ax.add_patch(Rectangle(
            (cx - r, cy - r), 2*r, 2*r,
            facecolor='lightcoral', edgecolor='darkred', linewidth=2
        ))

    # 5) Draw parking spot as green box (1×1 m)
    gs = 1.0
    gx, gy = sc['spot']['x'] - gs/2, sc['spot']['y'] - gs/2
    ax.add_patch(Rectangle(
        (gx, gy), gs, gs,
        facecolor='green', alpha=0.3, edgecolor='darkgreen', linewidth=2
    ))

    # 6) Center and zoom axes around approach region
    ap = 8.0  # max start distance
    pad = 1.0
    cx, cy = sc['spot']['x'], sc['spot']['y']
    if sc['mode'] == 'perpendicular':
        ax.set_xlim(cx - ap - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + ap + pad)
    else:
        ax.set_xlim(cx - pad, cx + ap + pad)
        ax.set_ylim(cy - ap - pad, cy + pad)

    # 7) Vehicle patch (bigger for visibility)
    length, width = 1.0, 0.6
    vehicle_patch = Rectangle(
        (-length/2, -width/2), length, width,
        facecolor='blue', edgecolor='navy', linewidth=2
    )
    ax.add_patch(vehicle_patch)

    # 8) Trail line (orange)
    trail, = ax.plot([], [], '-', lw=3, color='orange')

    # 9) Animation update
    def update(i):
        x, y, th = history[i]
        trail.set_data(history[:i+1,0], history[:i+1,1])
        trans = transforms.Affine2D().rotate(th).translate(x, y) + ax.transData
        vehicle_patch.set_transform(trans)
        return vehicle_patch, trail

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=50, blit=True)

    plt.show()
    print(f"Displayed animation for {sc['name']}")
