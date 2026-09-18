"""Scenery for the 3D viewer: signs, signals, crosswalks, sidewalks, buildings, trees, rails.
Purely visual and deterministic; nothing here feeds the simulation."""

import math

import numpy as np

BUILDING_COLORS = ["#4b5563", "#57534e", "#475569", "#52525b", "#5b5f68", "#4a4f57"]


def _r(v, digits: int = 2) -> float:
    return round(float(v), digits)


def stop_sign(x: float, y: float, heading: float) -> dict:
    """heading is the travel direction of the traffic the sign faces."""
    return {"type": "stop_sign", "x": _r(x), "y": _r(y), "h": _r(heading, 4)}


def signal(x: float, y: float, heading: float, key: str) -> dict:
    return {"type": "signal", "x": _r(x), "y": _r(y), "h": _r(heading, 4), "key": key}


def street_light(x: float, y: float, arm_heading: float) -> dict:
    return {"type": "light", "x": _r(x), "y": _r(y), "h": _r(arm_heading, 4)}


def tree(x: float, y: float, scale: float = 1.0) -> dict:
    return {"type": "tree", "x": _r(x), "y": _r(y), "s": _r(scale)}


def building(x: float, y: float, w: float, length: float, height: float, th: float, color: str) -> dict:
    return {"type": "building", "x": _r(x), "y": _r(y), "w": _r(w), "l": _r(length), "height": _r(height),
            "th": _r(th, 4), "color": color}


def slab(x: float, y: float, w: float, length: float, th: float = 0.0, height: float = 0.15) -> dict:
    """A raised sidewalk block: w along the rotated x axis, length along y."""
    return {"type": "slab", "x": _r(x), "y": _r(y), "w": _r(w), "l": _r(length), "th": _r(th, 4), "height": height}


def crosswalk(x: float, y: float, heading: float, span: float, depth: float = 2.2) -> dict:
    """Zebra crossing centered at (x, y): stripes run along `heading`, spread across `span`."""
    return {"type": "crosswalk", "x": _r(x), "y": _r(y), "h": _r(heading, 4), "span": span, "depth": depth}


def rail(points, kind: str) -> dict:
    return {"type": "rail", "kind": kind, "pts": [[_r(x), _r(y)] for x, y in points]}


def gantry(x: float, y: float, th: float, span: float, label: str) -> dict:
    return {"type": "gantry", "x": _r(x), "y": _r(y), "th": _r(th, 4), "span": _r(span), "label": label}


def _edge(centerline: np.ndarray, offset: float, step: int) -> list:
    """The centerline shifted `offset` to the left (negative for right), thinned to every `step` points."""
    idx = sorted(set(range(0, len(centerline), step)) | {len(centerline) - 1})
    return [(centerline[i, 0] - offset * math.sin(centerline[i, 2]), centerline[i, 1] + offset * math.cos(centerline[i, 2]))
            for i in idx]


def highway_props(centerline, table, offsets, lane_width, seed=0, rails=True, gantry_label="101") -> list:
    left, right = max(offsets) + lane_width / 2, min(offsets) - lane_width / 2
    props = []
    if rails:
        props += [rail(_edge(centerline, right - 3.0, 3), "guardrail"), rail(_edge(centerline, left + 3.0, 3), "barrier")]
    rng = np.random.default_rng(seed)
    s = table[0] + 12.0
    while s < table[-1] - 8.0:
        i = int(np.argmin(np.abs(table - s)))
        x, y, th = centerline[i]
        for side, base in ((-1, right - 7.0), (1, left + 7.0)):
            if rng.random() < 0.7:
                off = base + side * rng.uniform(0.0, 9.0)
                props.append(tree(x - off * math.sin(th), y + off * math.cos(th), rng.uniform(0.8, 1.4)))
        s += rng.uniform(16.0, 30.0)
    if rails:
        i = int(np.argmin(np.abs(table - (table[0] + 0.33 * (table[-1] - table[0])))))
        props.append(gantry(centerline[i, 0], centerline[i, 1], centerline[i, 2], (left - right) + 3.0, gantry_label))
    return props


def box_rect(legs: str, rw: float, half: float) -> tuple:
    """The paved conflict zone (xmin, xmax, ymin, ymax): full size toward legs that exist, road width otherwise."""
    return (-(half if "W" in legs else rw), half if "E" in legs else rw,
            -(half if "S" in legs else rw), half if "N" in legs else rw)


def _slab_rects(legs: str, rw: float, half: float, reach: float) -> list:
    """Sidewalk rectangles (xmin, xmax, ymin, ymax) filling the blocks between the roads, clear of the paved box."""
    left, right, bot, top = box_rect(legs, rw, half)
    rects = [(rw, reach, top, reach), (-reach, -rw, top, reach)] if "N" in legs else [(-reach, reach, top, reach)]
    rects += [(rw, reach, -reach, bot), (-reach, -rw, -reach, bot)] if "S" in legs else [(-reach, reach, -reach, bot)]
    rects += [(right, reach, rw, top), (right, reach, bot, -rw)] if "E" in legs else [(right, reach, bot, top)]
    rects += [(-reach, left, rw, top), (-reach, left, bot, -rw)] if "W" in legs else [(-reach, left, bot, top)]
    return [r for r in rects if r[1] - r[0] > 0.5 and r[3] - r[2] > 0.5]


def urban_props(legs: str, approaches: dict, half: float, road_w: float, reach: float, seed: int = 0,
                signal_keys: dict | None = None) -> list:
    """Street furniture and blocks around an intersection. `approaches` maps a leg name to its
    Approach (travel direction and position helper); `signal_keys` maps a leg to a signal key."""
    rw, rng, props = road_w / 2, np.random.default_rng(seed), []
    for x0, x1, y0, y1 in _slab_rects(legs, rw, half, reach):
        props.append(slab((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        if x1 - x0 >= 26.0 and y1 - y0 >= 26.0:
            for gx in np.arange(x0 + 16.0, x1 - 8.0, 22.0):
                for gy in np.arange(y0 + 16.0, y1 - 8.0, 22.0):
                    props.append(building(gx + rng.uniform(-2, 2), gy + rng.uniform(-2, 2), rng.uniform(11, 16),
                                          rng.uniform(11, 16), rng.uniform(4.0, 10.0), 0.0,
                                          BUILDING_COLORS[int(rng.integers(len(BUILDING_COLORS)))]))
        for near, lo, hi, fixed_x in ((x0, y0, y1, True), (x1, y0, y1, True), (y0, x0, x1, False), (y1, x0, x1, False)):
            if abs(near) not in (rw, half):
                continue
            inset = 2.2 if near in (x0, y0) else -2.2
            for t in np.arange(lo + 6.0, hi - 4.0, 11.0):
                if abs(t) > half + 4.0 and abs(t) < reach - 6.0:
                    props.append(tree(near + inset, t, rng.uniform(0.8, 1.2)) if fixed_x else tree(t, near + inset, rng.uniform(0.8, 1.2)))
    for leg in legs:
        a = approaches[leg]
        px, py = a.position(-9.3, 0.0)
        props.append(crosswalk(px, py, a.heading, road_w))
        sx, sy = a.position(-11.6, rw + 1.3)
        props.append(signal(sx, sy, a.heading, signal_keys[leg]) if signal_keys else stop_sign(sx, sy, a.heading))
    for sx in (-1, 1):
        for sy in (-1, 1):
            props.append(street_light(sx * (rw + 2.4), sy * (rw + 2.4), math.atan2(-sy, -sx)))
    return props


def route_props(path_xy: np.ndarray, road_w: float, seed: int = 0) -> list:
    """Sidewalks, trees and low buildings along a driven route (illustrative scenery)."""
    rng, props = np.random.default_rng(seed), []
    seg = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    head = np.arctan2(np.gradient(path_xy[:, 1]), np.gradient(path_xy[:, 0]))
    s = 2.0
    while s < arc[-1] - 2.0:
        i = int(np.argmin(np.abs(arc - s)))
        x, y, th = path_xy[i, 0], path_xy[i, 1], head[i]
        for side in (-1, 1):
            off = side * (road_w / 2 + 1.6)
            props.append(slab(x - off * math.sin(th), y + off * math.cos(th), 3.4, 3.0, th + math.pi / 2))
        s += 3.0
    s = 6.0
    while s < arc[-1] - 4.0:
        i = int(np.argmin(np.abs(arc - s)))
        x, y, th = path_xy[i, 0], path_xy[i, 1], head[i]
        for side in (-1, 1):
            if rng.random() < 0.55:
                off = side * (road_w / 2 + 3.4)
                props.append(tree(x - off * math.sin(th), y + off * math.cos(th), rng.uniform(0.8, 1.2)))
            if rng.random() < 0.45:
                off = side * (road_w / 2 + 12.0 + rng.uniform(0, 4))
                props.append(building(x - off * math.sin(th), y + off * math.cos(th), rng.uniform(9, 14), rng.uniform(9, 14),
                                      rng.uniform(4, 8), th, BUILDING_COLORS[int(rng.integers(len(BUILDING_COLORS)))]))
        s += rng.uniform(14.0, 26.0)
    return props
