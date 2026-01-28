"""Geometry helpers for Voronoi tests.

IMPORTANT: This module is intentionally a *thin wrapper* around the real math
implemented in `gym_pybullet_drones.examples.Voronoi` (the main simulation file).

That way, the pytest suite validates the exact clipping / half-plane logic used
by Voronoi.py, without duplicating the algorithm here.

"""

import numpy as np

# Import the real implementation used by the simulator.
from gym_pybullet_drones.examples import Voronoi as _V


def compute_bounded_voronoi_cells(points_xy, box):
    """Return bounded Voronoi cells (convex polygons) for sites in a rectangle.

    Parameters
    ----------
    points_xy : (N,2) array-like
        Voronoi sites.
    box : tuple (xmin, xmax, ymin, ymax)
        Axis-aligned bounding rectangle.

    Returns
    -------
    cells : list[np.ndarray]
        List of polygons (each M_i x 2), in the same order as points.
    """
    pts = np.asarray(points_xy, dtype=float)
    xmin, xmax, ymin, ymax = map(float, box)

    rect = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=float,
    )

    cells = []
    for i in range(len(pts)):
        poly = rect.copy()
        pi = pts[i]

        for j in range(len(pts)):
            if j == i:
                continue
            pj = pts[j]
            d = pj - pi
            # Guard against coincident sites (no separating bisector).
            if float(np.dot(d, d)) < 1e-24:
                continue

            n = d
            c = 0.5 * (float(np.dot(pj, pj)) - float(np.dot(pi, pi)))

            # Use the SAME half-plane clipper as Voronoi.py
            poly = _V.clip_polygon_halfplane(poly, n, c)

            if poly is None or len(poly) < 3:
                break

        cells.append(np.asarray(poly, dtype=float))
    return cells


def polygon_centroid(poly):
    """Centroid of a simple polygon.

    Uses Voronoi.py's area+centroid routine when available.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return np.array([np.nan, np.nan], dtype=float)

    try:
        _, ctr = _V.poly_area_centroid(poly)
        if ctr is None or (not np.all(np.isfinite(ctr))):
            return np.mean(poly, axis=0)
        return np.asarray(ctr, dtype=float)
    except Exception:
        # Fallback: standard polygon centroid formula
        x = poly[:, 0]
        y = poly[:, 1]
        a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if abs(a) < 1e-12:
            return np.mean(poly, axis=0)

        cx = (1.0 / (6.0 * a)) * np.sum(
            (x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
        )
        cy = (1.0 / (6.0 * a)) * np.sum(
            (y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
        )
        return np.array([cx, cy], dtype=float)
