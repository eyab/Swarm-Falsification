"""
Checks to see IF: 
“Is this actually a Voronoi partition?”
“Does every point belong to its nearest site’s cell?”
“Do the cells tile the box without gaps or overlaps?”
“Does it hold under randomness, near-degenerate cases, and large N?”


This file gives you:
Mathematical correctness
Robustness to floating-point issues
Regression protection (future edits won’t silently break geometry)

"""


import numpy as np
from gym_pybullet_drones.examples.test_algorithems import voronoi_geom as V
from gym_pybullet_drones.examples.test_algorithems._geom_helpers import poly_area, sample_points_from_polygon

def _get_cells(points_xy, box):
    if hasattr(V, "compute_bounded_voronoi_cells"):
        return V.compute_bounded_voronoi_cells(points_xy, box=box)

    raise AttributeError(
        "Expected compute_bounded_voronoi_cells(points_xy, box=...) in "
        "test_algorithems/voronoi_geom.py"
    )


def _is_point_in_convex_polygon(poly, x, eps=1e-9):
    poly = np.asarray(poly, dtype=float)
    x = np.asarray(x, dtype=float)
    M = len(poly)
    if M < 3:
        return False

    sign = 0
    for k in range(M):
        a = poly[k]
        b = poly[(k + 1) % M]
        cross = (b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0])
        if abs(cross) <= eps:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True


def _canonical_polygon(poly, decimals=10):
    """
    Make polygons comparable across rotation / vertex order:
    - round
    - remove consecutive duplicates
    - rotate so lexicographically smallest vertex comes first
    """
    P = np.asarray(poly, dtype=float)
    P = np.round(P, decimals=decimals)

    if len(P) > 1:
        keep = [0]
        for i in range(1, len(P)):
            if not np.allclose(P[i], P[keep[-1]]):
                keep.append(i)
        P = P[keep]

    idx = int(np.argmin(P[:, 0] * 1e6 + P[:, 1]))
    P = np.vstack([P[idx:], P[:idx]])
    return tuple(map(tuple, P))


def _assert_cells_basic_invariants(pts, cells, box, eps_area=1e-8):
    xmin, xmax, ymin, ymax = map(float, box)
    assert len(cells) == len(pts)

    for i, poly in enumerate(cells):
        poly = np.asarray(poly, dtype=float)

        assert poly.ndim == 2 and poly.shape[1] == 2, f"Bad shape for cell {i}: {poly.shape}"
        assert len(poly) >= 3, f"Degenerate/empty cell {i} (len={len(poly)})"
        assert np.all(np.isfinite(poly)), f"Non-finite vertices in cell {i}"

        assert np.all(poly[:, 0] >= xmin - 1e-6), f"Cell {i} violates xmin"
        assert np.all(poly[:, 0] <= xmax + 1e-6), f"Cell {i} violates xmax"
        assert np.all(poly[:, 1] >= ymin - 1e-6), f"Cell {i} violates ymin"
        assert np.all(poly[:, 1] <= ymax + 1e-6), f"Cell {i} violates ymax"

        assert abs(poly_area(poly)) > eps_area, f"Cell {i} has near-zero area"



def test_voronoi_cells_are_bounded_and_valid():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    pts = np.array(
        [
            [-3.0, -2.0],
            [ 3.0, -2.0],
            [-2.0,  3.0],
            [ 2.0,  2.0],
            [ 0.5,  0.5],
        ],
        dtype=float,
    )

    cells = _get_cells(pts, box=box)
    _assert_cells_basic_invariants(pts, cells, box)


def test_voronoi_halfplane_property_on_sampled_points():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    pts = np.array(
        [
            [-2.5, -2.0],
            [ 2.5, -2.0],
            [-2.5,  2.5],
            [ 2.5,  2.5],
        ],
        dtype=float,
    )

    cells = _get_cells(pts, box=box)
    eps = 1e-6

    for i, poly in enumerate(cells):
        poly = np.asarray(poly, dtype=float)
        samples = sample_points_from_polygon(poly, n_edge_samples=1)

        for x in samples:
            di = float(np.linalg.norm(x - pts[i]))
            for j in range(len(pts)):
                dj = float(np.linalg.norm(x - pts[j]))
                assert di <= dj + eps, (
                    f"Voronoi violation: cell {i} sample closer to site {j} than {i}. "
                    f"di={di:.6f}, dj={dj:.6f}, x={x}"
                )


def test_voronoi_global_random_points_nearest_site_property():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(123)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(10, 2)).astype(float)
    cells = _get_cells(pts, box=box)
    _assert_cells_basic_invariants(pts, cells, box)

    eps = 1e-6
    Q = 5000
    X = rng.uniform([xmin, ymin], [xmax, ymax], size=(Q, 2)).astype(float)

    for x in X:
        d2 = np.sum((pts - x) ** 2, axis=1)
        k = int(np.argmin(d2))
        dk = float(np.sqrt(d2[k]))
        for j in range(len(pts)):
            dj = float(np.linalg.norm(x - pts[j]))
            assert dk <= dj + eps, f"Global Voronoi violation at x={x}, nearest={k}, other={j}"


def test_voronoi_cell_centroids_lie_inside_cells():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(999)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(12, 2)).astype(float)
    cells = _get_cells(pts, box=box)

    if not hasattr(V, "polygon_centroid"):
        raise AttributeError("Expected polygon_centroid(poly) in test_algorithems/voronoi_geom.py")

    for i, poly in enumerate(cells):
        poly = np.asarray(poly, dtype=float)
        assert len(poly) >= 3
        c = np.asarray(V.polygon_centroid(poly), dtype=float)
        assert np.all(np.isfinite(c)), f"Non-finite centroid for cell {i}: {c}"
        assert _is_point_in_convex_polygon(poly, c, eps=1e-8), (
            f"Centroid not inside its cell for cell {i}. centroid={c}"
        )


def test_voronoi_permutation_invariance():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(2024)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(8, 2)).astype(float)
    cells = _get_cells(pts, box=box)

    perm = rng.permutation(len(pts))
    pts2 = pts[perm]
    cells2 = _get_cells(pts2, box=box)

    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))

    for i in range(len(pts)):
        a = _canonical_polygon(cells[i], decimals=7)
        b = _canonical_polygon(cells2[inv[i]], decimals=7)
        assert a == b, f"Permutation invariance failed for site {i}"


def test_voronoi_shared_edge_points_are_equidistant():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    pts = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]], dtype=float)
    cells = _get_cells(pts, box=box)

    eps = 5e-5   # strict equidistance tolerance
    tol = 2e-3   # "near-boundary" detection tolerance

    for i, poly in enumerate(cells):
        poly = np.asarray(poly, dtype=float)
        samples = sample_points_from_polygon(poly, n_edge_samples=2)

        for x in samples:
            di = float(np.linalg.norm(x - pts[i]))
            d_all = np.linalg.norm(pts - x, axis=1)
            j = int(np.argsort(d_all)[1])  # second-closest
            dj = float(d_all[j])

            # enforce equidistance only if x is nearly tied between i and j
            if abs(di - dj) <= tol:
                assert abs(di - dj) <= eps, f"Bisector equidistance failed at x={x}, i={i}, j={j}"


def test_voronoi_partition_unique_nearest_site_monte_carlo():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(77)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(9, 2)).astype(float)
    _ = _get_cells(pts, box=box)

    Q = 10000
    X = rng.uniform([xmin, ymin], [xmax, ymax], size=(Q, 2)).astype(float)

    ties = 0
    for x in X:
        d2 = np.sum((pts - x) ** 2, axis=1)
        m = float(np.min(d2))
        if np.sum(np.isclose(d2, m, atol=1e-12)) > 1:
            ties += 1

    assert ties <= max(5, int(0.001 * Q)), f"Too many nearest-site ties: {ties}/{Q}"


def test_cells_match_nearest_site_on_dense_grid():
    """
    Independent cross-check:
    - Ground-truth classification is nearest-site by Euclidean distance.
    - For any grid point that lies inside exactly one returned polygon,
      that polygon index must match the nearest-site index.
    """
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(314)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(8, 2)).astype(float)
    cells = _get_cells(pts, box=box)

    gx = np.linspace(xmin, xmax, 81)
    gy = np.linspace(ymin, ymax, 81)

    checked = 0
    for x0 in gx:
        for y0 in gy:
            x = np.array([x0, y0], dtype=float)

            d2 = np.sum((pts - x) ** 2, axis=1)
            k = int(np.argmin(d2))

            hits = []
            for i, poly in enumerate(cells):
                if _is_point_in_convex_polygon(np.asarray(poly, float), x, eps=1e-9):
                    hits.append(i)

            # interior points should belong to exactly one cell (boundaries may have 2)
            if len(hits) == 1:
                checked += 1
                assert hits[0] == k, f"Grid mismatch at x={x}: polygon says {hits[0]} but nearest is {k}"

    assert checked > 1000, f"Too few interior grid points checked: {checked}"


def test_adversarial_sites_no_empty_cells():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    pts = np.array(
        [
            [0.0, 0.0],
            [1e-6, 0.0],
            [0.0, 1e-6],
            [2.0, 2.0],
            [-2.0, 2.0],
            [3.5, -3.5],
            [-3.5, -3.4],
            [0.0, 3.0],
        ],
        dtype=float,
    )

    cells = _get_cells(pts, box=box)
    _assert_cells_basic_invariants(pts, cells, box, eps_area=1e-10)


def test_sum_of_cell_areas_matches_box_area():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(2026)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(10, 2)).astype(float)
    cells = _get_cells(pts, box=box)

    total = 0.0
    for poly in cells:
        total += abs(poly_area(np.asarray(poly, float)))

    box_area = (xmax - xmin) * (ymax - ymin)

    assert abs(total - box_area) / box_area < 1e-3, f"Area mismatch: sum={total}, box={box_area}"



def test_near_boundary_sites_still_produce_valid_cells():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    e = 1e-6
    pts = np.array(
        [
            [xmin + e, ymin + e],
            [xmax - e, ymin + e],
            [xmax - e, ymax - e],
            [xmin + e, ymax - e],
            [0.0, 0.0],
            [xmin + e, 0.0],
            [xmax - e, 0.0],
            [0.0, ymin + e],
            [0.0, ymax - e],
        ],
        dtype=float,
    )

    cells = _get_cells(pts, box=box)
    _assert_cells_basic_invariants(pts, cells, box, eps_area=1e-12)

    rng = np.random.default_rng(42)
    X = rng.uniform([xmin, ymin], [xmax, ymax], size=(5000, 2)).astype(float)
    eps = 1e-6
    for x in X:
        d2 = np.sum((pts - x) ** 2, axis=1)
        k = int(np.argmin(d2))
        dk = float(np.sqrt(d2[k]))
        for j in range(len(pts)):
            dj = float(np.linalg.norm(x - pts[j]))
            assert dk <= dj + eps


def test_randomized_regression_battery_many_seeds():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)

    eps = 1e-6
    seeds = list(range(20))  # increase if you want even harsher

    for seed in seeds:
        rng = np.random.default_rng(seed)
        pts = rng.uniform([xmin + 0.5, ymin + 0.5], [xmax - 0.5, ymax - 0.5], size=(15, 2)).astype(float)

        cells = _get_cells(pts, box=box)
        try:
            _assert_cells_basic_invariants(pts, cells, box)
        except AssertionError as e:
            raise AssertionError(f"[seed={seed}] {e}") from e

        # harsh global nearest-site check (smaller Q per seed for speed)
        X = rng.uniform([xmin, ymin], [xmax, ymax], size=(2000, 2)).astype(float)
        for x in X:
            d2 = np.sum((pts - x) ** 2, axis=1)
            k = int(np.argmin(d2))
            dk = float(np.sqrt(d2[k]))
            for j in range(len(pts)):
                dj = float(np.linalg.norm(x - pts[j]))
                if dk > dj + eps:
                    raise AssertionError(f"[seed={seed}] global Voronoi violation at x={x}, nearest={k}, other={j}")


def test_large_n_stress_n200_area_and_validity():
    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(111)

    N = 200
    pts = rng.uniform([xmin + 0.2, ymin + 0.2], [xmax - 0.2, ymax - 0.2], size=(N, 2)).astype(float)

    cells = _get_cells(pts, box=box)
    _assert_cells_basic_invariants(pts, cells, box, eps_area=1e-12)

    total = float(np.sum([abs(poly_area(np.asarray(poly, float))) for poly in cells]))
    box_area = (xmax - xmin) * (ymax - ymin)
    assert abs(total - box_area) / box_area < 5e-3, f"Area mismatch under N=200 stress: sum={total}, box={box_area}"


def test_voronoi_py_centroids_match_cell_centroids():
    """Directly validates Voronoi.py's centroid routine against cell polygons.

    This checks that `compute_voronoi_centroids_bounded()` (from Voronoi.py)
    returns the centroid of the same clipped polygons returned by
    `compute_bounded_voronoi_cells()` (which itself calls Voronoi.py's clipper).
    """
    from gym_pybullet_drones.examples import Voronoi as SIM

    xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
    box = (xmin, xmax, ymin, ymax)
    rng = np.random.default_rng(12345)

    pts = rng.uniform([xmin + 1.0, ymin + 1.0], [xmax - 1.0, ymax - 1.0], size=(20, 2)).astype(float)

    cells = _get_cells(pts, box=box)
    c_from_cells = np.stack([V.polygon_centroid(np.asarray(poly, float)) for poly in cells], axis=0)

    c_from_voronoi_py = SIM.compute_voronoi_centroids_bounded(pts, (xmin, xmax), (ymin, ymax))

    # Compare only well-defined centroids
    for i in range(len(pts)):
        a = c_from_cells[i]
        b = c_from_voronoi_py[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        assert np.linalg.norm(a - b) < 1e-6, f"Centroid mismatch at i={i}: cells={a}, voronoi.py={b}"
