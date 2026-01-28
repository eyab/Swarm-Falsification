import numpy as np

def poly_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def sample_points_from_polygon(poly, n_edge_samples=2, edge_samples=None):
    """
    Return a small set of points to test membership:
    - all vertices
    - midpoints of edges
    - a couple of points near "center"
    """
    if edge_samples is not None:
        n_edge_samples = edge_samples

    poly = np.asarray(poly, dtype=float)
    pts = []
    M = len(poly)
    for k in range(M):
        a = poly[k]
        b = poly[(k + 1) % M]
        pts.append(a)
        pts.append(0.5 * (a + b))
        for t in np.linspace(0.25, 0.75, n_edge_samples):
            pts.append((1 - t) * a + t * b)

    c = np.mean(poly, axis=0)
    pts.append(c)
    return np.unique(np.asarray(pts), axis=0)
