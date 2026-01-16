import numpy as np
import pytest
from gym_pybullet_drones.examples.Voronoi import (
    compute_voronoi_centroids_bounded,
    orca_step_2d,
    clip_polygon_halfplane,
    poly_area_centroid
)

def test_poly_area_centroid():
    # Square 1x1
    poly = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=float)
    area, centroid = poly_area_centroid(poly)
    assert np.isclose(area, 1.0)
    assert np.allclose(centroid, [0.5, 0.5])

    # Triangle
    poly = np.array([[0,0], [2,0], [0,2]], dtype=float)
    area, centroid = poly_area_centroid(poly)
    assert np.isclose(area, 2.0)
    # Centroid of (0,0), (2,0), (0,2) is (2/3, 2/3)
    assert np.allclose(centroid, [2/3, 2/3])

def test_clip_polygon_halfplane():
    # Square 1x1 clipped by x < 0.5 (n=[1,0], c=0.5) -> keep x <= 0.5?
    # Function is dot(n, pt) <= c
    # If n=[1,0], c=0.5, then x <= 0.5 is kept.
    
    poly = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=float)
    n = np.array([1, 0], dtype=float)
    c = 0.5
    
    clipped = clip_polygon_halfplane(poly, n, c)
    # Result should be rectangle [0,0] to [0.5, 1]
    # Area should be 0.5
    area, center = poly_area_centroid(clipped)
    assert np.isclose(area, 0.5)

def test_compute_voronoi_centroids_bounded():
    # 1 drone in 10x10 box centered at 0
    # Box: -5 to 5
    drone_xy = np.array([[0.0, 0.0]])
    x_range = (-5.0, 5.0)
    y_range = (-5.0, 5.0)
    
    centroids = compute_voronoi_centroids_bounded(drone_xy, x_range, y_range)
    # Should be at 0,0 (center of box)
    assert np.allclose(centroids[0], [0.0, 0.0])
    
    # 2 drones symmetric
    drone_xy = np.array([[-2.0, 0.0], [2.0, 0.0]])
    centroids = compute_voronoi_centroids_bounded(drone_xy, x_range, y_range)
    
    # Left drone centroid should be (-2.5, 0) because left cell is rectangle [-5, 0] x [-5, 5] (width 5, height 10)
    # Center of left rect is -2.5, 0
    assert np.allclose(centroids[0], [-2.5, 0.0])
    # Right drone centroid should be (2.5, 0)
    assert np.allclose(centroids[1], [2.5, 0.0])

def test_orca_step_2d_no_collision():
    # 1 drone moving right, no neighbors
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[1.0, 0.0]])
    pref_velocities = np.array([[1.0, 0.0]])
    agent_radius = 0.5
    neighbor_dist = 5.0
    time_horizon = 2.0
    max_speed = 2.0
    dt = 0.1
    
    new_vels = orca_step_2d(positions, velocities, pref_velocities, agent_radius, neighbor_dist, time_horizon, max_speed, dt)
    
    # Should keep preferred velocity
    assert np.allclose(new_vels, pref_velocities)

def test_orca_step_2d_head_on():
    # 2 drones moving head-on
    positions = np.array([[-2.0, 0.0], [2.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [-1.0, 0.0]]) # Moving towards each other
    pref_velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
    
    agent_radius = 0.5
    neighbor_dist = 10.0 # They see each other
    time_horizon = 5.0
    max_speed = 2.0
    dt = 0.1
    
    new_vels = orca_step_2d(positions, velocities, pref_velocities, agent_radius, neighbor_dist, time_horizon, max_speed, dt)
    
    # They should deviate. E.g. y component should become non-zero or x component reduced.
    # ORCA usually diverts to the right relative to motion.
    
    # Check that they don't collide in velocity space (hard to check exactly without full ORCA logic replication)
    # checking that velocity changed is good enough for a basic test.
    assert not np.allclose(new_vels, pref_velocities)
    
    # Velocities should still respect max speed
    speeds = np.linalg.norm(new_vels, axis=1)
    assert np.all(speeds <= max_speed + 1e-5)
