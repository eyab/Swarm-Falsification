# tests/test_leader_follower_correctness.py
#
# PyTest correctness tests for LeaderFollower.py
# (same style/level as the Voronoi correctness tests we wrote)
#
# What these tests check:
# 1) ORCA: does not leave an imminent collision in a simple head-on scenario
# 2) Boundary repulsion: pushes you back inside when near each boundary
# 3) LIDAR avoidance: repulsion vector points away from closest obstacle hit
# 4) Rescue ring: rescuer goes to victim; supporters get unique ring slots at ~ring radius
# 5) Leader–Follower goals: leader waypoint stays inside bounds; followers track offsets; goals are clipped
# 6) Spawn: inter-drone min distance + clearance from obstacles and victim
#
# IMPORTANT:
# - These tests assume your file is named: LeaderFollower.py
# - And that it exposes these functions:
#   orca_step_2d, boundary_repulsion,
#   compute_lidar_avoidance_offset, min_lidar_distance, closest_lidar_hit_vector,
#   rescue_update_goals_ring, leader_follower_compute_goals,
#   sample_safe_initial_xy
#
# If your module name differs, change the import at top.

import numpy as np
import pytest

from gym_pybullet_drones.examples.LeaderFollower import (
    orca_step_2d,
    boundary_repulsion,
    compute_lidar_avoidance_offset,
    min_lidar_distance,
    closest_lidar_hit_vector,
    rescue_update_goals_ring,
    leader_follower_compute_goals,
    sample_safe_initial_xy,
)

# -----------------------------
# Helpers (pure test-side)
# -----------------------------
def will_collide_within_horizon(p_i, v_i, p_j, v_j, radius, T, eps=1e-9):
    """
    Continuous-time disc collision check for two agents with constant velocities.

    We treat "collision" as STRICT overlap:
        exists t in [0, T) such that ||(p_i-p_j) + (v_i-v_j) t|| < 2r

    Notes:
    - We intentionally exclude t == T to avoid brittle "touching exactly at the horizon" failures.
    - We use strict < (not <=) for overlap; touching is not overlap.
    """
    p_i = np.asarray(p_i, dtype=float)
    p_j = np.asarray(p_j, dtype=float)
    v_i = np.asarray(v_i, dtype=float)
    v_j = np.asarray(v_j, dtype=float)

    dp = p_i - p_j
    dv = v_i - v_j
    R = 2.0 * float(radius)

    a = float(np.dot(dv, dv))
    b = 2.0 * float(np.dot(dp, dv))
    c = float(np.dot(dp, dp)) - R * R

    # Relative speed ~0: collide only if already strictly overlapping
    if a < 1e-12:
        return c < 0.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Solutions where distance == R occur on [min(t1,t2), max(t1,t2)].
    # Strict overlap happens strictly between the roots (when c>0 etc).
    t_enter = min(t1, t2)
    t_exit = max(t1, t2)

    # Check intersection with [0, T) (exclude T with eps margin)
    lo = max(0.0, t_enter)
    hi = min(T - eps, t_exit)

    # If there's any time interval with lo < hi, then there exists t in [0, T)
    # where distance <= R. Because we want STRICT overlap (< R), we keep it strict:
    return lo < hi


# ============================================================
# 1) ORCA correctness
# ============================================================
def test_orca_avoids_head_on_collision_basic():
    """
    Two agents head-on with equal and opposite pref velocities.
    After ORCA, they should not be on a collision course (strict overlap) within horizon T.
    """
    dt = 0.1
    T = 4.0
    r = 0.45
    neighbor_dist = 5.0
    max_speed = 2.0

    pos = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    vel = np.zeros((2, 2), dtype=float)
    pref = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=float)

    new_vel = orca_step_2d(
        positions=pos,
        velocities=vel,
        pref_velocities=pref,
        agent_radius=r,
        neighbor_dist=neighbor_dist,
        time_horizon=T,
        max_speed=max_speed,
        dt=dt,
    )

    assert np.all(np.isfinite(new_vel))

    # Strong guard: it should modify the risky symmetric preferences
    assert np.linalg.norm(new_vel - pref) > 1e-6

    # Core property: should avoid strict overlap within [0, T)
    assert not will_collide_within_horizon(pos[0], new_vel[0], pos[1], new_vel[1], r, T)


# ============================================================
# 2) Boundary repulsion correctness
# ============================================================
@pytest.mark.parametrize(
    "pt, expected_sign",
    [
        (np.array([-5.9, 0.0]), +1),  # near left: push +x
        (np.array([+5.9, 0.0]), -1),  # near right: push -x
    ],
)
def test_boundary_repulsion_x_direction(pt, expected_sign):
    x_range = (-6.0, 6.0)
    y_range = (-6.0, 6.0)
    v = boundary_repulsion(pt, x_range, y_range, margin=1.2, gain=0.8, cap=2.0)
    assert np.isfinite(v).all()
    assert v[0] * expected_sign > 0.0


@pytest.mark.parametrize(
    "pt, expected_sign",
    [
        (np.array([0.0, -5.9]), +1),  # near bottom: push +y
        (np.array([0.0, +5.9]), -1),  # near top: push -y
    ],
)
def test_boundary_repulsion_y_direction(pt, expected_sign):
    x_range = (-6.0, 6.0)
    y_range = (-6.0, 6.0)
    v = boundary_repulsion(pt, x_range, y_range, margin=1.2, gain=0.8, cap=2.0)
    assert np.isfinite(v).all()
    assert v[1] * expected_sign > 0.0


# ============================================================
# 3) LIDAR avoidance correctness
# ============================================================
def test_lidar_avoidance_points_away_from_hit():
    """
    If a hit is to the right of drone, the avoidance offset should point left (negative x).
    """
    drone_xy = np.array([[0.0, 0.0]], dtype=float)
    hits = [[np.array([1.0, 0.0], dtype=float)]]

    v = compute_lidar_avoidance_offset(
        drone_xy=drone_xy,
        idx=0,
        lidar_hits_per_drone=hits,
        influence_radius=2.5,
        gain=0.5,
        cap=6.0,
    )
    assert np.isfinite(v).all()
    assert v[0] < 0.0  # push left


def test_min_lidar_distance():
    xy = np.array([0.0, 0.0])
    hits = [np.array([2.0, 0.0]), np.array([0.5, 0.0])]
    d = min_lidar_distance(xy, hits)
    assert np.isclose(d, 0.5, atol=1e-9)


def test_closest_lidar_hit_vector_unit_direction():
    xy = np.array([0.0, 0.0])
    hits = [np.array([1.0, 0.0]), np.array([3.0, 0.0])]
    u = closest_lidar_hit_vector(xy, hits)
    # closest hit is at +x, so vector away is -x
    assert np.allclose(u, np.array([-1.0, 0.0]), atol=1e-6)


# ============================================================
# 4) Rescue ring correctness
# ============================================================
def test_rescue_ring_assignments_unique_and_on_ring():
    N = 5
    goals = np.zeros((N, 2))
    rescuer = 2
    victim_xy = np.array([1.0, -2.0], dtype=float)
    R = 2.8
    phase = 25.0

    out = rescue_update_goals_ring(
        goals_xy=goals,
        found_id=rescuer,
        victim_xy=victim_xy,
        num_drones=N,
        ring_radius=R,
        phase_deg=phase,
    )
    assert out.shape == (N, 2)

    # rescuer goes to victim exactly
    assert np.allclose(out[rescuer], victim_xy, atol=1e-9)

    # supporters are around ring, distinct positions
    supporters = [i for i in range(N) if i != rescuer]
    dists = [np.linalg.norm(out[i] - victim_xy) for i in supporters]
    for d in dists:
        assert np.isclose(d, R, atol=1e-6)

    uniq = {tuple(np.round(out[i], 6)) for i in supporters}
    assert len(uniq) == len(supporters)


# ============================================================
# 5) Leader–Follower goal correctness
# ============================================================
def test_leader_follower_goals_respect_offsets_and_bounds():
    rng = np.random.default_rng(0)
    area_x = (-6.0, 6.0)
    area_y = (-6.0, 6.0)

    N = 5
    drone_xy = np.array(
        [
            [0.0, 0.0],  # leader
            [-1.0, 0.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [-2.0, 0.0],
        ],
        dtype=float,
    )
    goals = np.zeros((N, 2), dtype=float)

    lf_cfg = {
        "leader_idx": 0,
        "leader_waypoint_reached_m": 0.40,
        "leader_waypoint_margin": 1.2,
        "offsets_xy": {
            1: np.array([-1.2, 0.0], dtype=float),
            2: np.array([-1.2, 1.0], dtype=float),
            3: np.array([-1.2, -1.0], dtype=float),
            4: np.array([-2.2, 0.0], dtype=float),
        },
    }
    leader_state = {"leader_wp": None}

    out = leader_follower_compute_goals(
        drone_xy=drone_xy,
        goals_xy=goals,
        rng=rng,
        area_x=area_x,
        area_y=area_y,
        lf_cfg=lf_cfg,
        leader_state=leader_state,
    )
    assert out.shape == (N, 2)

    wp = out[0]
    assert (area_x[0] + 0.3) <= wp[0] <= (area_x[1] - 0.3)
    assert (area_y[0] + 0.3) <= wp[1] <= (area_y[1] - 0.3)

    leader_pos = drone_xy[0]
    for j in [1, 2, 3, 4]:
        expected = leader_pos + lf_cfg["offsets_xy"][j]
        assert np.allclose(out[j], expected, atol=1e-6)


def test_leader_waypoint_changes_when_reached():
    rng = np.random.default_rng(0)
    area_x = (-6.0, 6.0)
    area_y = (-6.0, 6.0)

    N = 2
    drone_xy = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    goals = np.zeros((N, 2), dtype=float)

    lf_cfg = {
        "leader_idx": 0,
        "leader_waypoint_reached_m": 100.0,  # huge => "reached" immediately
        "leader_waypoint_margin": 1.2,
        "offsets_xy": {1: np.array([-1.0, 0.0])},
    }
    leader_state = {"leader_wp": None}

    out1 = leader_follower_compute_goals(drone_xy, goals, rng, area_x, area_y, lf_cfg, leader_state)
    wp1 = out1[0].copy()

    out2 = leader_follower_compute_goals(drone_xy, goals, rng, area_x, area_y, lf_cfg, leader_state)
    wp2 = out2[0].copy()

    assert not np.allclose(wp1, wp2)


# ============================================================
# 6) Spawn correctness (same style as Voronoi)
# ============================================================
def test_spawn_safe_initial_xy_respects_clearances():
    rng = np.random.default_rng(123)
    area_x = (-6.0, 6.0)
    area_y = (-6.0, 6.0)

    spawn_cfg = {
        "boundary_margin": 1.0,
        "min_interdrone_dist": 1.0,
        "max_tries": 4000,
    }
    obstacle_centers = [np.array([0.0, 2.0]), np.array([-2.0, -1.0])]
    obstacle_clear = 1.2
    victim_centers = [np.array([1.0, -2.0])]
    victim_clear = 1.0

    N = 5
    pts = sample_safe_initial_xy(
        num_drones=N,
        rng=rng,
        area_x=area_x,
        area_y=area_y,
        spawn_cfg=spawn_cfg,
        obstacle_centers=obstacle_centers,
        obstacle_clear=obstacle_clear,
        victim_centers=victim_centers,
        victim_clear=victim_clear,
    )

    assert pts.shape == (N, 2)
    assert np.isfinite(pts).all()

    # inter-drone spacing
    for i in range(N):
        for j in range(i + 1, N):
            assert np.linalg.norm(pts[i] - pts[j]) >= spawn_cfg["min_interdrone_dist"] - 1e-9

    # obstacle clearance
    for pxy in pts:
        for oc in obstacle_centers:
            assert np.linalg.norm(pxy - oc) >= obstacle_clear - 1e-9

    # victim clearance
    for pxy in pts:
        for vc in victim_centers:
            assert np.linalg.norm(pxy - vc) >= victim_clear - 1e-9
