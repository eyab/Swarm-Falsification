import numpy as np

# IMPORTANT: this imports ORCA from *Voronoi.py* (the simulator code you care about)
from gym_pybullet_drones.examples.Voronoi import orca_step_2d


def _will_collide_within_horizon(p_i, v_i, p_j, v_j, radius, T):
    """Continuous-time disc collision check for two agents with constant velocities."""
    dp = p_i - p_j
    dv = v_i - v_j
    R = 2.0 * radius

    a = float(np.dot(dv, dv))
    b = 2.0 * float(np.dot(dp, dv))
    c = float(np.dot(dp, dp)) - R * R

    if a < 1e-12:
        return c < 0.0

    t_star = -b / (2.0 * a)
    t_star = float(np.clip(t_star, 0.0, T))
    d2 = a * t_star * t_star + b * t_star + c
    return d2 < 0.0


def _bruteforce_best_safe_velocity_two_agent(
    p_i, p_j, v_j_fixed, pref_i, radius, T, v_max,
    n_angles=720, n_radii=120,
):
    """Independent best-response search for agent i given agent j fixed."""
    best = None
    best_cost = float("inf")

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(0.0, v_max, n_radii)

    cs = np.cos(angles)
    sn = np.sin(angles)

    for r in radii:
        cand = np.stack([r * cs, r * sn], axis=1)
        for u in cand:
            if _will_collide_within_horizon(p_i, u, p_j, v_j_fixed, radius, T):
                continue
            cost = float(np.linalg.norm(u - pref_i))
            if cost < best_cost:
                best_cost = cost
                best = u.copy()

    return best, best_cost


def test_orca_voronoi_py_matches_bruteforce_best_response_two_agent():
    """
    Checks Voronoi.py's ORCA implementation (orca_step_2d):

    - finite and speed-bounded outputs
    - pairwise safety within time horizon (continuous-time)
    - agent 0 close to independent brute-force best-response, conditioning on agent 1's ORCA velocity
    """
    agent_radius = 0.45
    neighbor_dist = 10.0
    time_horizon = 4.0
    max_speed = 1.5
    dt = 0.05

    pos = np.array([[-2.0, 0.0],
                    [ 2.0, 0.0]], dtype=float)

    vel = np.array([[ 1.2, 0.0],
                    [-1.2, 0.0]], dtype=float)

    pref = vel.copy()

    new_vel = orca_step_2d(
        positions=pos,
        velocities=vel,
        pref_velocities=pref,
        agent_radius=agent_radius,
        neighbor_dist=neighbor_dist,
        time_horizon=time_horizon,
        max_speed=max_speed,
        dt=dt,
    )

    assert np.all(np.isfinite(new_vel))
    assert np.all(np.linalg.norm(new_vel, axis=1) <= max_speed + 1e-9)

    assert not _will_collide_within_horizon(
        pos[0], new_vel[0], pos[1], new_vel[1], agent_radius, time_horizon
    ), "ORCA produced velocities that still collide within time horizon"

    bf_v0, _ = _bruteforce_best_safe_velocity_two_agent(
        p_i=pos[0],
        p_j=pos[1],
        v_j_fixed=new_vel[1],
        pref_i=pref[0],
        radius=agent_radius,
        T=time_horizon,
        v_max=max_speed,
    )
    assert bf_v0 is not None

    dist = float(np.linalg.norm(new_vel[0] - bf_v0))
    assert dist <= 0.25, f"ORCA deviates too much from brute-force best-response: dist={dist:.3f}"
