# ============================================================
# Leader–Follower Swarm Formation 
# Run: python LeaderFollower.py --gui True
# ============================================================

import time
import argparse
import numpy as np

import pybullet as p
import pybullet_data

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")

SIM_FREQ = 240
CTRL_FREQ = 48


CONFIG = {
    "area": {"x_min": -12.0, "x_max": 12.0, "y_min": -12.0, "y_max": 12.0},

    "num_drones": 5,
    "leader_idx": 0,

    # Target
    "goal_xy": np.array([8.0, -8.0], dtype=float),
    "leader_stop_radius": 1.2,  

    # Hover mode
    "hover_ring": {
        "enabled": True,
        "ring_r": None,          
        "guard_margin": 0.25,   
    },

    # Altitude
    "drone_z_start": 0.6,
    "altitude": 1.25,

    # Speed: meters per control step
    "max_xy_step": 0.060,

    "kp": 1.20,

    "formation_scale": 1.0,
    "formation_offsets_body": {
        1: np.array([-0.9,  0.0], dtype=float),
        2: np.array([-1.6,  0.8], dtype=float),
        3: np.array([-1.6, -0.8], dtype=float),
        4: np.array([-2.3,  0.0], dtype=float),
    },

    "formation_acquire": {
        "enabled": True,
        "hold_err_m": 0.45,
        "hold_min_sec": 1.0,
        "acquire_timeout_sec": 6.0,
    },

    "fixed_heading": {"enabled": True},

    # ORCA
    "orca": {
        "enabled": True,
        "agent_radius": 0.28,
        "neighbor_dist": 4.5,
        "time_horizon": 3.0,
        "max_speed_mps": 2.0,  # overridden to match exec_max_speed
        "hover_radius_scale": 1.15,  # slightly more conservative in hover
    },

    # Boundary repulsion
    "boundary": {
        "enabled": True,
        "margin": 1.2,
        "gain": 0.9,
        "cap": 2.0,
    },


    "obstacles": {
        "enabled": True,
        "num": 4,
        "seed_offset": 101,          
        "cyl_radius": 0.35,
        "cyl_height": 1.6,
        "rgba": [0.30, 0.22, 0.15, 1.0],
        "spawn_margin": 2.5,
        "keepout_from_goal": 2.8,   
        "keepout_from_spawn": 2.5,   
        "keepout_from_each": 2.2,    #

        # avoidance field (XY)
        "avoid": {
            "enabled": True,
            "influence": 3.0,        
            "gain": 1.25,
            "cap": 2.5,
        },
    },

    # Stability
    "stability": {
        "a_max_mps2": 5.0,
        "vel_lpf_alpha": 0.86,
        "near_ground_z": 0.35,
        "recover_z_boost": 0.8,
    },

    # Tilt scaling
    "safety": {
        "enabled": True,
        "tilt_soft_rad": 0.55,
        "tilt_hard_rad": 0.90,
        "z_boost": 0.60,
    },

    # Arrival latch + snap
    "arrive": {
        "enabled": True,
        "reach_r": 0.35,      # latch hover when leader is within this of standoff_fixed
        "hysteresis": 0.10,
        "stop_r_all": 0.18,   # snap-to-goal radius for ALL drones
        "hover_stop_r_followers": 0.18,
        "hover_damp_followers": 0.85,
    },

    "leader_slow": {
        "enabled": True,
        "err_ref": 1.00,
        "min_scale": 0.20,
        "max_scale": 1.00,
    },

    "visuals": {
        "enabled": True,
        "leader_rgba": [1.0, 0.9, 0.1, 1.0],
        "follower_rgba": [0.3, 0.7, 1.0, 1.0],

        "markers": {"enabled": False},

        "balloons": {
            "enabled": True,
            "radius": 0.16,
            "z_offset": 0.55,
            "alpha": 0.90,
            "update_every": 2,
            "string": {
                "enabled": False,
                "width": 2.0,
            },
        },

        "beacon": {"enabled": False},
    },

    "print_status": {"enabled": True, "every_sec": 2.5},
}


# -----------------------------
# Helpers
# -----------------------------
def vec_norm(v):
    return float(np.linalg.norm(v))

def normalize(v, eps=1e-12):
    n = vec_norm(v)
    return np.zeros_like(v) if n < eps else (v / n)

def cap_vector(v, cap):
    n = vec_norm(v)
    if cap is not None and cap > 0 and n > cap:
        return (v / n) * cap
    return v

def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)

def heading_yaw(from_xy, to_xy):
    d = np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
    if vec_norm(d) < 1e-9:
        return 0.0
    return float(np.arctan2(d[1], d[0]))

def boundary_repulsion(xy, x_range, y_range, margin=1.2, gain=0.8, cap=2.0):
    x, y = float(xy[0]), float(xy[1])
    fx, fy = 0.0, 0.0

    d = x - x_range[0]
    if d < margin:
        fx += gain * (1.0 / max(d, 1e-3) - 1.0 / margin)

    d = x_range[1] - x
    if d < margin:
        fx -= gain * (1.0 / max(d, 1e-3) - 1.0 / margin)

    d = y - y_range[0]
    if d < margin:
        fy += gain * (1.0 / max(d, 1e-3) - 1.0 / margin)

    d = y_range[1] - y
    if d < margin:
        fy -= gain * (1.0 / max(d, 1e-3) - 1.0 / margin)

    return cap_vector(np.array([fx, fy], dtype=float), cap)

def obstacle_repulsion(xy, obstacles, influence=3.0, gain=1.0, cap=2.5):
    if not obstacles:
        return np.zeros(2, dtype=float)

    pxy = np.asarray(xy, dtype=float)
    acc = np.zeros(2, dtype=float)

    for ob in obstacles:
        c = ob["xy"]
        r = float(ob["radius"])
        dvec = pxy - c
        d = float(np.linalg.norm(dvec))
        if d < 1e-9:
            dvec = np.array([1.0, 0.0], dtype=float)
            d = 1e-9

        ds = d - r
        if ds >= influence:
            continue

        w = (1.0 / max(ds, 1e-3) - 1.0 / max(influence, 1e-3))
        acc += gain * w * (dvec / d)

    return cap_vector(acc, cap)

def rand_xy(rng, area_x, area_y, margin=1.0):
    return np.array(
        [rng.uniform(area_x[0] + margin, area_x[1] - margin),
         rng.uniform(area_y[0] + margin, area_y[1] - margin)],
        dtype=float,
    )

def min_pairwise_dist(xy):
    N = xy.shape[0]
    dmin = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dmin = min(dmin, float(np.linalg.norm(xy[i] - xy[j])))
    return dmin


def arrive_clamp_velocity(pos_xy, goal_xy, v_xy, dt, stop_r=0.18, allow_reverse=True):
    e = np.asarray(goal_xy, dtype=float) - np.asarray(pos_xy, dtype=float)
    d = float(np.linalg.norm(e))
    if d <= stop_r:
        return np.zeros(2, dtype=float)

    if (not allow_reverse) and float(np.dot(v_xy, e)) <= 0.0:
        return np.zeros(2, dtype=float)

    vmag = float(np.linalg.norm(v_xy))
    if vmag * dt > d:
        return e / max(dt, 1e-6)

    return v_xy


# -----------------------------
# ORCA (2D)
# -----------------------------
class OrcaLine:
    __slots__ = ("point", "direction")
    def __init__(self, point, direction):
        self.point = np.asarray(point, dtype=float)
        self.direction = np.asarray(direction, dtype=float)

def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def linear_program_1(lines, line_no, radius, opt_vel, direction_opt):
    line = lines[line_no]
    dotp = float(np.dot(line.point, line.direction))
    discriminant = dotp * dotp + radius * radius - float(np.dot(line.point, line.point))
    if discriminant < 0.0:
        return False, None

    sqrt_disc = float(np.sqrt(discriminant))
    t_left = -dotp - sqrt_disc
    t_right = -dotp + sqrt_disc

    for i in range(line_no):
        other = lines[i]
        denom = cross2(line.direction, other.direction)
        numer = cross2(other.direction, line.point - other.point)

        if abs(denom) < 1e-12:
            if numer < 0.0:
                return False, None
            continue

        t = numer / denom
        if denom > 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return False, None

    if direction_opt:
        t = t_right if float(np.dot(opt_vel, line.direction)) > 0.0 else t_left
    else:
        denom = float(np.dot(line.direction, line.direction))
        t = float(np.dot(line.direction, opt_vel - line.point)) / max(denom, 1e-12)
        t = min(max(t, t_left), t_right)

    return True, line.point + t * line.direction

def linear_program_2(lines, radius, opt_vel, direction_opt):
    if direction_opt:
        result = normalize(opt_vel) * radius
    else:
        result = normalize(opt_vel) * radius if vec_norm(opt_vel) > radius else opt_vel.copy()

    for i, line in enumerate(lines):
        if cross2(line.direction, result - line.point) < 0.0:
            success, tmp = linear_program_1(lines, i, radius, opt_vel, direction_opt)
            if not success:
                return result, i
            result = tmp
    return result, len(lines)

def linear_program_3(lines, begin_line, radius, result):
    distance = 0.0
    for i in range(begin_line, len(lines)):
        line = lines[i]
        if cross2(line.direction, result - line.point) < distance:
            proj_lines = []
            for j in range(i):
                other = lines[j]
                denom = cross2(line.direction, other.direction)
                if abs(denom) < 1e-12:
                    continue
                t = cross2(other.direction, line.point - other.point) / denom
                pt = line.point + t * line.direction
                dir_ = normalize(other.direction - line.direction)
                proj_lines.append(OrcaLine(pt, dir_))

            opt_dir = np.array([-line.direction[1], line.direction[0]], dtype=float)
            new_result, fail = linear_program_2(proj_lines, radius, opt_dir, direction_opt=True)
            if fail == len(proj_lines):
                result = new_result
            distance = cross2(line.direction, result - line.point)
    return result

def orca_step_2d(positions, velocities, pref_velocities, agent_radius, neighbor_dist, time_horizon, max_speed, dt):
    N = positions.shape[0]
    new_vels = np.zeros((N, 2), dtype=float)

    inv_time_h = 1.0 / max(time_horizon, 1e-6)
    combined_radius = 2.0 * agent_radius
    combined_radius_sq = combined_radius * combined_radius
    inv_time_step = 1.0 / max(dt, 1e-6)

    for i in range(N):
        p_i = positions[i]
        v_i = velocities[i]
        lines = []

        for j in range(N):
            if j == i:
                continue
            p_j = positions[j]
            v_j = velocities[j]

            rel_pos = p_j - p_i
            dist_sq = float(np.dot(rel_pos, rel_pos))
            if dist_sq > neighbor_dist * neighbor_dist:
                continue

            rel_vel = v_i - v_j

            if dist_sq > combined_radius_sq:
                w = rel_vel - inv_time_h * rel_pos
                w_len_sq = float(np.dot(w, w))
                dot = float(np.dot(w, rel_pos))

                if dot < 0.0 and dot * dot > combined_radius_sq * w_len_sq:
                    w_len = float(np.sqrt(max(w_len_sq, 1e-12)))
                    unit_w = w / max(w_len, 1e-12)
                    direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
                    u = (combined_radius * inv_time_h - w_len) * unit_w
                else:
                    leg = float(np.sqrt(max(dist_sq - combined_radius_sq, 1e-12)))
                    if cross2(rel_pos, w) > 0.0:
                        direction = (
                            np.array([rel_pos[0] * leg - rel_pos[1] * combined_radius,
                                      rel_pos[0] * combined_radius + rel_pos[1] * leg], dtype=float)
                            / max(dist_sq, 1e-12)
                        )
                    else:
                        direction = -(
                            np.array([rel_pos[0] * leg + rel_pos[1] * combined_radius,
                                      -rel_pos[0] * combined_radius + rel_pos[1] * leg], dtype=float)
                            / max(dist_sq, 1e-12)
                        )
                    direction = normalize(direction)
                    u = (float(np.dot(rel_vel, direction)) - float(np.dot(inv_time_h * rel_pos, direction))) * direction - rel_vel
            else:
                w = rel_vel - inv_time_step * rel_pos
                w_len = vec_norm(w)
                unit_w = w / max(w_len, 1e-12)
                direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
                u = (combined_radius * inv_time_step - w_len) * unit_w

            lines.append(OrcaLine(v_i + 0.5 * u, direction))

        pref = pref_velocities[i].copy()
        if vec_norm(pref) > max_speed:
            pref = normalize(pref) * max_speed

        result, failed = linear_program_2(lines, max_speed, pref, direction_opt=False)
        if failed < len(lines):
            result = linear_program_3(lines, failed, max_speed, result)

        new_vels[i] = result

    return new_vels


# -----------------------------
# Visuals / World
# -----------------------------
def add_ground(client_id, safe_visuals=False):
    if not p.isConnected(client_id):
        return
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    if safe_visuals:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[30, 30, 0.01], physicsClientId=client_id)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[30, 30, 0.01],
            rgbaColor=[0.30, 0.38, 0.28, 1.0], physicsClientId=client_id
        )
        p.createMultiBody(0, col, vis, basePosition=[0, 0, -0.01], physicsClientId=client_id)
    else:
        p.loadURDF("plane.urdf", physicsClientId=client_id)

def add_target_marker(client_id, goal_xy):
    if not p.isConnected(client_id):
        return None
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.10, length=2.4,
                             rgbaColor=[1.0, 0.0, 0.0, 1.0], physicsClientId=client_id)
    pole = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                             basePosition=[gx, gy, 1.2], physicsClientId=client_id)
    try:
        p.addUserDebugLine([gx - 1.2, gy, 0.03], [gx + 1.2, gy, 0.03], [1, 0, 0], 2, physicsClientId=client_id)
        p.addUserDebugLine([gx, gy - 1.2, 0.03], [gx, gy + 1.2, 0.03], [1, 0, 0], 2, physicsClientId=client_id)
    except Exception:
        pass
    return pole

def recolor_drone_body(client_id, body_id, rgba):
    if not p.isConnected(client_id):
        return
    try:
        p.changeVisualShape(body_id, -1, rgbaColor=list(rgba), physicsClientId=client_id)
        nJ = p.getNumJoints(body_id, physicsClientId=client_id)
        for link in range(nJ):
            p.changeVisualShape(body_id, link, rgbaColor=list(rgba), physicsClientId=client_id)
    except Exception:
        pass

def create_balloon(client_id, rgba, radius=0.16):
    if not p.isConnected(client_id):
        return None
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(rgba),
        physicsClientId=client_id,
    )
    bid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, 0],
        physicsClientId=client_id,
    )
    return bid

def update_balloon(client_id, balloon_id, drone_pos_xyz, rgba, z_offset=0.55,
                   string_enabled=True, string_width=2.0):
    if balloon_id is None or (not p.isConnected(client_id)):
        return
    x, y, z = float(drone_pos_xyz[0]), float(drone_pos_xyz[1]), float(drone_pos_xyz[2])
    bx, by, bz = x, y, z + float(z_offset)
    p.resetBasePositionAndOrientation(balloon_id, [bx, by, bz], [0, 0, 0, 1], physicsClientId=client_id)

    if string_enabled:
        try:
            p.addUserDebugLine(
                [x, y, z + 0.05],
                [bx, by, bz - 0.02],
                lineColorRGB=[float(rgba[0]), float(rgba[1]), float(rgba[2])],
                lineWidth=float(string_width),
                lifeTime=0.0,
                physicsClientId=client_id,
            )
        except Exception:
            pass

def add_cylinder_obstacle(client_id, xy, radius=0.35, height=1.6, rgba=(0.3, 0.22, 0.15, 1.0)):
    if not p.isConnected(client_id):
        return None
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(height),
                                physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(radius), length=float(height),
                             rgbaColor=list(rgba), physicsClientId=client_id)
    bid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[float(xy[0]), float(xy[1]), float(height) / 2.0],
        physicsClientId=client_id,
    )
    return bid

def place_minimal_obstacles(client_id, rng, area_x, area_y, goal_xy, leader_xy0):
    cfg = CONFIG["obstacles"]
    if not cfg.get("enabled", True):
        return [], []

    n = int(cfg.get("num", 4))
    r = float(cfg.get("cyl_radius", 0.35))
    h = float(cfg.get("cyl_height", 1.6))
    rgba = cfg.get("rgba", [0.30, 0.22, 0.15, 1.0])

    spawn_margin = float(cfg.get("spawn_margin", 2.5))
    keep_goal = float(cfg.get("keepout_from_goal", 2.8))
    keep_spawn = float(cfg.get("keepout_from_spawn", 2.5))
    keep_each = float(cfg.get("keepout_from_each", 2.2))

    obstacles = []
    body_ids = []

    tries = 0
    while len(obstacles) < n and tries < 500:
        tries += 1
        cand = rand_xy(rng, area_x, area_y, margin=spawn_margin)

        if vec_norm(cand - goal_xy) < keep_goal:
            continue
        if vec_norm(cand - leader_xy0) < keep_spawn:
            continue
        ok = True
        for ob in obstacles:
            if vec_norm(cand - ob["xy"]) < keep_each:
                ok = False
                break
        if not ok:
            continue

        bid = add_cylinder_obstacle(client_id, cand, radius=r, height=h, rgba=rgba)
        if bid is None:
            break
        obstacles.append({"xy": cand.copy(), "radius": r})
        body_ids.append(bid)

    return obstacles, body_ids


# -----------------------------
# Main
# -----------------------------
def run(
    gui=True,
    user_debug_gui=False,
    safe_visuals=False,
    duration_sec=90,
    fast=False,
    no_sync=False,
    num_drones=None,
    orca=True,
    seed=2,
    # knobs:
    max_xy_step=None,
    kp=None,
    hold_err=None,
    hold_min_sec=None,
    acquire_timeout_sec=None,
    formation_scale=None,
    goal_x=None,
    goal_y=None,
    leader_stop_radius=None,
):
    cfg = CONFIG
    N = int(cfg["num_drones"] if num_drones is None else num_drones)
    leader = int(np.clip(int(cfg["leader_idx"]), 0, N - 1))

    area_x = (float(cfg["area"]["x_min"]), float(cfg["area"]["x_max"]))
    area_y = (float(cfg["area"]["y_min"]), float(cfg["area"]["y_max"]))

    goal_xy = np.array(cfg["goal_xy"], dtype=float)
    if goal_x is not None:
        goal_xy[0] = float(goal_x)
    if goal_y is not None:
        goal_xy[1] = float(goal_y)

    leader_stop_radius = float(cfg["leader_stop_radius"] if leader_stop_radius is None else leader_stop_radius)

    hover_cfg = cfg.get("hover_ring", {})
    HOVER_ENABLED = bool(hover_cfg.get("enabled", True))
    HOVER_R = leader_stop_radius if hover_cfg.get("ring_r", None) is None else float(hover_cfg["ring_r"])
    HOVER_GUARD_MARGIN = float(hover_cfg.get("guard_margin", 0.25))
    hover_on = False

    rng = np.random.default_rng(int(seed))

    # Spawn in formation near leader
    FORM_SCALE0 = float(cfg["formation_scale"] if formation_scale is None else formation_scale)
    leader_xy0 = rand_xy(rng, area_x, area_y, margin=4.0)
    psi0 = heading_yaw(leader_xy0, goal_xy)
    R0 = rot2(psi0)

    offsets0 = cfg["formation_offsets_body"]
    spawn_xy = [leader_xy0]
    for j in range(1, N):
        r0 = offsets0.get(j, np.array([-0.9 - 0.7 * (j - 1), 0.0], dtype=float)) * FORM_SCALE0
        p0 = leader_xy0 + (R0 @ r0)
        p0[0] = np.clip(p0[0], area_x[0] + 0.8, area_x[1] - 0.8)
        p0[1] = np.clip(p0[1], area_y[0] + 0.8, area_y[1] - 0.8)
        spawn_xy.append(p0)

    INIT_XYZS = np.zeros((N, 3), dtype=float)
    for i in range(N):
        INIT_XYZS[i, 0] = float(spawn_xy[i][0])
        INIT_XYZS[i, 1] = float(spawn_xy[i][1])
        INIT_XYZS[i, 2] = float(cfg["drone_z_start"])

    env = CtrlAviary(
        drone_model=DEFAULT_DRONE,
        num_drones=N,
        initial_xyzs=INIT_XYZS,
        initial_rpys=np.zeros((N, 3)),
        physics=DEFAULT_PHYSICS,
        pyb_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        user_debug_gui=user_debug_gui,
        obstacles=False,
    )

    pyb = env.getPyBulletClient()
    add_ground(pyb, safe_visuals=safe_visuals)
    add_target_marker(pyb, goal_xy)

    # Place minimal obstacles (physical) + keep list for avoidance
    obs_cfg = cfg.get("obstacles", {})
    obs_rng = np.random.default_rng(int(seed) + int(obs_cfg.get("seed_offset", 101)))
    obstacles, _obstacle_body_ids = place_minimal_obstacles(
        client_id=pyb,
        rng=obs_rng,
        area_x=area_x,
        area_y=area_y,
        goal_xy=goal_xy,
        leader_xy0=leader_xy0,
    )

    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONE) for _ in range(N)]
    action = np.zeros((N, 4), dtype=float)

    drone_body_ids = list(getattr(env, "DRONE_IDS", []))

    # ---- Visuals: recolor + balloons ----
    vis_cfg = cfg["visuals"]
    balloon_ids = [None] * N
    balloon_update_every = 2

    if vis_cfg.get("enabled", True) and p.isConnected(pyb):
        leader_rgba = vis_cfg["leader_rgba"]
        follower_rgba = vis_cfg["follower_rgba"]

        for j in range(N):
            if j < len(drone_body_ids):
                rgba = leader_rgba if j == leader else follower_rgba
                recolor_drone_body(pyb, drone_body_ids[j], rgba)

        bl = vis_cfg.get("balloons", {})
        if bl.get("enabled", True):
            balloon_update_every = int(max(1, bl.get("update_every", 2)))
            alpha = float(bl.get("alpha", 0.90))
            radius = float(bl.get("radius", 0.16))

            for j in range(N):
                base = (leader_rgba if j == leader else follower_rgba).copy()
                base[3] = alpha
                balloon_ids[j] = create_balloon(pyb, rgba=base, radius=radius)

    dt = env.CTRL_TIMESTEP

    # Speed knobs
    MAX_STEP = float(cfg["max_xy_step"] if max_xy_step is None else max_xy_step)
    exec_max_speed = MAX_STEP / max(dt, 1e-6)
    KP = float(cfg["kp"] if kp is None else kp)

    # Acquire knobs
    acq = cfg["formation_acquire"]
    acquire_enabled = bool(acq.get("enabled", True))
    HOLD_ERR = float(acq["hold_err_m"] if hold_err is None else hold_err)
    HOLD_MIN = float(acq["hold_min_sec"] if hold_min_sec is None else hold_min_sec)
    TIMEOUT = float(acq["acquire_timeout_sec"] if acquire_timeout_sec is None else acquire_timeout_sec)

    FORM_SCALE = float(cfg["formation_scale"] if formation_scale is None else formation_scale)

    orca_cfg = cfg["orca"]
    orca_max_speed = float(exec_max_speed)

    boundary_cfg = cfg["boundary"]
    stab_cfg = cfg["stability"]
    safety_cfg = cfg["safety"]

    A_MAX = float(stab_cfg["a_max_mps2"])
    vel_alpha = float(stab_cfg["vel_lpf_alpha"])
    NEAR_GROUND_Z = float(stab_cfg["near_ground_z"])
    RECOVER_Z_BOOST = float(stab_cfg["recover_z_boost"])

    # Arrival latch + snap
    arr_cfg = cfg["arrive"]
    ARRIVE_ENABLED = bool(arr_cfg.get("enabled", True))
    REACH_R = float(arr_cfg.get("reach_r", 0.35))
    HYST = float(arr_cfg.get("hysteresis", 0.10))
    STOP_R_ALL = float(arr_cfg.get("stop_r_all", 0.18))
    STOP_R_HOVER_FOLLOW = float(arr_cfg.get("hover_stop_r_followers", 0.18))
    HOVER_DAMP_FOLLOW = float(arr_cfg.get("hover_damp_followers", 0.85))
    leader_reached = False

    # Leader slow
    slow_cfg = cfg["leader_slow"]
    leader_slow_enabled = bool(slow_cfg.get("enabled", True))
    err_ref = float(slow_cfg.get("err_ref", 1.0))
    min_scale = float(slow_cfg.get("min_scale", 0.20))
    max_scale = float(slow_cfg.get("max_scale", 1.00))

    cmd_vel = np.zeros((N, 2), dtype=float)
    prev_xy = None

    fixed_heading_enabled = bool(cfg["fixed_heading"]["enabled"])


    leader_goal_hold = None

    obs, *_ = env.step(np.zeros((N, 4)))
    start_leader_xy = np.array([obs[leader][0], obs[leader][1]], dtype=float)
    fixed_psi = heading_yaw(start_leader_xy, goal_xy)

    approach_dir = np.array([np.cos(fixed_psi), np.sin(fixed_psi)], dtype=float)
    standoff_fixed = goal_xy - approach_dir * leader_stop_radius

    for j in range(N):
        hover_target = np.array([obs[j][0], obs[j][1], float(cfg["altitude"])], dtype=float)
        action[j], _, _ = ctrl[j].computeControlFromState(dt, obs[j], hover_target, np.zeros(3))

    stable_hold_time = 0.0
    in_move = False
    gate_elapsed = 0.0

    START = time.time()
    print("=== START (LEADER-FOLLOWER -> FIXED TARGET POINT ON RING -> TRUE HOVER) ===")
    print(f"N={N} leader={leader} orca={int(orca)}")
    print(f"goal_xy={goal_xy} leader_stop_radius={leader_stop_radius:.2f}")
    print(f"standoff_fixed={standoff_fixed} (fixed)")
    print(f"step={MAX_STEP:.3f} m/step -> exec_max_speed≈{exec_max_speed:.2f} m/s | kp={KP:.2f}")
    if acquire_enabled:
        print(f"gate: hold_err={HOLD_ERR:.2f}m hold_min={HOLD_MIN:.1f}s timeout={TIMEOUT:.1f}s")
    print(f"hover: enabled={int(HOVER_ENABLED)} r={HOVER_R:.2f}")
    print(f"snap stop_r_all={STOP_R_ALL:.2f} hover_stop_r_followers={STOP_R_HOVER_FOLLOW:.2f}")
    print(f"ORCA max_speed={orca_max_speed:.2f} (matched to exec_max_speed)")
    if obs_cfg.get("enabled", True):
        print(f"obstacles: num={len(obstacles)} radius≈{obs_cfg.get('cyl_radius', 0.35)}")

    status_every_steps = max(1, int(round(float(cfg["print_status"]["every_sec"]) * env.CTRL_FREQ)))

    try:
        for i in range(int(duration_sec * env.CTRL_FREQ)):
            if gui and not p.isConnected(pyb):
                break

            obs, *_ = env.step(action)
            t_sim = i / env.CTRL_FREQ

            xy = np.array([[obs[j][0], obs[j][1]] for j in range(N)], dtype=float)

            # velocities (finite difference)
            if prev_xy is None:
                vxy = np.zeros((N, 2), dtype=float)
            else:
                vxy = (xy - prev_xy) / max(dt, 1e-6)
            prev_xy = xy.copy()

            bl = cfg["visuals"].get("balloons", {})
            if cfg["visuals"].get("enabled", True) and bl.get("enabled", False) and p.isConnected(pyb):
                if (i % balloon_update_every) == 0:
                    zoff = float(bl.get("z_offset", 0.55))
                    string_cfg = bl.get("string", {})
                    string_on = bool(string_cfg.get("enabled", True))
                    string_w = float(string_cfg.get("width", 2.0))

                    for j in range(N):
                        if balloon_ids[j] is None:
                            continue
                        rgba = cfg["visuals"]["leader_rgba"] if j == leader else cfg["visuals"]["follower_rgba"]
                        update_balloon(
                            pyb,
                            balloon_ids[j],
                            drone_pos_xyz=[obs[j][0], obs[j][1], obs[j][2]],
                            rgba=rgba,
                            z_offset=zoff,
                            string_enabled=string_on,
                            string_width=string_w,
                        )

            leader_xy = xy[leader].copy()

            psi = fixed_psi if fixed_heading_enabled else heading_yaw(leader_xy, goal_xy)
            R = rot2(psi)

            # leader reference for slot construction
            leader_ref_xy = leader_goal_hold.copy() if (hover_on and leader_goal_hold is not None) else leader_xy

            offsets = cfg["formation_offsets_body"]
            slot_xy = np.zeros((N, 2), dtype=float)
            slot_xy[leader] = leader_ref_xy
            for j in range(N):
                if j == leader:
                    continue
                r = offsets.get(j, np.array([-0.9 - 0.7 * (j - 1), 0.0], dtype=float)) * FORM_SCALE
                slot_xy[j] = leader_ref_xy + (R @ r)

            # clip slots inside bounds
            slot_xy[:, 0] = np.clip(slot_xy[:, 0], area_x[0] + 0.5, area_x[1] - 0.5)
            slot_xy[:, 1] = np.clip(slot_xy[:, 1], area_y[0] + 0.5, area_y[1] - 0.5)

            # formation error vs actual positions (diagnostic)
            errs = [vec_norm(slot_xy[j] - xy[j]) for j in range(N) if j != leader]
            max_err = float(max(errs)) if errs else 0.0
            mean_err = float(np.mean(errs)) if errs else 0.0

            # leader speed scaling (disable during hover)
            leader_scale = 1.0
            if leader_slow_enabled and (not hover_on):
                leader_scale = float(np.clip(err_ref / max(max_err, 1e-6), min_scale, max_scale))

            # -------------------------
            # Leader goal 
            # -------------------------
            leader_goal = standoff_fixed.copy()

            # gate: hold leader until acquired OR timeout
            if acquire_enabled and (not in_move):
                gate_elapsed += dt
                if max_err <= HOLD_ERR:
                    stable_hold_time += dt
                else:
                    stable_hold_time = 0.0

                if (stable_hold_time < HOLD_MIN) and (gate_elapsed < TIMEOUT):
                    leader_goal = leader_xy.copy()  # hold position during ACQ
                else:
                    in_move = True

            # goals: followers track slots; leader tracks leader_goal
            goals_xy = slot_xy.copy()
            goals_xy[leader] = leader_goal

            # -------------------------
            # Enter HOVER when leader reaches standoff_fixed
            # -------------------------
            if ARRIVE_ENABLED and in_move and (not hover_on) and HOVER_ENABLED:
                d_standoff = vec_norm(standoff_fixed - xy[leader])

                if not leader_reached:
                    if d_standoff <= REACH_R:
                        leader_reached = True
                else:
                    if d_standoff > (REACH_R + HYST):
                        leader_reached = False

                if leader_reached:
                    hover_on = True
                    leader_goal_hold = standoff_fixed.copy()

            # In hover, leader stays at the fixed hold point
            if hover_on and leader_goal_hold is not None:
                goals_xy[leader] = leader_goal_hold.copy()

            # preferred velocities (P)
            pref = np.zeros((N, 2), dtype=float)
            for j in range(N):
                e = goals_xy[j] - xy[j]
                u = KP * e
                v = cap_vector(u, exec_max_speed)
                if j == leader:
                    v *= leader_scale
                pref[j] = v

            # add boundary + obstacle repulsion
            obs_avoid_cfg = cfg.get("obstacles", {}).get("avoid", {})
            obs_avoid_on = bool(cfg.get("obstacles", {}).get("enabled", True) and obs_avoid_cfg.get("enabled", True))
            for j in range(N):
                add = np.zeros(2, dtype=float)
                if boundary_cfg.get("enabled", True):
                    add += boundary_repulsion(
                        xy[j], area_x, area_y,
                        margin=float(boundary_cfg["margin"]),
                        gain=float(boundary_cfg["gain"]),
                        cap=float(boundary_cfg["cap"]),
                    )
                if obs_avoid_on:
                    add += obstacle_repulsion(
                        xy[j],
                        obstacles,
                        influence=float(obs_avoid_cfg.get("influence", 3.0)),
                        gain=float(obs_avoid_cfg.get("gain", 1.25)),
                        cap=float(obs_avoid_cfg.get("cap", 2.5)),
                    )
                pref[j] = cap_vector(pref[j] + add, exec_max_speed)

            # ORCA
            safe = pref.copy()
            if orca:
                agent_r = float(orca_cfg["agent_radius"])
                if hover_on:
                    agent_r *= float(orca_cfg.get("hover_radius_scale", 1.15))

                safe = orca_step_2d(
                    positions=xy, velocities=vxy, pref_velocities=pref,
                    agent_radius=agent_r,
                    neighbor_dist=float(orca_cfg["neighbor_dist"]),
                    time_horizon=float(orca_cfg["time_horizon"]),
                    max_speed=float(orca_max_speed),
                    dt=dt,
                )

            # >>> HOVER-LOCK PATCH (1/2): in hover, ORCA cannot move the leader
            if hover_on and (leader_goal_hold is not None):
                safe[leader] = np.zeros(2, dtype=float)

            # -------------------------
            # -------------------------
            if in_move:
                d_to_goal_L = vec_norm(goal_xy - xy[leader])
                if d_to_goal_L <= (HOVER_R + HOVER_GUARD_MARGIN):
                    r_vec = xy[leader] - goal_xy
                    r_norm = vec_norm(r_vec)
                    if r_norm > 1e-9:
                        r_hat = r_vec / r_norm
                        vL = safe[leader]
                        v_inward = float(np.dot(vL, -r_hat))
                        if v_inward > 0.0:
                            safe[leader] = vL + v_inward * r_hat

            safe[leader] = cap_vector(safe[leader] * leader_scale, exec_max_speed)

            dv_max = float(A_MAX) * dt
            for j in range(N):
                dv = safe[j] - cmd_vel[j]
                n = vec_norm(dv)
                if n > dv_max:
                    safe[j] = cmd_vel[j] + (dv / n) * dv_max

            for j in range(N):
                cmd_vel[j] = vel_alpha * cmd_vel[j] + (1.0 - vel_alpha) * safe[j]
                cmd_vel[j] = cap_vector(cmd_vel[j], exec_max_speed)

            if hover_on and (leader_goal_hold is not None):
                cmd_vel[leader] = np.zeros(2, dtype=float)

            if hover_on:
                for j in range(N):
                    if j != leader:
                        cmd_vel[j] *= HOVER_DAMP_FOLLOW

            for j in range(N):
                stop_r = float(STOP_R_ALL)
                if hover_on and (j != leader):
                    stop_r = max(stop_r, float(STOP_R_HOVER_FOLLOW))

                allow_reverse = (j != leader)  # leader False, followers True
                cmd_vel[j] = arrive_clamp_velocity(
                    xy[j], goals_xy[j], cmd_vel[j], dt,
                    stop_r=stop_r,
                    allow_reverse=allow_reverse
                )

            altitude_cmd = float(cfg["altitude"])
            tilt_soft = float(safety_cfg["tilt_soft_rad"])
            tilt_hard = float(safety_cfg["tilt_hard_rad"])
            z_boost = float(safety_cfg["z_boost"])

            targets = np.zeros((N, 3), dtype=float)
            for j in range(N):
                roll = float(obs[j][3]); pitch = float(obs[j][4])
                tilt = max(abs(roll), abs(pitch))
                alt = float(obs[j][2])

                v = cmd_vel[j].copy()

                # Tilt scaling
                if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                    s = 1.0 - (tilt - tilt_soft) / max(tilt_hard - tilt_soft, 1e-6)
                    v *= float(np.clip(s, 0.0, 1.0))

                goal_err = vec_norm(goals_xy[j] - xy[j])

                if hover_on and (j == leader) and (leader_goal_hold is not None):
                    new_xy = leader_goal_hold.copy()
                elif goal_err <= STOP_R_ALL:
                    new_xy = goals_xy[j].copy()
                else:
                    new_xy = xy[j] + v * dt

                if not (hover_on and (j == leader) and (leader_goal_hold is not None)):
                    step = new_xy - xy[j]
                    step_len = vec_norm(step)
                    if step_len > MAX_STEP:
                        new_xy = xy[j] + (step / step_len) * MAX_STEP

                z_cmd = altitude_cmd
                if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                    z_cmd = max(z_cmd, altitude_cmd + z_boost)

                if alt < NEAR_GROUND_Z:
                    new_xy = xy[j].copy()
                    z_cmd = max(z_cmd, altitude_cmd + float(RECOVER_Z_BOOST))

                z_cmd = float(np.clip(z_cmd, float(cfg["drone_z_start"]), 3.0))
                targets[j, :] = np.array([new_xy[0], new_xy[1], z_cmd], dtype=float)

            for j in range(N):
                action[j], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=dt,
                    state=obs[j],
                    target_pos=targets[j],
                    target_rpy=np.zeros(3),
                )

            # status
            if cfg["print_status"]["enabled"] and (i % status_every_steps == 0):
                phase = "HOVER" if hover_on else ("MOVE" if in_move else "ACQ")
                dmin = min_pairwise_dist(xy)
                dL_goal = vec_norm(goal_xy - xy[leader])
                dL_standoff = vec_norm(standoff_fixed - xy[leader])
                print(f"t={t_sim:5.1f}s phase={phase} dL(goal)={dL_goal:4.2f} dL(standoff)={dL_standoff:4.2f} "
                      f"max_err={max_err:4.2f} mean_err={mean_err:4.2f} dmin={dmin:4.2f}")

            if gui and (not no_sync) and (not fast):
                sync(i, START, dt)

    except p.error:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leader–Follower -> fixed ring point -> true hover + balloons + minimal obstacles"
    )

    parser.add_argument("--gui", default=True, type=str2bool)
    parser.add_argument("--user_debug_gui", default=False, type=str2bool)
    parser.add_argument("--safe_visuals", default=False, type=str2bool)
    parser.add_argument("--duration_sec", default=90, type=int)

    parser.add_argument("--fast", default=False, type=str2bool)
    parser.add_argument("--no_sync", default=False, type=str2bool)

    parser.add_argument("--num_drones", default=None, type=int)
    parser.add_argument("--orca", default=True, type=str2bool)
    parser.add_argument("--seed", default=2, type=int)

    parser.add_argument("--max_xy_step", default=None, type=float, help="meters per control step (bigger=faster)")
    parser.add_argument("--kp", default=None, type=float, help="P gain for slot/goal tracking")

    parser.add_argument("--hold_err", default=None, type=float)
    parser.add_argument("--hold_min_sec", default=None, type=float)
    parser.add_argument("--acquire_timeout_sec", default=None, type=float)

    parser.add_argument("--formation_scale", default=None, type=float)
    parser.add_argument("--goal_x", default=None, type=float)
    parser.add_argument("--goal_y", default=None, type=float)
    parser.add_argument("--leader_stop_radius", default=None, type=float)

    args = parser.parse_args()

    run(
        gui=args.gui,
        user_debug_gui=args.user_debug_gui,
        safe_visuals=args.safe_visuals,
        duration_sec=args.duration_sec,
        fast=args.fast,
        no_sync=args.no_sync,
        num_drones=args.num_drones,
        orca=args.orca,
        seed=args.seed,
        max_xy_step=args.max_xy_step,
        kp=args.kp,
        hold_err=args.hold_err,
        hold_min_sec=args.hold_min_sec,
        acquire_timeout_sec=args.acquire_timeout_sec,
        formation_scale=args.formation_scale,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        leader_stop_radius=args.leader_stop_radius,
    )
