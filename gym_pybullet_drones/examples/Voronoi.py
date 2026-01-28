# ============================================================
# Forest SAR: Voronoi-Based Swarm Search & Rescue
# Run: python Voronoi.py --gui True --user_debug_gui True --safe_visuals True --plot False
# ============================================================

import os
import time
import argparse
import numpy as np

import pybullet as p
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CONFIG = {
    "area": {"x_min": -6.0, "x_max": 6.0, "y_min": -6.0, "y_max": 6.0},

    "num_drones": 5,
    "drone_z_start": 0.6,
    "search_altitude": 1.2,

    "max_xy_step": 0.18,

    "orca": {
        "enabled": True,
        "agent_radius": 0.45,
        "neighbor_dist": 4.0,
        "time_horizon": 4.0,
        "orca_planning_speed_mps": 8.0,
    },

    "boundary": {
        "enabled": True,
        "margin": 1.2,
        "gain": 0.8,
        "cap": 2.0,
    },

    "lidar": {
        "enabled": True,
        "num_rays": 12,
        "max_range": 4.0,
        "z_offsets": [0.15, -0.15],
        "influence_radius": 2.5,
        "gain": 0.5,
        "cap": 6.0,
    },

    "lidar_safety": {
        "enabled": True,
        "d_stop": 0.70,         
        "d_release": 0.80,       
        "d_slow": 1.60,
        "slow_scale": 0.15,

        "crawl_scale": 0.05,

        "unstick_enabled": True,
        "unstick_hold_steps": 12,
        "unstick_duration_steps": 20,
        "unstick_speed": 3.0,
        "unstick_z_bump": 0.50,
    },

    "victim": {
        "pos": np.array([0.0, 0.0, 0.0], dtype=float),
        "urdf": "humanoid/humanoid.urdf",
        "use_fixed_base": True,
        "scale": 0.12,
        "z_offset": 0.02,
        "upright_euler": [1.5708, 0.0, 0.0],
        "yaw_deg": 25.0,
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "detect_radius_m": 2,
    },

    "rescue": {
        "hover_altitude": 1.0,
        "rescuer_stop_radius": 0.30,
        "support_speed_scale": 0.65,
        "rescuer_speed_scale": 0.55,

        "support_min_radius": 2.5,

        "ring_radius": 2.8,
        "ring_phase_deg": 25.0,
        "support_slot_stop_radius": 0.35,
        "support_altitude": 1.3,
    },

    "support": {
        "loiter_radius_min": 3.0,
        "loiter_radius_max": 4.0,
        "loiter_altitude": 1.5,
        "angular_speed": 0.3,
    },

    "exploration": {"dist_threshold": 0.05, "jitter_scale": 0.4},

    "safety": {
        "enabled": True,
        "tilt_soft_rad": 0.55,
        "tilt_hard_rad": 0.90,
        "z_boost": 0.60,
    },

    "stability": {
        "a_max_mps2": 10.0,
        "near_ground_z": 0.35,
        "recover_z_boost": 0.8,
    },

    "motion": {
        "goal_lpf_alpha": 0.88,
        "voronoi_every": 2,
        "vel_lpf_alpha": 0.82,
    },

    "spawn": {
        "enabled": True,
        "max_tries": 5000,
        "boundary_margin": 1.0,
        "min_interdrone_dist": 1.0,
        "obstacle_clearance": 1.2,
        "victim_clearance": 1.0,
    },

    "forest": {
        "enabled": True,
        "seed": 7,
        "terrain": {
            "enabled": False,      # IMPORTANT: default is False (your previous code accidentally treated missing as True)
            "cell_size": 0.25,
            "height_scale": 0.45,
            "base_height": 0.0,
        },
        "obstacles": {
            "num_trees": 16,
            "num_rocks": 14,
            "num_logs": 10,
            "num_bushes": 12,
            "min_spacing": 0.95,
        },
    },

    "logging": {
        "enabled": True,
        "print_every_sec": 1.0,

        "print_drone_idx": 0,
        "print_all_drones": True,
        "print_rescue_status": True,

        "also_close_radius_m": 1.25,
        "very_close_radius_m": 0.60,
        "print_victim_distances": True,
    }
}


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = CONFIG["num_drones"]
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False

DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 120
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

AREA_X = (CONFIG["area"]["x_min"], CONFIG["area"]["x_max"])
AREA_Y = (CONFIG["area"]["y_min"], CONFIG["area"]["y_max"])
DRONE_Z = float(CONFIG["drone_z_start"])
SEARCH_ALTITUDE = float(CONFIG["search_altitude"])
MAX_STEP = float(CONFIG["max_xy_step"])


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def vec_norm(v):
    return float(np.linalg.norm(v))


def cap_vector(v, cap):
    n = vec_norm(v)
    if cap is not None and cap > 0.0 and n > cap:
        return (v / n) * cap
    return v


def rand_xy(rng, x_range, y_range, margin=1.0):
    return np.array(
        [
            rng.uniform(x_range[0] + margin, x_range[1] - margin),
            rng.uniform(y_range[0] + margin, y_range[1] - margin),
        ],
        dtype=float,
    )


def sample_safe_victim_xy(rng, area_x, area_y, margin, obstacle_centers, obstacle_clear, max_tries=5000):
    obstacle_centers = list(obstacle_centers or [])
    for _ in range(int(max_tries)):
        xy = rand_xy(rng, area_x, area_y, margin=margin)
        if all(vec_norm(xy - oc) >= float(obstacle_clear) for oc in obstacle_centers):
            return xy
    return rand_xy(rng, area_x, area_y, margin=margin)


# =========================
# RESCUE MODE: RING SLOTS
# =========================
_RESCUE_SLOT_OF = {}  # drone_id -> slot index


def _compute_ring_slot(victim_xy, R, k, M, phase_deg=0.0):
    if M <= 0:
        return np.array(victim_xy, dtype=float)
    phase = np.deg2rad(phase_deg)
    theta = phase + 2.0 * np.pi * (k / float(M))
    return np.array(
        [victim_xy[0] + R * np.cos(theta), victim_xy[1] + R * np.sin(theta)],
        dtype=float,
    )


def rescue_update_goals_ring(goals_xy, found_id, victim_xy, num_drones, ring_radius=2.6, phase_deg=0.0):
    goals_xy = np.array(goals_xy, dtype=float, copy=True)

    non_resc = [i for i in range(num_drones) if i != found_id]

    for k in list(_RESCUE_SLOT_OF.keys()):
        if k not in non_resc:
            _RESCUE_SLOT_OF.pop(k, None)

    taken = set(_RESCUE_SLOT_OF.values())
    nxt = 0
    for i in non_resc:
        if i in _RESCUE_SLOT_OF:
            continue
        while nxt in taken:
            nxt += 1
        _RESCUE_SLOT_OF[i] = nxt
        taken.add(nxt)
        nxt += 1

    M = max(1, num_drones - 1)
    for i in range(num_drones):
        if i == found_id:
            goals_xy[i] = np.array(victim_xy, dtype=float)
        else:
            slot = _RESCUE_SLOT_OF[i]
            goals_xy[i] = _compute_ring_slot(victim_xy, ring_radius, slot, M, phase_deg)

    return goals_xy


# -----------------------------------------------------------------------------
# Victim (humanoid or fallback capsule)
# -----------------------------------------------------------------------------
def set_humanoid_standing_pose(client_id, humanoid_id):
    n_joints = p.getNumJoints(humanoid_id, physicsClientId=client_id)
    for j in range(n_joints):
        try:
            p.resetJointState(humanoid_id, j, targetValue=0.0, physicsClientId=client_id)
        except Exception:
            pass

    def safe_set(j, val):
        if 0 <= j < n_joints:
            try:
                p.resetJointState(humanoid_id, j, targetValue=val, physicsClientId=client_id)
            except Exception:
                pass

    for j in range(n_joints):
        info = p.getJointInfo(humanoid_id, j, physicsClientId=client_id)
        name = info[1].decode("utf-8", errors="ignore").lower()
        if "knee" in name:
            safe_set(j, 0.25)
        elif "hip" in name:
            safe_set(j, -0.15)
        elif "ankle" in name:
            safe_set(j, -0.05)
        elif "shoulder" in name:
            safe_set(j, 0.10)
        elif "elbow" in name:
            safe_set(j, 0.15)


def add_capsule_victim(client_id, pos_xyz, rgba=(1, 0, 0, 1)):
    pos = [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]) + 0.55]
    col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.14, height=0.8, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.14, length=0.8, rgbaColor=list(rgba), physicsClientId=client_id)
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        physicsClientId=client_id,
    )
    return body_id


def add_humanoid_victim(client_id, victim_cfg, safe_visuals=False):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    if safe_visuals:
        return add_capsule_victim(client_id, victim_cfg["pos"], rgba=victim_cfg.get("rgba", [1, 0, 0, 1]))

    pos = victim_cfg["pos"].astype(float).tolist()
    pos[2] = float(pos[2]) + float(victim_cfg.get("z_offset", 0.0))

    yaw = np.deg2rad(float(victim_cfg.get("yaw_deg", 0.0)))
    upright_euler = victim_cfg.get("upright_euler", [0.0, 0.0, 0.0])
    roll = float(upright_euler[0])
    pitch = float(upright_euler[1])
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])

    urdf_path = str(victim_cfg.get("urdf", "humanoid/humanoid.urdf"))
    use_fixed = bool(victim_cfg.get("use_fixed_base", True))
    rgba = victim_cfg.get("rgba", [1.0, 0.0, 0.0, 1.0])
    scale = float(victim_cfg.get("scale", 1.0))

    try:
        victim_id = p.loadURDF(
            urdf_path,
            basePosition=pos,
            baseOrientation=orn,
            useFixedBase=use_fixed,
            globalScaling=scale,
            physicsClientId=client_id,
        )
    except Exception as e:
        print(f"[WARN] Could not load '{urdf_path}' ({e}). Falling back to capsule victim.")
        return add_capsule_victim(client_id, victim_cfg["pos"], rgba=rgba)

    set_humanoid_standing_pose(client_id, victim_id)

    try:
        for link in range(-1, p.getNumJoints(victim_id, physicsClientId=client_id)):
            p.changeVisualShape(victim_id, link, rgbaColor=rgba, physicsClientId=client_id)
    except Exception:
        pass

    return victim_id


def victim_found_by_distance(drone_xy, victim_id, client_id, radius_m):
    vpos, _ = p.getBasePositionAndOrientation(victim_id, physicsClientId=client_id)
    vxy = np.array([vpos[0], vpos[1]], dtype=float)
    d = float(np.linalg.norm(np.asarray(drone_xy, dtype=float) - vxy))
    return (d <= float(radius_m)), vxy


# -----------------------------------------------------------------------------
# Bounded Voronoi / Lloyd
# -----------------------------------------------------------------------------
def poly_area_centroid(poly):
    if poly is None or len(poly) < 3:
        return 0.0, None

    x = poly[:, 0]
    y = poly[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)

    cross = x * y2 - x2 * y
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-12:
        return 0.0, None

    cx = (1.0 / (6.0 * area)) * np.sum((x + x2) * cross)
    cy = (1.0 / (6.0 * area)) * np.sum((y + y2) * cross)
    return float(abs(area)), np.array([cx, cy], dtype=float)


def clip_polygon_halfplane(poly, n, c, eps=1e-12):
    if poly is None or len(poly) == 0:
        return np.zeros((0, 2), dtype=float)

    n = np.asarray(n, dtype=float)

    def inside(pt):
        return (np.dot(n, pt) - c) <= eps

    def intersect(p1, p2):
        d = p2 - p1
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-12:
            return p1
        t = (c - float(np.dot(n, p1))) / denom
        t = float(np.clip(t, 0.0, 1.0))
        return p1 + t * d

    out = []
    prev = poly[-1]
    prev_in = inside(prev)

    for curr in poly:
        curr_in = inside(curr)
        if curr_in:
            if not prev_in:
                out.append(intersect(prev, curr))
            out.append(curr)
        else:
            if prev_in:
                out.append(intersect(prev, curr))
        prev = curr
        prev_in = curr_in

    if len(out) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(out, dtype=float)


def compute_voronoi_centroids_bounded(drone_xy, x_range, y_range):
    pts = np.asarray(drone_xy, dtype=float)
    N = pts.shape[0]
    centroids = np.zeros_like(pts)

    box = np.array(
        [
            [x_range[0], y_range[0]],
            [x_range[1], y_range[0]],
            [x_range[1], y_range[1]],
            [x_range[0], y_range[1]],
        ],
        dtype=float,
    )

    if N == 0:
        return centroids
    if N == 1:
        return np.array(
            [[(x_range[0] + x_range[1]) * 0.5, (y_range[0] + y_range[1]) * 0.5]],
            dtype=float,
        )
    if np.allclose(pts, pts[0]):
        return pts.copy()

    for i in range(N):
        pi = pts[i]
        cell = box.copy()

        for j in range(N):
            if j == i:
                continue
            pj = pts[j]
            n = (pj - pi)
            c = (np.dot(pj, pj) - np.dot(pi, pi)) / 2.0
            cell = clip_polygon_halfplane(cell, n, c)
            if len(cell) < 3:
                break

        _, ctr = poly_area_centroid(cell)
        centroids[i] = ctr if ctr is not None else pi

    return centroids


# -----------------------------------------------------------------------------
# Boundary repulsion
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# LIDAR
# -----------------------------------------------------------------------------
def lidar_scan_2d_multi_z(drone_pos, pyb_client, cfg, ignore_body_ids=None):
    if not cfg.get("enabled", False):
        return []

    ignore_body_ids = set(ignore_body_ids or [])

    num_rays = int(cfg["num_rays"])
    max_range = float(cfg["max_range"])
    z_offsets = cfg.get("z_offsets", [0.1])

    hits_xy = []
    for zo in z_offsets:
        z = float(drone_pos[2]) + float(zo)

        ray_from, ray_to = [], []
        for k in range(num_rays):
            ang = 2.0 * np.pi * k / num_rays
            dx, dy = np.cos(ang), np.sin(ang)

            start = [float(drone_pos[0]), float(drone_pos[1]), z]
            end = [float(drone_pos[0] + dx * max_range), float(drone_pos[1] + dy * max_range), z]

            ray_from.append(start)
            ray_to.append(end)

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=pyb_client)

        for r in results:
            body_id = r[0]
            hit_frac = r[2]
            if hit_frac >= 1.0:
                continue
            if body_id in ignore_body_ids:
                continue

            hit_pos = np.array(r[3], dtype=float)
            hits_xy.append(hit_pos[:2])

    return hits_xy


def compute_lidar_avoidance_offset(drone_xy, idx, lidar_hits_per_drone, influence_radius, gain, cap=2.0):
    p_i = drone_xy[idx]
    offset = np.zeros(2)
    hits = lidar_hits_per_drone[idx]
    if not hits:
        return offset

    for hp in hits:
        hp = np.asarray(hp, dtype=float)
        diff = p_i - hp
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            diff = np.array([1.0, 0.0])
            dist = 1e-3

        if dist < influence_radius:
            direction = diff / dist
            rep_mag = gain * max(0.0, (1.0 / max(dist, 1e-3) - 1.0 / influence_radius))
            offset += direction * rep_mag

    return cap_vector(offset, cap)


def min_lidar_distance(drone_xy, hits_xy):
    if not hits_xy:
        return np.inf
    dmin = np.inf
    for hp in hits_xy:
        d = np.linalg.norm(drone_xy - np.asarray(hp, dtype=float))
        dmin = min(dmin, d)
    return dmin


def closest_lidar_hit_vector(drone_xy, hits_xy):
    if not hits_xy:
        return None
    best = None
    best_d = np.inf
    for hp in hits_xy:
        v = drone_xy - np.asarray(hp, dtype=float)
        d = float(np.linalg.norm(v))
        if d < best_d:
            best_d = d
            best = v
    if best is None or best_d < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return best / best_d


# -----------------------------------------------------------------------------
# Forest scene
# -----------------------------------------------------------------------------
def add_box_ground_visual(client_id, area_x, area_y):
    hx = (area_x[1] - area_x[0]) * 0.5
    hy = (area_y[1] - area_y[0]) * 0.5
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, 0.01],
        rgbaColor=[0.30, 0.40, 0.28, 1.0],
        physicsClientId=client_id,
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, 0.0],
        physicsClientId=client_id,
    )


def add_tree_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])

    trunk_h = 1.1 + 0.5 * rng.random()
    trunk_r = 0.10 + 0.05 * rng.random()

    col1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h, physicsClientId=client_id)
    vis1 = p.createVisualShape(
        p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h, rgbaColor=[0.45, 0.28, 0.12, 1.0], physicsClientId=client_id
    )
    p.createMultiBody(0, col1, vis1, basePosition=[x, y, trunk_h * 0.5], physicsClientId=client_id)

    crown_r = 0.45 + 0.25 * rng.random()
    col2 = p.createCollisionShape(p.GEOM_SPHERE, radius=crown_r, physicsClientId=client_id)
    vis2 = p.createVisualShape(p.GEOM_SPHERE, radius=crown_r, rgbaColor=[0.12, 0.42, 0.14, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col2, vis2, basePosition=[x, y, trunk_h + crown_r * 0.8], physicsClientId=client_id)


def add_rock_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    r = 0.18 + 0.22 * rng.random()
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.35, 0.35, 0.35, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col, vis, basePosition=[x, y, r], physicsClientId=client_id)


def add_log_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    length = 1.0 + 0.8 * rng.random()
    radius = 0.10 + 0.06 * rng.random()
    yaw = float(rng.uniform(0, 180))
    orn = p.getQuaternionFromEuler([0, 0, np.deg2rad(yaw)])
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=client_id)
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[0.40, 0.26, 0.12, 1.0], physicsClientId=client_id
    )
    p.createMultiBody(0, col, vis, basePosition=[x, y, radius], baseOrientation=orn, physicsClientId=client_id)


def add_bush_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    r = 0.25 + 0.18 * rng.random()
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.15, 0.45, 0.15, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col, vis, basePosition=[x, y, r], physicsClientId=client_id)


def precompute_obstacle_positions(area_x, area_y, forest_cfg):
    rng_place = np.random.default_rng(int(forest_cfg.get("seed", 0)))
    obst_cfg = forest_cfg.get("obstacles", {})
    min_spacing = float(obst_cfg.get("min_spacing", 0.95))

    counts = {
        "trees": int(obst_cfg.get("num_trees", 16)),
        "rocks": int(obst_cfg.get("num_rocks", 14)),
        "logs": int(obst_cfg.get("num_logs", 10)),
        "bushes": int(obst_cfg.get("num_bushes", 12)),
    }

    placed = []

    def place_xy():
        for _ in range(5000):
            xy = rand_xy(rng_place, area_x, area_y, margin=1.0)
            if vec_norm(xy - np.array([0.0, 0.0])) < 1.2:
                continue
            if all(vec_norm(xy - q) >= min_spacing for q in placed):
                placed.append(xy)
                return xy
        xy = rand_xy(rng_place, area_x, area_y, margin=1.0)
        placed.append(xy)
        return xy

    pos = {k: [] for k in ["trees", "rocks", "logs", "bushes"]}
    for _ in range(counts["trees"]):
        pos["trees"].append(place_xy())
    for _ in range(counts["rocks"]):
        pos["rocks"].append(place_xy())
    for _ in range(counts["logs"]):
        pos["logs"].append(place_xy())
    for _ in range(counts["bushes"]):
        pos["bushes"].append(place_xy())

    return pos


def build_forest_scene_from_positions(client_id, area_x, area_y, forest_cfg, obst_positions, safe_visuals=True):
    rng = np.random.default_rng(int(forest_cfg.get("seed", 0)) + 999)
    obstacle_centers = []

    terrain_cfg = forest_cfg.get("terrain", {})
    # FIX: default should be False (heightfield off unless explicitly enabled)
    if terrain_cfg.get("enabled", False):
        cell = float(terrain_cfg.get("cell_size", 0.25))
        hx = int(max(16, round((area_x[1] - area_x[0]) / cell)))
        hy = int(max(16, round((area_y[1] - area_y[0]) / cell)))

        noise = rng.normal(0.0, 1.0, size=(hy, hx))
        for _ in range(3):
            noise = (
                noise
                + np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0)
                + np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)
            ) / 5.0

        height_scale = float(terrain_cfg.get("height_scale", 0.45))
        base_h = float(terrain_cfg.get("base_height", 0.0))
        heights = (noise * height_scale + base_h).astype(np.float32)
        heightfield_data = heights.flatten(order="C")

        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[(area_x[1] - area_x[0]) / hx, (area_y[1] - area_y[0]) / hy, 1.0],
            heightfieldData=heightfield_data,
            numHeightfieldRows=hy,
            numHeightfieldColumns=hx,
            physicsClientId=client_id,
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, -0.02],
            physicsClientId=client_id,
        )
        add_box_ground_visual(client_id, area_x, area_y)

    def remember_center(xy):
        obstacle_centers.append(np.array([xy[0], xy[1]], dtype=float))

    for xy in obst_positions["trees"]:
        add_tree_primitive(client_id, xy, rng)
        remember_center(xy)
    for xy in obst_positions["rocks"]:
        add_rock_primitive(client_id, xy, rng)
        remember_center(xy)
    for xy in obst_positions["logs"]:
        add_log_primitive(client_id, xy, rng)
        remember_center(xy)
    for xy in obst_positions["bushes"]:
        add_bush_primitive(client_id, xy, rng)
        remember_center(xy)

    return obstacle_centers


# -----------------------------------------------------------------------------
# ORCA
# -----------------------------------------------------------------------------
class OrcaLine:
    __slots__ = ("point", "direction")
    def __init__(self, point, direction):
        self.point = np.asarray(point, dtype=float)
        self.direction = np.asarray(direction, dtype=float)

def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def norm(a):
    return float(np.linalg.norm(a))

def normalize(a, eps=1e-12):
    n = norm(a)
    if n < eps:
        return np.zeros_like(a)
    return a / n

def linear_program_1(lines, line_no, radius, opt_vel, direction_opt):
    line = lines[line_no]
    dotp = np.dot(line.point, line.direction)
    discriminant = dotp * dotp + radius * radius - np.dot(line.point, line.point)
    if discriminant < 0.0:
        return False, None

    sqrt_disc = np.sqrt(discriminant)
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
        t = t_right if np.dot(opt_vel, line.direction) > 0.0 else t_left
    else:
        t = np.dot(line.direction, opt_vel - line.point) / max(np.dot(line.direction, line.direction), 1e-12)
        t = min(max(t, t_left), t_right)

    return True, line.point + t * line.direction

def linear_program_2(lines, radius, opt_vel, direction_opt):
    if direction_opt:
        result = normalize(opt_vel) * radius
    else:
        result = normalize(opt_vel) * radius if norm(opt_vel) > radius else opt_vel.copy()

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

            opt_dir = np.array([-line.direction[1], line.direction[0]])
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
            dist_sq = np.dot(rel_pos, rel_pos)
            if dist_sq > neighbor_dist * neighbor_dist:
                continue

            rel_vel = v_i - v_j

            if dist_sq > combined_radius_sq:
                w = rel_vel - inv_time_h * rel_pos
                w_len_sq = np.dot(w, w)
                dot = np.dot(w, rel_pos)

                if dot < 0.0 and dot * dot > combined_radius_sq * w_len_sq:
                    w_len = np.sqrt(max(w_len_sq, 1e-12))
                    unit_w = w / w_len
                    direction = np.array([unit_w[1], -unit_w[0]])
                    u = (combined_radius * inv_time_h - w_len) * unit_w
                else:
                    dist = np.sqrt(max(dist_sq, 1e-12))
                    leg = np.sqrt(max(dist_sq - combined_radius_sq, 1e-12))
                    if cross2(rel_pos, w) > 0.0:
                        direction = np.array(
                            [rel_pos[0] * leg - rel_pos[1] * combined_radius,
                             rel_pos[0] * combined_radius + rel_pos[1] * leg]
                        ) / max(dist_sq, 1e-12)
                    else:
                        direction = -np.array(
                            [rel_pos[0] * leg + rel_pos[1] * combined_radius,
                             -rel_pos[0] * combined_radius + rel_pos[1] * leg]
                        ) / max(dist_sq, 1e-12)
                    direction = normalize(direction)
                    u = (np.dot(rel_vel, direction) - np.dot(inv_time_h * rel_pos, direction)) * direction - rel_vel
            else:
                w = rel_vel - inv_time_step * rel_pos
                w_len = norm(w)
                unit_w = w / max(w_len, 1e-12)
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_step - w_len) * unit_w

            line_point = v_i + 0.5 * u
            lines.append(OrcaLine(line_point, direction))

        pref = pref_velocities[i].copy()
        if norm(pref) > max_speed:
            pref = normalize(pref) * max_speed

        result, failed = linear_program_2(lines, max_speed, pref, direction_opt=False)
        if failed < len(lines):
            result = linear_program_3(lines, failed, max_speed, result)

        new_vels[i] = result

    return new_vels


# -----------------------------------------------------------------------------
# Spawn + env id helpers
# -----------------------------------------------------------------------------
def sample_safe_initial_xy(
    num_drones,
    rng,
    area_x,
    area_y,
    spawn_cfg,
    obstacle_centers,
    obstacle_clear,
    victim_centers,
    victim_clear,
):
    bx0, bx1 = area_x
    by0, by1 = area_y

    bmargin = float(spawn_cfg["boundary_margin"])
    min_d = float(spawn_cfg["min_interdrone_dist"])
    max_tries = int(spawn_cfg["max_tries"])

    sx0, sx1 = bx0 + bmargin, bx1 - bmargin
    sy0, sy1 = by0 + bmargin, by1 - bmargin

    if sx1 <= sx0 or sy1 <= sy0:
        sx0, sx1 = bx0 + 0.5, bx1 - 0.5
        sy0, sy1 = by0 + 0.5, by1 - 0.5

    pts = []
    tries = 0

    obstacle_centers = list(obstacle_centers or [])
    victim_centers = list(victim_centers or [])

    while len(pts) < num_drones and tries < max_tries:
        tries += 1
        cand = np.array([rng.uniform(sx0, sx1), rng.uniform(sy0, sy1)], dtype=float)

        if any(vec_norm(cand - prev) < min_d for prev in pts):
            continue
        if any(vec_norm(cand - oc) < obstacle_clear for oc in obstacle_centers):
            continue
        if any(vec_norm(cand - vc) < victim_clear for vc in victim_centers):
            continue

        pts.append(cand)

    if len(pts) < num_drones:
        pts = []
        for _ in range(num_drones):
            pts.append(
                np.array(
                    [rng.uniform(bx0 * 0.7, bx1 * 0.7), rng.uniform(by0 * 0.7, by1 * 0.7)],
                    dtype=float,
                )
            )

    return np.stack(pts, axis=0)


def _get_drone_body_ids_from_env(env):
    # Try a few known attributes across gym-pybullet-drones versions
    for name in ["DRONE_IDS", "DRONE_IDS_LIST", "DRONES_IDS", "DRONE_IDs"]:
        if hasattr(env, name):
            ids = getattr(env, name)
            try:
                ids = list(ids)
                if len(ids) > 0 and all(isinstance(x, (int, np.integer)) for x in ids):
                    return ids
            except Exception:
                pass
    return []


# -----------------------------------------------------------------------------
# Required-metrics logging + plotting
# -----------------------------------------------------------------------------
def append_run_metrics_csv(csv_path, run_id, found, ttf_sec):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if need_header:
            f.write("run_id,found,ttf_sec\n")
        ttf_str = f"{float(ttf_sec):.3f}" if ttf_sec is not None else ""
        f.write(f"{int(run_id)},{1 if found else 0},{ttf_str}\n")


def plot_ttf_from_csv(csv_path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Plot requested but pandas/matplotlib not available: {e}")
        return

    if not os.path.exists(csv_path):
        print(f"[WARN] Plot requested but CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "found" not in df.columns or "ttf_sec" not in df.columns:
        print(f"[WARN] CSV missing required columns; cannot plot: {csv_path}")
        return

    success_rate = float(df["found"].mean()) if len(df) else 0.0
    ttf = df.loc[df["found"] == 1, "ttf_sec"].dropna()

    print(f"\n[METRICS] CSV={csv_path}")
    print(f"[METRICS] runs={len(df)} | success_rate={success_rate*100:.1f}%")
    if len(ttf) > 0:
        mu = float(ttf.mean())
        sd = float(ttf.std(ddof=1)) if len(ttf) > 1 else 0.0
        print(f"[METRICS] TTF mean={mu:.2f}s | std={sd:.2f}s | n={len(ttf)}")
    else:
        print("[METRICS] No successful runs -> no TTF stats")

    plt.figure()
    plt.hist(ttf, bins=10)
    plt.xlabel("Time to Find (s)")
    plt.ylabel("Count")
    plt.title("Time-to-Find Distribution (successful runs)")
    plt.show()


# -----------------------------------------------------------------------------
# Main sim loop
# -----------------------------------------------------------------------------
def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=False,
    plot=True,
    user_debug_gui=False,
    obstacles=False,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
    fast=True,
    render_every=10,
    lidar_every=4,
    camera_every=8,
    no_sync=True,
    safe_visuals=True,
    # metrics logging
    log_csv="results_runs.csv",
    run_id=-1,
):
    rng = np.random.default_rng()

    motion_cfg = CONFIG.get("motion", {})
    goal_alpha = float(motion_cfg.get("goal_lpf_alpha", 0.88))
    vel_alpha = float(motion_cfg.get("vel_lpf_alpha", 0.82))
    voronoi_every = int(motion_cfg.get("voronoi_every", 2))

    if fast:
        lidar_every = 1

    if not fast:
        render_every = 1
        lidar_every = 1
        no_sync = False

    forest_cfg = CONFIG["forest"]
    obst_positions = precompute_obstacle_positions(AREA_X, AREA_Y, forest_cfg)

    obstacle_centers_for_spawn = []
    for k in obst_positions:
        obstacle_centers_for_spawn += obst_positions[k]

    spawn_cfg = CONFIG["spawn"]
    obstacle_clear = float(spawn_cfg["obstacle_clearance"])
    victim_clear = float(spawn_cfg["victim_clearance"])

    # -----------------------------
    # Randomize victim each run
    # -----------------------------
    victim_margin = 1.0
    victim_xy = sample_safe_victim_xy(
        rng=rng,
        area_x=AREA_X,
        area_y=AREA_Y,
        margin=victim_margin,
        obstacle_centers=obstacle_centers_for_spawn,
        obstacle_clear=obstacle_clear,
        max_tries=5000,
    )

    CONFIG["victim"]["pos"] = np.array([victim_xy[0], victim_xy[1], 0.0], dtype=float)
    victim_center_for_spawn = [victim_xy.copy()]

    INIT_XYZS = np.zeros((num_drones, 3))
    if spawn_cfg["enabled"]:
        init_xy = sample_safe_initial_xy(
            num_drones, rng, AREA_X, AREA_Y,
            spawn_cfg, obstacle_centers_for_spawn, obstacle_clear,
            victim_center_for_spawn, victim_clear,
        )
    else:
        init_xy = np.stack(
            [
                np.array(
                    [rng.uniform(AREA_X[0] * 0.7, AREA_X[1] * 0.7),
                     rng.uniform(AREA_Y[0] * 0.7, AREA_Y[1] * 0.7)],
                    dtype=float,
                )
                for _ in range(num_drones)
            ],
            axis=0,
        )

    for j in range(num_drones):
        INIT_XYZS[j, :] = np.array([init_xy[j, 0], init_xy[j, 1], DRONE_Z], dtype=float)

    INIT_RPYS = np.zeros((num_drones, 3))

    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )

    PYB_CLIENT = env.getPyBulletClient()

    # -----------------------------
    # Shared state
    # -----------------------------
    comm = {
        "victim_found": False,
        "rescuer_idx": None,
        "victim_pos_est": None,
        "marker_ids": [],
        "drone_body_ids": [],
        "marker_height": 0.35,
        "blink_enabled": True,
        "marker_base_colors": [],
    }

    # required metric
    ttf_sec = None  # time-to-find in seconds

    # =========================================================
    # HIGH VISIBILITY DRONES (colors + realistic beacon markers)
    # =========================================================
    HIGH_VIS_COLORS = [
        [1.0, 0.2, 0.2, 1.0],   # red
        [0.2, 1.0, 0.2, 1.0],   # green
        [0.2, 0.6, 1.0, 1.0],   # cyan-blue
        [1.0, 1.0, 0.2, 1.0],   # yellow
        [1.0, 0.2, 1.0, 1.0],   # magenta
    ]

    drone_body_ids = _get_drone_body_ids_from_env(env)
    if not drone_body_ids:
        drone_body_ids = list(getattr(env, "DRONE_IDS", []))
    comm["drone_body_ids"] = list(drone_body_ids)

    #recolor drones (all links)
    for i, body_id in enumerate(comm["drone_body_ids"]):
        color = HIGH_VIS_COLORS[i % len(HIGH_VIS_COLORS)]
        try:
            n_links = p.getNumJoints(body_id, physicsClientId=PYB_CLIENT)
            for link in range(-1, n_links):
                p.changeVisualShape(body_id, link, rgbaColor=color, physicsClientId=PYB_CLIENT)
        except Exception:
            pass

    MAST_H = 0.28
    MAST_R = 0.015
    BEACON_R = 0.06

    MARKER_HEIGHT = 0.45  # above drone
    comm["marker_height"] = MARKER_HEIGHT

    marker_ids = []
    marker_base_colors = []

    for i, body_id in enumerate(comm["drone_body_ids"]):
        base_color = HIGH_VIS_COLORS[i % len(HIGH_VIS_COLORS)]
        marker_base_colors.append(base_color)

        mast_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=MAST_R,
            length=MAST_H,
            rgbaColor=[0.08, 0.08, 0.08, 1.0],
            physicsClientId=PYB_CLIENT,
        )
        beacon_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=BEACON_R,
            rgbaColor=base_color,
            physicsClientId=PYB_CLIENT,
        )

        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0],
            linkMasses=[0, 0],
            linkCollisionShapeIndices=[-1, -1],
            linkVisualShapeIndices=[mast_vis, beacon_vis],
            linkPositions=[
                [0.0, 0.0, MAST_H * 0.5],
                [0.0, 0.0, MAST_H + BEACON_R],
            ],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]],
            physicsClientId=PYB_CLIENT,
        )
        marker_ids.append(marker_id)

    comm["marker_ids"] = marker_ids
    comm["marker_base_colors"] = marker_base_colors

    if gui:
        p.addUserDebugLine([AREA_X[0], AREA_Y[0], 0.01], [AREA_X[1], AREA_Y[0], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[1], AREA_Y[0], 0.01], [AREA_X[1], AREA_Y[1], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[1], AREA_Y[1], 0.01], [AREA_X[0], AREA_Y[1], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[0], AREA_Y[1], 0.01], [AREA_X[0], AREA_Y[0], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)

    obstacle_centers = []
    if CONFIG["forest"].get("enabled", True):
        obstacle_centers = build_forest_scene_from_positions(
            PYB_CLIENT, AREA_X, AREA_Y, CONFIG["forest"], obst_positions, safe_visuals=safe_visuals
        )

    victim_id = add_humanoid_victim(PYB_CLIENT, CONFIG["victim"], safe_visuals=safe_visuals)
    victim_detect_radius = float(CONFIG["victim"].get("detect_radius_m", 0.9))

    Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder, colab=colab)
    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    # ignore victim + drones for LIDAR
    IGNORE_LIDAR_BODIES = set([victim_id] + list(comm["drone_body_ids"]))
    print("Ignoring LIDAR bodies:", IGNORE_LIDAR_BODIES)

    drone_xy = np.zeros((num_drones, 2))
    prev_drone_xy = None
    drone_vel = np.zeros((num_drones, 2))
    action = np.zeros((num_drones, 4))

    explore_cfg = CONFIG["exploration"]
    lidar_cfg = CONFIG["lidar"]
    lidar_safety = CONFIG["lidar_safety"]
    orca_cfg = CONFIG["orca"]
    boundary_cfg = CONFIG["boundary"]
    safety_cfg = CONFIG["safety"]
    stab_cfg = CONFIG.get("stability", {})
    rescue_cfg = CONFIG.get("rescue", {})

    dt = env.CTRL_TIMESTEP
    exec_max_speed = MAX_STEP / max(dt, 1e-6)
    orca_max_speed = float(orca_cfg["orca_planning_speed_mps"])

    A_MAX = float(stab_cfg.get("a_max_mps2", 10.0))
    NEAR_GROUND_Z = float(stab_cfg.get("near_ground_z", 0.35))
    RECOVER_Z_BOOST = float(stab_cfg.get("recover_z_boost", 0.8))

    too_close_count = np.zeros(num_drones, dtype=int)
    unstick_timer = np.zeros(num_drones, dtype=int)
    braking = np.zeros(num_drones, dtype=bool)

    log_cfg = CONFIG.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", True))
    log_every_sec = float(log_cfg.get("print_every_sec", 1.0))
    log_idx = int(log_cfg.get("print_drone_idx", 0))
    log_all = bool(log_cfg.get("print_all_drones", True))
    log_rescue = bool(log_cfg.get("print_rescue_status", True))

    also_close_r = float(log_cfg.get("also_close_radius_m", 1.25))
    very_close_r = float(log_cfg.get("very_close_radius_m", 0.60))
    log_victim_dists = bool(log_cfg.get("print_victim_distances", True))

    log_every_steps = max(1, int(round(log_every_sec * env.CTRL_FREQ)))
    victim_min_dist = np.full(num_drones, np.inf, dtype=float)

    START = time.time()
    print("=== START ===")
    print(f"fast={fast} gui={gui} safe_visuals={safe_visuals} num_drones={num_drones}")
    print(f"CTRL dt={dt:.4f}s | MAX_STEP={MAX_STEP:.2f}m | exec_max_speed≈{exec_max_speed:.2f}m/s")
    print(f"ORCA max_speed(planning)={orca_max_speed:.2f}m/s")
    print(f"Accel limit A_MAX={A_MAX:.1f} m/s^2 | near_ground_z={NEAR_GROUND_Z:.2f}m")
    print("Spawn XY:\n", INIT_XYZS[:, :2])
    print("Victim XY:", victim_xy)
    print(f"Victim: {'CAPSULE' if safe_visuals else CONFIG['victim'].get('urdf')} | detect_radius={victim_detect_radius:.2f}m")
    print(f"LIDAR: rays={lidar_cfg['num_rays']} range={lidar_cfg['max_range']} z_offsets={lidar_cfg.get('z_offsets')} lidar_every={lidar_every}")
    print(f"[RUN] run_id={run_id} log_csv={log_csv}")

    lidar_hits_per_drone = [[] for _ in range(num_drones)]
    goals_xy_filt = None
    cmd_vel = np.zeros((num_drones, 2), dtype=float)
    centroids_xy_cache = None

    targets = np.zeros((num_drones, 3))
    goals_xy = np.zeros((num_drones, 2))
    pref_vels = np.zeros((num_drones, 2))

    # initial step (hover)
    obs, _, _, _, _ = env.step(np.zeros((num_drones, 4)))
    for j in range(num_drones):
        hover_target = np.array([obs[j][0], obs[j][1], SEARCH_ALTITUDE], dtype=float)
        action[j, :], _, _ = ctrl[j].computeControlFromState(
            control_timestep=dt,
            state=obs[j],
            target_pos=hover_target,
            target_rpy=np.zeros(3),
        )

    # -----------------------------
    # Main loop
    # -----------------------------
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"[WARN] env.step failed ({e}). Exiting loop.")
            break

        t_sim = i / env.CTRL_FREQ

        for j in range(num_drones):
            drone_xy[j, 0] = obs[j][0]
            drone_xy[j, 1] = obs[j][1]

        # --- Update beacon markers to follow drones (+ optional strobe blink) ---
        if comm.get("marker_ids"):
            mh = float(comm.get("marker_height", 0.35))
            blink_on = True
            if comm.get("blink_enabled", True):
                blink_on = ((i // 10) % 2) == 0

            for j, marker_id in enumerate(comm["marker_ids"]):
                pos = obs[j][0:3]
                p.resetBasePositionAndOrientation(
                    marker_id,
                    [float(pos[0]), float(pos[1]), float(pos[2]) + mh],
                    [0, 0, 0, 1],
                    physicsClientId=PYB_CLIENT,
                )

                if "marker_base_colors" in comm:
                    base = comm["marker_base_colors"][j]
                    rgba = [base[0], base[1], base[2], 1.0] if blink_on else [base[0], base[1], base[2], 0.12]
                    try:
                        p.changeVisualShape(marker_id, 1, rgbaColor=rgba, physicsClientId=PYB_CLIENT)
                    except Exception:
                        pass

        # velocities
        if prev_drone_xy is None:
            drone_vel[:] = 0.0
        else:
            drone_vel = (drone_xy - prev_drone_xy) / max(dt, 1e-6)
        prev_drone_xy = drone_xy.copy()

        # lidar update
        if lidar_cfg["enabled"] and (i % max(1, int(lidar_every)) == 0):
            lidar_hits_per_drone = [
                lidar_scan_2d_multi_z(obs[j][0:3], PYB_CLIENT, lidar_cfg, ignore_body_ids=IGNORE_LIDAR_BODIES)
                for j in range(num_drones)
            ]

        # ---- detection ----
        if not comm["victim_found"]:
            for j in range(num_drones):
                found, vxy = victim_found_by_distance(drone_xy[j], victim_id, PYB_CLIENT, victim_detect_radius)
                if found:
                    comm["victim_found"] = True
                    comm["rescuer_idx"] = j
                    comm["victim_pos_est"] = np.array(
                        [vxy[0], vxy[1], float(rescue_cfg.get("hover_altitude", 1.0))],
                        dtype=float,
                    )

                    if ttf_sec is None:
                        ttf_sec = float(t_sim)

                    print(f"\n*** Drone {j} FOUND HUMAN at t={t_sim:.2f}s | victim_xy={vxy} ***")

                    rescuer_body = comm.get("drone_body_ids", [])[j] if comm.get("drone_body_ids") else None
                    if rescuer_body is not None:
                        try:
                            n_links = p.getNumJoints(rescuer_body, physicsClientId=PYB_CLIENT)
                            for link in range(-1, n_links):
                                p.changeVisualShape(rescuer_body, link, rgbaColor=[1, 1, 1, 1], physicsClientId=PYB_CLIENT)
                        except Exception:
                            pass

                    if comm.get("marker_ids") and j < len(comm["marker_ids"]):
                        try:
                            comm["marker_base_colors"][j] = [1.0, 1.0, 1.0, 1.0]
                            p.changeVisualShape(comm["marker_ids"][j], 1, rgbaColor=[1, 1, 1, 1], physicsClientId=PYB_CLIENT)
                        except Exception:
                            pass

                    dists_now = []
                    for k in range(num_drones):
                        dk = float(np.linalg.norm(drone_xy[k] - vxy))
                        victim_min_dist[k] = min(victim_min_dist[k], dk)
                        dists_now.append((dk, k))
                    dists_now.sort()
                    closest_str = ", ".join([f"d{kk}:{dd:.2f}m" for (dd, kk) in dists_now[:min(5, len(dists_now))]])
                    print(f"    [victim-dist@found] closest drones: {closest_str}\n")
                    break

        # =========================================================
        # Goals: Voronoi pre-found; rescue ring after found
        # =========================================================
        if not comm["victim_found"]:
            if (centroids_xy_cache is None) or (i % max(1, int(voronoi_every)) == 0):
                centroids_xy_cache = compute_voronoi_centroids_bounded(drone_xy, AREA_X, AREA_Y)
            centroids_xy = centroids_xy_cache

            for j in range(num_drones):
                current_xy = drone_xy[j]
                goal_xy = centroids_xy[j].copy()

                if np.linalg.norm(goal_xy - current_xy) < float(explore_cfg["dist_threshold"]):
                    goal_xy = current_xy + float(explore_cfg["jitter_scale"]) * rng.standard_normal(2)

                goal_xy[0] = np.clip(goal_xy[0], AREA_X[0] + 0.3, AREA_X[1] - 0.3)
                goal_xy[1] = np.clip(goal_xy[1], AREA_Y[0] + 0.3, AREA_Y[1] - 0.3)

                goals_xy[j, :] = goal_xy
        else:
            vxy = comm["victim_pos_est"][:2].copy()
            rescuer = int(comm["rescuer_idx"])

            ring_r = float(rescue_cfg.get("ring_radius", 2.8))
            ring_phase = float(rescue_cfg.get("ring_phase_deg", 0.0))
            support_min_r = float(rescue_cfg.get("support_min_radius", 0.0))

            goals_xy = rescue_update_goals_ring(
                goals_xy=goals_xy,
                found_id=rescuer,
                victim_xy=vxy,
                num_drones=num_drones,
                ring_radius=ring_r,
                phase_deg=ring_phase,
            )

            if support_min_r > 0.0:
                for j in range(num_drones):
                    if j == rescuer:
                        continue
                    vec = goals_xy[j] - vxy
                    d = np.linalg.norm(vec)
                    if d > 1e-6 and d < support_min_r:
                        goals_xy[j] = vxy + (vec / d) * support_min_r

            for j in range(num_drones):
                goals_xy[j, 0] = np.clip(goals_xy[j, 0], AREA_X[0] + 0.3, AREA_X[1] - 0.3)
                goals_xy[j, 1] = np.clip(goals_xy[j, 1], AREA_Y[0] + 0.3, AREA_Y[1] - 0.3)

        if goals_xy_filt is None:
            goals_xy_filt = goals_xy.copy()
        else:
            goals_xy_filt = goal_alpha * goals_xy_filt + (1.0 - goal_alpha) * goals_xy

        rescuer_stop_r = float(rescue_cfg.get("rescuer_stop_radius", 0.30))
        support_speed_scale = float(rescue_cfg.get("support_speed_scale", 0.65))
        rescuer_speed_scale = float(rescue_cfg.get("rescuer_speed_scale", 0.55))
        support_slot_stop_r = float(rescue_cfg.get("support_slot_stop_radius", 0.35))

        for j in range(num_drones):
            to_goal = goals_xy_filt[j] - drone_xy[j]
            d = np.linalg.norm(to_goal)

            if d < 1e-6:
                pref_vels[j] = 0.0
                continue

            base_speed = min(exec_max_speed, orca_max_speed)

            if comm["victim_found"]:
                resc = int(comm["rescuer_idx"])
                if j == resc and d <= rescuer_stop_r:
                    pref_vels[j] = 0.0
                    continue
                if j != resc and d <= support_slot_stop_r:
                    pref_vels[j] = 0.0
                    continue

                scale = rescuer_speed_scale if j == resc else support_speed_scale
                desired_speed = base_speed * float(np.clip(scale, 0.1, 1.0))
            else:
                desired_speed = base_speed

            pref_vels[j] = (to_goal / d) * desired_speed

        # =========================================================
        # Safety in velocity space (ALWAYS ON)
        # =========================================================
        for j in range(num_drones):
            v_safe = np.zeros(2, dtype=float)

            v_safe += compute_lidar_avoidance_offset(
                drone_xy,
                j,
                lidar_hits_per_drone,
                influence_radius=float(lidar_cfg["influence_radius"]),
                gain=float(lidar_cfg["gain"]),
                cap=float(lidar_cfg.get("cap", 6.0)),
            )

            if boundary_cfg["enabled"]:
                v_safe += boundary_repulsion(
                    drone_xy[j],
                    AREA_X,
                    AREA_Y,
                    margin=float(boundary_cfg["margin"]),
                    gain=float(boundary_cfg["gain"]),
                    cap=float(boundary_cfg.get("cap", 2.0)),
                )

            pref_vels[j] = pref_vels[j] + v_safe

        safe_vels = pref_vels.copy()
        if orca_cfg.get("enabled", True):
            safe_vels = orca_step_2d(
                positions=drone_xy,
                velocities=drone_vel,
                pref_velocities=pref_vels,
                agent_radius=float(orca_cfg["agent_radius"]),
                neighbor_dist=float(orca_cfg["neighbor_dist"]),
                time_horizon=float(orca_cfg["time_horizon"]),
                max_speed=float(orca_cfg["orca_planning_speed_mps"]),
                dt=dt,
            )

        if lidar_safety.get("enabled", True):
            d_stop = float(lidar_safety["d_stop"])
            d_release = float(lidar_safety.get("d_release", d_stop + 0.10))
            d_slow = float(lidar_safety["d_slow"])
            slow_scale = float(lidar_safety["slow_scale"])
            crawl_scale = float(lidar_safety.get("crawl_scale", 0.05))

            unstick_enabled = bool(lidar_safety.get("unstick_enabled", True))
            hold_steps = int(lidar_safety.get("unstick_hold_steps", 12))
            dur_steps = int(lidar_safety.get("unstick_duration_steps", 20))
            unstick_speed = float(lidar_safety.get("unstick_speed", 3.0))

            for j in range(num_drones):
                dmin = min_lidar_distance(drone_xy[j], lidar_hits_per_drone[j])

                if braking[j]:
                    if dmin > d_release:
                        braking[j] = False
                else:
                    if dmin < d_stop:
                        braking[j] = True

                if dmin < d_stop:
                    too_close_count[j] += 1
                else:
                    too_close_count[j] = 0

                if unstick_enabled and unstick_timer[j] == 0 and too_close_count[j] >= hold_steps:
                    unstick_timer[j] = dur_steps
                    too_close_count[j] = 0

                if unstick_timer[j] > 0:
                    away = closest_lidar_hit_vector(drone_xy[j], lidar_hits_per_drone[j])
                    if away is None:
                        away = np.array([1.0, 0.0], dtype=float)
                    safe_vels[j] = away * unstick_speed
                    unstick_timer[j] -= 1
                    continue

                if braking[j]:
                    safe_vels[j] *= crawl_scale
                elif dmin < d_slow:
                    safe_vels[j] *= slow_scale

        # =========================================================
        # Stability: accel limit + velocity smoothing + speed cap
        # =========================================================
        dv_max = A_MAX * dt
        for j in range(num_drones):
            dv = safe_vels[j] - cmd_vel[j]
            n = np.linalg.norm(dv)
            if n > dv_max:
                safe_vels[j] = cmd_vel[j] + (dv / n) * dv_max

        for j in range(num_drones):
            cmd_vel[j] = vel_alpha * cmd_vel[j] + (1.0 - vel_alpha) * safe_vels[j]
            spd = np.linalg.norm(cmd_vel[j])
            if spd > exec_max_speed:
                cmd_vel[j] = (cmd_vel[j] / spd) * exec_max_speed

        # =========================================================
        # dt integration + tilt scaling + altitude mode + near-ground recovery
        # =========================================================
        z_boost = float(safety_cfg.get("z_boost", 0.60))
        tilt_soft = float(safety_cfg.get("tilt_soft_rad", 0.55))
        tilt_hard = float(safety_cfg.get("tilt_hard_rad", 0.90))
        unstick_z_bump = float(lidar_safety.get("unstick_z_bump", 0.50))

        for j in range(num_drones):
            roll = float(obs[j][3])
            pitch = float(obs[j][4])
            tilt = max(abs(roll), abs(pitch))
            alt = float(obs[j][2])

            if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                s = 1.0 - (tilt - tilt_soft) / max(tilt_hard - tilt_soft, 1e-6)
                s = float(np.clip(s, 0.0, 1.0))
                cmd_vel[j] *= s

            new_xy = drone_xy[j] + cmd_vel[j] * dt

            step_vec = new_xy - drone_xy[j]
            step_len = np.linalg.norm(step_vec)
            if step_len > MAX_STEP:
                new_xy = drone_xy[j] + (step_vec / step_len) * MAX_STEP

            if comm["victim_found"]:
                if j == int(comm["rescuer_idx"]):
                    z_cmd = float(rescue_cfg.get("hover_altitude", 1.0))
                else:
                    z_cmd = float(rescue_cfg.get("support_altitude", 1.3))
            else:
                z_cmd = SEARCH_ALTITUDE

            if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                z_cmd = max(z_cmd, SEARCH_ALTITUDE + z_boost)

            if unstick_timer[j] > 0:
                z_cmd = max(z_cmd, SEARCH_ALTITUDE + unstick_z_bump)

            if alt < NEAR_GROUND_Z:
                cmd_vel[j] *= 0.0
                new_xy = drone_xy[j].copy()
                z_cmd = max(z_cmd, SEARCH_ALTITUDE + RECOVER_Z_BOOST)

            Z_MIN = DRONE_Z
            Z_MAX = 3.0
            z_cmd = float(np.clip(z_cmd, Z_MIN, Z_MAX))

            targets[j, :] = np.array([new_xy[0], new_xy[1], z_cmd], dtype=float)

        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=dt,
                state=obs[j],
                target_pos=targets[j, :],
                target_rpy=np.zeros(3),
            )

        # ---------------------------------------------------------
        # LOGGING
        # ---------------------------------------------------------
        if log_enabled and (i % log_every_steps == 0):
            if log_all:
                lines = []
                for j in range(num_drones):
                    dmin_j = min_lidar_distance(drone_xy[j], lidar_hits_per_drone[j])
                    spd_j = float(np.linalg.norm(drone_vel[j]))
                    lines.append(
                        f"d{j} xy={drone_xy[j][0]:+.2f},{drone_xy[j][1]:+.2f} "
                        f"z={obs[j][2]:.2f} spd={spd_j:.2f} dmin={dmin_j:.2f} "
                        f"brk={'1' if braking[j] else '0'} "
                        f"cmd=({cmd_vel[j][0]:+.2f},{cmd_vel[j][1]:+.2f}) "
                        f"goal=({goals_xy_filt[j][0]:+.2f},{goals_xy_filt[j][1]:+.2f})"
                    )
                print(f"t={t_sim:5.2f}s | " + " | ".join(lines))
            else:
                j = int(np.clip(log_idx, 0, num_drones - 1))
                dmin_j = min_lidar_distance(drone_xy[j], lidar_hits_per_drone[j])
                spd_j = float(np.linalg.norm(drone_vel[j]))
                print(
                    f"t={t_sim:5.2f}s | drone{j} xy={drone_xy[j]} z={obs[j][2]:.2f} "
                    f"speed={spd_j:.2f} dmin={dmin_j:.2f} brk={braking[j]} cmd_vel={cmd_vel[j]} goal={goals_xy_filt[j]}"
                )

            if comm["victim_found"] and log_rescue:
                resc = int(comm["rescuer_idx"])
                vxy = comm["victim_pos_est"][:2]
                print(f"    [rescue] rescuer=d{resc} victim_xy=({vxy[0]:+.2f},{vxy[1]:+.2f})")

            if comm["victim_found"] and log_victim_dists:
                vxy = comm["victim_pos_est"][:2]
                dists = np.zeros(num_drones, dtype=float)
                for j in range(num_drones):
                    d = float(np.linalg.norm(drone_xy[j] - vxy))
                    dists[j] = d
                    victim_min_dist[j] = min(victim_min_dist[j], d)

                resc = int(comm["rescuer_idx"])
                close_idxs = [j for j in range(num_drones) if j != resc and dists[j] <= also_close_r]
                very_close_idxs = [j for j in range(num_drones) if j != resc and dists[j] <= very_close_r]

                dist_line = " ".join([f"d{j}:{dists[j]:.2f}m" for j in range(num_drones)])
                print(f"    [victim-dist] {dist_line}")

                if len(very_close_idxs) > 0:
                    print(f"    [victim-VERY-CLOSE <= {very_close_r:.2f}m] " +
                          ", ".join([f"d{j}({dists[j]:.2f}m)" for j in very_close_idxs]))
                elif len(close_idxs) > 0:
                    print(f"    [victim-also-close <= {also_close_r:.2f}m] " +
                          ", ".join([f"d{j}({dists[j]:.2f}m)" for j in close_idxs]))
                else:
                    others = [(dists[j], j) for j in range(num_drones) if j != resc]
                    others.sort()
                    if len(others) > 0:
                        d0, j0 = others[0]
                        print(f"    [victim-nearest-support] d{j0} at {d0:.2f}m | "
                              f"min_seen={victim_min_dist[j0]:.2f}m")

        if gui and not no_sync:
            sync(i, START, env.CTRL_TIMESTEP)

    # -----------------------------
    # end-of-run: close + write metrics
    # -----------------------------
    env.close()

    found_flag = bool(comm.get("victim_found", False))
    append_run_metrics_csv(log_csv, run_id, found_flag, ttf_sec)

    print(f"\n[RUN END] found={found_flag} ttf_sec={ttf_sec} -> appended to {log_csv}")

    # If plot requested: plot from CSV (useful when you ran many runs into same CSV)
    if plot:
        plot_ttf_from_csv(log_csv)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest SAR Voronoi (fixed + stable + unstick + rescue ring slots)")

    parser.add_argument("--drone", default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument("--physics", default=DEFAULT_PHYSICS, type=Physics)

    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--record_video", default=False, type=str2bool)
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument("--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES, type=str2bool)

    parser.add_argument("--simulation_freq_hz", default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument("--control_freq_hz", default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument("--duration_sec", default=DEFAULT_DURATION_SEC, type=int)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=str2bool)

    parser.add_argument("--fast", default=True, type=str2bool)
    parser.add_argument("--render_every", default=10, type=int)
    parser.add_argument("--lidar_every", default=4, type=int)
    parser.add_argument("--camera_every", default=8, type=int)
    parser.add_argument("--no_sync", default=True, type=str2bool)

    # required metrics logging
    parser.add_argument("--log_csv", type=str, default="results_runs.csv",
                        help="CSV file to append per-run results")
    parser.add_argument("--run_id", type=int, default=-1,
                        help="Optional run id to store in CSV")

    parser.add_argument(
        "--safe_visuals",
        default=True,
        type=str2bool,
        help="Avoid textured URDFs/plane.urdf and use primitives instead (prevents OpenGL texture crashes).",
    )

    args = parser.parse_args()

    run(
        drone=args.drone,
        num_drones=args.num_drones,
        physics=args.physics,
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        user_debug_gui=args.user_debug_gui,
        obstacles=args.obstacles,
        simulation_freq_hz=args.simulation_freq_hz,
        control_freq_hz=args.control_freq_hz,
        duration_sec=args.duration_sec,
        output_folder=args.output_folder,
        colab=args.colab,
        fast=args.fast,
        render_every=args.render_every,
        lidar_every=args.lidar_every,
        camera_every=args.camera_every,
        no_sync=args.no_sync,
        safe_visuals=args.safe_visuals,
        log_csv=args.log_csv,
        run_id=args.run_id,
    )
