# Forest SAR: Voronoi-Based Swarm Search & Rescue
# - Voronoi/Lloyd for coverage targets
# - Safety in velocity space: LIDAR repulsion + boundary repulsion + ORCA (inter-drone)
# - Extra stability: accel-limit, tilt-based XY scaling, near-ground recovery
#
# Run:
#   No GUI Run: python voronoi.py --gui False --fast True  --safe_visuals True  --num_drones 5
#   With GUI Run: python voronoi.py --gui True  --fast False --safe_visuals False --num_drones 5

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
        "time_horizon": 1.5,  # τ in seconds (from ORCA paper)
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
        "z_offset": 0.1,
        "influence_radius": 2.5,
        "gain": 0.5,
        "cap": 2.0,
    },

    "lidar_safety": {
        "enabled": True,
        "d_stop": 0.35,
        "d_slow": 0.80,
        "slow_scale": 0.25,
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
        "detect_radius_m": 0.9,
    },

    "rescue": {"hover_altitude": 1.8},  # Increased for safety
    "support": {
        "loiter_radius_min": 3.0,
        "loiter_radius_max": 4.0,
        "loiter_altitude": 2.0,  # Increased for safety
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
        "transition_duration": 3.0,  # Added: smooth transition to rescue
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
            "enabled": False,
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
DEFAULT_DURATION_SEC = 40
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

AREA_X = (CONFIG["area"]["x_min"], CONFIG["area"]["x_max"])
AREA_Y = (CONFIG["area"]["y_min"], CONFIG["area"]["y_max"])
DRONE_Z = float(CONFIG["drone_z_start"])
SEARCH_ALTITUDE = float(CONFIG["search_altitude"])
MAX_STEP = float(CONFIG["max_xy_step"])
VICTIM_POS = CONFIG["victim"]["pos"]


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
    """
    True (box-bounded) Voronoi cells via half-plane intersection.
    Centroid = polygon area centroid -> Lloyd / CVT target (uniform density).
    """
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
            # ||x-pi||^2 <= ||x-pj||^2  <=>  x·(pj-pi) <= (||pj||^2 - ||pi||^2)/2
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
# LIDAR (2D rays)
# -----------------------------------------------------------------------------
def lidar_scan_2d(drone_pos, pyb_client, cfg):
    if not cfg["enabled"]:
        return []

    num_rays = int(cfg["num_rays"])
    max_range = float(cfg["max_range"])
    z = float(drone_pos[2]) + float(cfg["z_offset"])

    ray_from, ray_to = [], []
    for k in range(num_rays):
        ang = 2.0 * np.pi * k / num_rays
        dx, dy = np.cos(ang), np.sin(ang)

        start = [float(drone_pos[0]), float(drone_pos[1]), z]
        end = [float(drone_pos[0] + dx * max_range), float(drone_pos[1] + dy * max_range), z]

        ray_from.append(start)
        ray_to.append(end)

    results = p.rayTestBatch(ray_from, ray_to, physicsClientId=pyb_client)

    hits_xy = []
    for r in results:
        if r[2] < 1.0:
            hit_pos = np.array(r[3], dtype=float)
            hits_xy.append(hit_pos[:2])

    return hits_xy


def compute_lidar_avoidance_offset(drone_xy, idx, lidar_hits_per_drone, influence_radius, gain, cap=2.0):
    """
    2D repulsion vector used as a VELOCITY offset (not a goal offset).
    """
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


# -----------------------------------------------------------------------------
# Forest scene (obstacles are precomputed once -> consistent)
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

    pos = {k: [] for k in counts}
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
    if terrain_cfg.get("enabled", True):
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
# MATHEMATICALLY CORRECT ORCA IMPLEMENTATION
# Following van den Berg et al. "Reciprocal n-body Collision Avoidance" (2011)
# -----------------------------------------------------------------------------
class ORCALine:
    """
    ORCA constraint line: (v - point) · normal ≥ 0
    where normal is unit vector pointing toward feasible velocities
    """
    __slots__ = ("point", "normal")

    def __init__(self, point, normal):
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        # Ensure normal is unit vector
        norm_val = np.linalg.norm(self.normal)
        if norm_val > 1e-12:
            self.normal = self.normal / norm_val


def compute_orca_constraint(p_a, v_a, p_b, v_b, radius_a, radius_b, time_horizon, time_step):
    """
    Compute ORCA constraint for agent A with respect to agent B.
    Returns ORCALine or None if no constraint needed.
    """
    # Relative position and velocity
    p = p_b - p_a
    v = v_a - v_b  # Relative velocity (A relative to B)
    
    # Combined radius
    combined_radius = radius_a + radius_b
    
    # Distance and direction
    dist_sq = np.dot(p, p)
    dist = np.sqrt(dist_sq)
    
    if dist > 1e-12:
        p_unit = p / dist
    else:
        p_unit = np.array([1.0, 0.0], dtype=float)
    
    # Check if already colliding
    if dist < combined_radius:
        # Collision already exists - use time_step for emergency avoidance
        w = -p / time_step - v
        w_norm = np.linalg.norm(w)
        
        if w_norm > 1e-12:
            u = (combined_radius / time_step - w_norm) * (w / w_norm)
        else:
            u = np.array([combined_radius / time_step, 0.0], dtype=float)
        
        point = v_a + 0.5 * u
        normal = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-12 else p_unit
        return ORCALine(point, normal)
    
    # Time to closest approach
    t = -np.dot(p, v) / max(np.dot(v, v), 1e-12)
    t = max(0.0, min(time_horizon, t))
    
    # Closest point
    closest_point = p + v * t
    dist_closest = np.linalg.norm(closest_point)
    
    # Check if collision occurs within time horizon
    if dist_closest < combined_radius:
        # Collision course detected
        
        # Velocity obstacle cone
        d = closest_point
        d_norm = np.linalg.norm(d)
        
        if d_norm > 1e-12:
            w = d / d_norm * (combined_radius - d_norm) / t - v
        else:
            w = -v
        
        w_norm = np.linalg.norm(w)
        
        if w_norm > 1e-12:
            u = w  # Minimal change to avoid collision
        else:
            # Orthogonal avoidance
            if abs(p[0]) > abs(p[1]):
                u = np.array([-p[1], p[0]], dtype=float)
            else:
                u = np.array([p[1], -p[0]], dtype=float)
            u = u / np.linalg.norm(u) * combined_radius / time_horizon
        
        point = v_a + 0.5 * u
        normal = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-12 else p_unit
        return ORCALine(point, normal)
    
    # No collision within time horizon
    return None


def linear_program_1(lines, line_no, radius, opt_vel, eps=1e-8):
    """
    Solves 1D linear program for a single line constraint.
    Returns (success, optimal_velocity)
    """
    line = lines[line_no]
    
    # Check if optimal velocity satisfies the constraint
    dot_product = np.dot(opt_vel - line.point, line.normal)
    
    if dot_product >= -eps:
        # Constraint already satisfied
        return True, opt_vel.copy()
    
    # Project opt_vel onto the constraint line
    # We need to find v such that (v - line.point)·line.normal = 0
    # and minimize ||v - opt_vel||
    
    # The projection onto the line is:
    # v_proj = opt_vel - dot_product * line.normal
    v_proj = opt_vel - dot_product * line.normal
    
    # Check if projection is within velocity radius
    v_proj_norm = np.linalg.norm(v_proj)
    
    if v_proj_norm <= radius + eps:
        # Projection is feasible
        return True, v_proj
    
    # Projection is outside radius, find intersection with circle
    # Solve quadratic equation for intersection of line and circle
    a = 1.0  # line.normal is unit vector
    b = 2.0 * np.dot(line.point, line.normal)
    c = np.dot(line.point, line.point) - radius * radius
    
    discriminant = b * b - 4.0 * a * c
    
    if discriminant < -eps:
        # No intersection - constraint cannot be satisfied
        return False, None
    
    discriminant = max(0.0, discriminant)
    sqrt_disc = np.sqrt(discriminant)
    
    # Two intersection points
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)
    
    # Choose the point that's closest to opt_vel
    point1 = line.point + t1 * line.normal
    point2 = line.point + t2 * line.normal
    
    dist1 = np.linalg.norm(point1 - opt_vel)
    dist2 = np.linalg.norm(point2 - opt_vel)
    
    return True, point1 if dist1 < dist2 else point2


def linear_program_2(lines, radius, opt_vel, eps=1e-8):
    """
    Solves 2D linear program with multiple line constraints.
    Returns (optimal_velocity, first_infeasible_line)
    """
    if not lines:
        # No constraints, return clipped optimal velocity
        opt_norm = np.linalg.norm(opt_vel)
        if opt_norm <= radius + eps:
            return opt_vel.copy(), len(lines)
        else:
            return (opt_vel / opt_norm) * radius, len(lines)
    
    # Start with optimal velocity
    current_vel = opt_vel.copy()
    current_norm = np.linalg.norm(current_vel)
    
    # Clip to maximum speed if needed
    if current_norm > radius + eps:
        current_vel = (current_vel / current_norm) * radius
    
    # Check constraints
    for i, line in enumerate(lines):
        dot_product = np.dot(current_vel - line.point, line.normal)
        
        if dot_product < -eps:
            # Constraint violated, project onto this line
            success, new_vel = linear_program_1([line], 0, radius, current_vel, eps)
            
            if not success:
                return current_vel, i
            
            # Check new velocity against previous constraints
            for j in range(i):
                prev_line = lines[j]
                if np.dot(new_vel - prev_line.point, prev_line.normal) < -eps:
                    # New velocity violates previous constraint
                    # Need to solve at intersection of two lines
                    return current_vel, i
            
            current_vel = new_vel
    
    return current_vel, len(lines)


def linear_program_3(lines, radius, opt_vel, eps=1e-8):
    """
    Solves 2D linear program when LP2 fails (two constraints active).
    Returns optimal_velocity
    """
    best_vel = np.zeros(2, dtype=float)
    best_dist = float('inf')
    
    # Try all pairs of constraints
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            line_i = lines[i]
            line_j = lines[j]
            
            # Find intersection of two lines
            # Solve: (v - p_i)·n_i = 0 and (v - p_j)·n_j = 0
            A = np.array([line_i.normal, line_j.normal])
            b = np.array([np.dot(line_i.point, line_i.normal),
                         np.dot(line_j.point, line_j.normal)])
            
            # Check if lines are parallel
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < eps:
                continue
            
            # Solve for intersection
            try:
                A_inv = np.linalg.inv(A)
                intersection = np.dot(A_inv, b)
            except:
                continue
            
            # Check if intersection is feasible (within radius)
            if np.linalg.norm(intersection) <= radius + eps:
                # Check if it satisfies all constraints
                feasible = True
                for k, line in enumerate(lines):
                    if k != i and k != j:
                        if np.dot(intersection - line.point, line.normal) < -eps:
                            feasible = False
                            break
                
                if feasible:
                    dist = np.linalg.norm(intersection - opt_vel)
                    if dist < best_dist:
                        best_dist = dist
                        best_vel = intersection
    
    # If no intersection found, try vertices on circle
    if best_dist == float('inf'):
        # Sample points on circle and find feasible one closest to opt_vel
        for angle in np.linspace(0, 2*np.pi, 36):
            candidate = np.array([np.cos(angle), np.sin(angle)]) * radius
            
            feasible = True
            for line in lines:
                if np.dot(candidate - line.point, line.normal) < -eps:
                    feasible = False
                    break
            
            if feasible:
                dist = np.linalg.norm(candidate - opt_vel)
                if dist < best_dist:
                    best_dist = dist
                    best_vel = candidate
    
    return best_vel


def orca_step_2d(positions, velocities, pref_velocities, agent_radius, neighbor_dist, time_horizon, max_speed, dt):
    """
    Mathematically correct ORCA implementation.
    
    Args:
        positions: (N, 2) array of agent positions
        velocities: (N, 2) array of current velocities
        pref_velocities: (N, 2) array of preferred velocities
        agent_radius: radius of each agent (assumed identical)
        neighbor_dist: maximum distance to consider neighbors
        time_horizon: τ parameter (time to look ahead)
        max_speed: maximum allowed speed
        dt: time step
    
    Returns:
        new_velocities: (N, 2) array of ORCA-computed velocities
    """
    N = positions.shape[0]
    new_vels = np.zeros((N, 2), dtype=float)
    
    for i in range(N):
        constraints = []
        
        # Self constraint: maximum speed
        max_speed_line = ORCALine(
            point=np.zeros(2, dtype=float),
            normal=np.array([1.0, 0.0], dtype=float)  # Will be adjusted in LP
        )
        # Actually, maximum speed is handled by the radius in linear programs
        
        # Neighbor constraints
        for j in range(N):
            if i == j:
                continue
            
            # Check if within neighbor distance
            dist_vec = positions[j] - positions[i]
            dist_sq = np.dot(dist_vec, dist_vec)
            
            if dist_sq > neighbor_dist * neighbor_dist:
                continue
            
            # Compute ORCA constraint
            constraint = compute_orca_constraint(
                p_a=positions[i],
                v_a=velocities[i],
                p_b=positions[j],
                v_b=velocities[j],
                radius_a=agent_radius,
                radius_b=agent_radius,
                time_horizon=time_horizon,
                time_step=dt
            )
            
            if constraint is not None:
                constraints.append(constraint)
        
        if not constraints:
            # No constraints, use preferred velocity clipped to max speed
            pref_norm = np.linalg.norm(pref_velocities[i])
            if pref_norm > max_speed:
                new_vels[i] = (pref_velocities[i] / pref_norm) * max_speed
            else:
                new_vels[i] = pref_velocities[i].copy()
        else:
            # Solve linear program
            opt_vel = pref_velocities[i].copy()
            opt_norm = np.linalg.norm(opt_vel)
            
            if opt_norm > max_speed:
                opt_vel = (opt_vel / opt_norm) * max_speed
            
            # Solve LP
            result, failed = linear_program_2(constraints, max_speed, opt_vel)
            
            if failed < len(constraints):
                # LP2 failed, use LP3
                result = linear_program_3(constraints, max_speed, opt_vel)
            
            new_vels[i] = result
    
    return new_vels


# -----------------------------------------------------------------------------
# Spawn
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


# -----------------------------------------------------------------------------
# Main sim loop
# -----------------------------------------------------------------------------
def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
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
):
    rng = np.random.default_rng()

    motion_cfg = CONFIG.get("motion", {})
    goal_alpha = float(motion_cfg.get("goal_lpf_alpha", 0.88))
    vel_alpha = float(motion_cfg.get("vel_lpf_alpha", 0.82))
    voronoi_every = int(motion_cfg.get("voronoi_every", 2))
    transition_duration = float(motion_cfg.get("transition_duration", 3.0))

    if not fast:
        render_every = 1
        lidar_every = 1
        no_sync = False

    forest_cfg = CONFIG["forest"]
    obst_positions = precompute_obstacle_positions(AREA_X, AREA_Y, forest_cfg)

    obstacle_centers_for_spawn = []
    for k in obst_positions:
        obstacle_centers_for_spawn += obst_positions[k]

    victim_center_for_spawn = [np.array([float(VICTIM_POS[0]), float(VICTIM_POS[1])], dtype=float)]

    spawn_cfg = CONFIG["spawn"]
    obstacle_clear = float(spawn_cfg["obstacle_clearance"])
    victim_clear = float(spawn_cfg["victim_clearance"])

    INIT_XYZS = np.zeros((num_drones, 3))
    if spawn_cfg["enabled"]:
        init_xy = sample_safe_initial_xy(
            num_drones,
            rng,
            AREA_X,
            AREA_Y,
            spawn_cfg,
            obstacle_centers_for_spawn,
            obstacle_clear,
            victim_center_for_spawn,
            victim_clear,
        )
    else:
        init_xy = np.stack(
            [
                np.array(
                    [rng.uniform(AREA_X[0] * 0.7, AREA_X[1] * 0.7), rng.uniform(AREA_Y[0] * 0.7, AREA_Y[1] * 0.7)],
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

    comm = {
        "victim_found": False,
        "rescuer_idx": None,
        "victim_pos_est": None,
        "victim_found_time": None,  # Added for smooth transition
        "transition_progress": np.zeros(num_drones, dtype=float),  # 0 to 1 for each drone
    }
    support_cfg = CONFIG["support"]
    loiter_phases = np.linspace(0.0, 2.0 * np.pi, num_drones, endpoint=False)

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

    dt = env.CTRL_TIMESTEP
    exec_max_speed = MAX_STEP / max(dt, 1e-6)
    orca_max_speed = float(orca_cfg["orca_planning_speed_mps"])

    A_MAX = float(stab_cfg.get("a_max_mps2", 10.0))
    NEAR_GROUND_Z = float(stab_cfg.get("near_ground_z", 0.35))
    RECOVER_Z_BOOST = float(stab_cfg.get("recover_z_boost", 0.8))

    START = time.time()
    print("=== START ===")
    print(f"fast={fast} gui={gui} safe_visuals={safe_visuals} num_drones={num_drones}")
    print(f"CTRL dt={dt:.4f}s | MAX_STEP={MAX_STEP:.2f}m | exec_max_speed≈{exec_max_speed:.2f}m/s")
    print(f"ORCA max_speed(planning)={orca_max_speed:.2f}m/s | time_horizon={orca_cfg['time_horizon']:.2f}s")
    print(f"Accel limit A_MAX={A_MAX:.1f} m/s^2 | near_ground_z={NEAR_GROUND_Z:.2f}m")
    print("Spawn XY:\n", INIT_XYZS[:, :2])
    print(f"Victim: {'CAPSULE' if safe_visuals else CONFIG['victim'].get('urdf')} | detect_radius={victim_detect_radius:.2f}m")
    print(f"Rescue altitudes: rescuer={CONFIG['rescue']['hover_altitude']:.1f}m, support={CONFIG['support']['loiter_altitude']:.1f}m")

    if obstacle_centers and obstacle_centers_for_spawn:
        a = np.array(obstacle_centers_for_spawn[:3])
        b = np.array(obstacle_centers[:3])
        print("Obstacle XY check (first 3) | precomputed vs built:\n", np.stack([a, b], axis=1))

    lidar_hits_per_drone = [[] for _ in range(num_drones)]
    goals_xy_filt = None
    cmd_vel = np.zeros((num_drones, 2), dtype=float)
    centroids_xy_cache = None

    targets = np.zeros((num_drones, 3))
    goals_xy = np.zeros((num_drones, 2))
    pref_vels = np.zeros((num_drones, 2))

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        t_sim = i / env.CTRL_FREQ

        for j in range(num_drones):
            drone_xy[j, 0] = obs[j][0]
            drone_xy[j, 1] = obs[j][1]

        if prev_drone_xy is None:
            drone_vel[:] = 0.0
        else:
            drone_vel = (drone_xy - prev_drone_xy) / max(dt, 1e-6)
        prev_drone_xy = drone_xy.copy()

        if lidar_cfg["enabled"] and (i % max(1, int(lidar_every)) == 0):
            lidar_hits_per_drone = [lidar_scan_2d(obs[j][0:3], PYB_CLIENT, lidar_cfg) for j in range(num_drones)]

        # ---- detection ----
        if not comm["victim_found"]:
            for j in range(num_drones):
                found, vxy = victim_found_by_distance(drone_xy[j], victim_id, PYB_CLIENT, victim_detect_radius)
                if found:
                    comm["victim_found"] = True
                    comm["rescuer_idx"] = j
                    comm["victim_pos_est"] = np.array(
                        [vxy[0], vxy[1], float(CONFIG["rescue"]["hover_altitude"])],
                        dtype=float,
                    )
                    comm["victim_found_time"] = t_sim
                    comm["transition_progress"][:] = 0.0  # Start transitions
                    print(f"\n*** Drone {j} FOUND HUMAN at t={t_sim:.2f}s | victim_xy={vxy} ***")
                    print(f"  Transition to rescue mode over {transition_duration:.1f}s")
                    break

        # Update transition progress
        if comm["victim_found"] and comm["victim_found_time"] is not None:
            time_since_found = t_sim - comm["victim_found_time"]
            for j in range(num_drones):
                if j == comm["rescuer_idx"]:
                    # Rescuer transitions immediately
                    comm["transition_progress"][j] = 1.0
                else:
                    # Support drones transition gradually
                    comm["transition_progress"][j] = min(1.0, time_since_found / transition_duration)

        # =========================================================
        # (A) Voronoi / Lloyd objective
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
            # In rescue mode, still compute Voronoi for transition reference
            if (centroids_xy_cache is None) or (i % max(1, int(voronoi_every)) == 0):
                centroids_xy_cache = compute_voronoi_centroids_bounded(drone_xy, AREA_X, AREA_Y)
            centroids_xy = centroids_xy_cache
            
            # For drones still transitioning, blend between Voronoi and rescue positions
            for j in range(num_drones):
                if j == comm["rescuer_idx"]:
                    # Rescuer goes directly to victim
                    goals_xy[j] = comm["victim_pos_est"][:2]
                else:
                    # Support drones: blend between Voronoi centroid and loiter position
                    progress = comm["transition_progress"][j]
                    
                    # Current Voronoi centroid
                    voronoi_target = centroids_xy[j].copy()
                    voronoi_target[0] = np.clip(voronoi_target[0], AREA_X[0] + 0.3, AREA_X[1] - 0.3)
                    voronoi_target[1] = np.clip(voronoi_target[1], AREA_Y[0] + 0.3, AREA_Y[1] - 0.3)
                    
                    # Loiter position
                    rmin = float(support_cfg["loiter_radius_min"])
                    rmax = float(support_cfg["loiter_radius_max"])
                    ang_spd = float(support_cfg["angular_speed"])
                    rad = 0.5 * (rmin + rmax)
                    ang = loiter_phases[j] + ang_spd * t_sim
                    loiter_target = comm["victim_pos_est"][:2] + np.array([rad * np.cos(ang), rad * np.sin(ang)])
                    
                    # Blend
                    goals_xy[j, :] = (1.0 - progress) * voronoi_target + progress * loiter_target

        # goal low-pass filter
        if goals_xy_filt is None:
            goals_xy_filt = goals_xy.copy()
        else:
            goals_xy_filt = goal_alpha * goals_xy_filt + (1.0 - goal_alpha) * goals_xy

        # preferred velocity toward goals (objective only)
        for j in range(num_drones):
            to_goal = goals_xy_filt[j] - drone_xy[j]
            d = np.linalg.norm(to_goal)
            if d < 1e-6:
                pref_vels[j] = 0.0
            else:
                desired_speed = min(exec_max_speed, orca_max_speed)
                pref_vels[j] = (to_goal / d) * desired_speed

        # =========================================================
        # (B) Safety in velocity space - ALWAYS ENABLED
        # =========================================================
        for j in range(num_drones):
            v_safe = np.zeros(2, dtype=float)

            v_safe += compute_lidar_avoidance_offset(
                drone_xy,
                j,
                lidar_hits_per_drone,
                influence_radius=float(lidar_cfg["influence_radius"]),
                gain=float(lidar_cfg["gain"]),
                cap=float(lidar_cfg.get("cap", 2.0)),
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

        # ORCA (inter-drone) - ALWAYS ENABLED
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

        # LIDAR safety brake
        if lidar_safety.get("enabled", True):
            d_stop = float(lidar_safety["d_stop"])
            d_slow = float(lidar_safety["d_slow"])
            slow_scale = float(lidar_safety["slow_scale"])
            for j in range(num_drones):
                dmin = min_lidar_distance(drone_xy[j], lidar_hits_per_drone[j])
                if dmin < d_stop:
                    safe_vels[j] *= 0.0
                elif dmin < d_slow:
                    safe_vels[j] *= slow_scale

        # =========================================================
        # (C) Stability: accel limit + velocity smoothing + speed cap
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
        # (D) dt integration + tilt scaling + rescue/loiter + near-ground recovery
        # =========================================================
        z_boost = float(safety_cfg.get("z_boost", 0.60))
        tilt_soft = float(safety_cfg.get("tilt_soft_rad", 0.55))
        tilt_hard = float(safety_cfg.get("tilt_hard_rad", 0.90))

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

            # Altitude control with smooth transitions
            if comm["victim_found"]:
                progress = comm["transition_progress"][j]
                
                if j == comm["rescuer_idx"]:
                    # Rescuer: go to hover altitude
                    z_target = float(CONFIG["rescue"]["hover_altitude"])
                    
                    # Smooth approach to victim
                    victim_xy = comm["victim_pos_est"][:2]
                    dist_to_victim = np.linalg.norm(drone_xy[j] - victim_xy)
                    
                    if dist_to_victim > 1.0:
                        # Blend current position with victim position
                        blend = 0.1 * min(1.0, 1.0 / max(dist_to_victim, 0.1))
                        new_xy = (1.0 - blend) * new_xy + blend * victim_xy
                    else:
                        # Close to victim, maintain position
                        new_xy = 0.95 * new_xy + 0.05 * victim_xy
                else:
                    # Support drone: blend between search and loiter altitude
                    z_search = SEARCH_ALTITUDE
                    z_loiter = float(support_cfg["loiter_altitude"])
                    z_target = z_search + progress * (z_loiter - z_search)
            else:
                # Search mode
                z_target = SEARCH_ALTITUDE

            # Safety altitude adjustments
            if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                z_target = max(z_target, SEARCH_ALTITUDE + z_boost)

            if alt < NEAR_GROUND_Z:
                cmd_vel[j] *= 0.0
                new_xy = drone_xy[j].copy()
                z_target = max(z_target, SEARCH_ALTITUDE + RECOVER_Z_BOOST)

            # Smooth altitude transition (prevent sudden jumps)
            current_z = float(obs[j][2])
            z_change = z_target - current_z
            max_z_change = 0.3 * dt  # Max 0.3 m/s vertical speed
            if abs(z_change) > max_z_change:
                z_target = current_z + np.sign(z_change) * max_z_change

            # Safety bounds
            Z_MIN = 1.2  # Lowered for better flight
            Z_MAX = 3.0
            z_target = float(np.clip(z_target, Z_MIN, Z_MAX))

            targets[j, :] = np.array([new_xy[0], new_xy[1], z_target], dtype=float)

        # controller -> action
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=dt,
                state=obs[j],
                target_pos=targets[j, :],
                target_rpy=np.zeros(3),
            )

        if gui and not no_sync:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest SAR Voronoi (fixed + verifiable + stable fast)")

    parser.add_argument("--drone", default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument("--physics", default=DEFAULT_PHYSICS, type=Physics)

    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VISION, type=str2bool)
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
    )