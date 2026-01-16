# ============================================================
# Forest Search-and-Rescue (SAR): Leader–Follower Swarm
# with ORCA-based Collision Avoidance and LIDAR Sensing
#
# Overview:
#   - A leader–follower coordination strategy for multi-drone
#     search-and-rescue in cluttered environments.
#   - Followers track dynamically defined formation goals
#     relative to a designated leader.
#
# Safety and Motion Handling:
#   - ORCA is used for inter-drone collision avoidance.
#   - LIDAR-based sensing influences local motion to account
#     for nearby obstacles.
#   - Motion commands are integrated over time to ensure
#     physically consistent execution.
#
# Formation Behavior:
#   - The follower formation adapts smoothly in the presence
#     of obstacles while remaining coupled to the leader.
#   - Interaction between formation control and safety
#     mechanisms is handled continuously.
#
# Run:
#   python leader-follower.py --gui False --fast True --safe_visuals True --num_drones 5
#   python leader-follower.py --gui True  --fast True --safe_visuals True --num_drones 5
# ============================================================


import time
import argparse
import numpy as np
import pybullet as p
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool


# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    # Area of operation
    "area": {"x_min": -6.0, "x_max": 6.0, "y_min": -6.0, "y_max": 6.0},

    # Drones
    "num_drones": 5,
    "drone_z_start": 0.9,
    "search_altitude": 1.7,

    # Nominal motion
    "max_xy_step": 0.12,          # max XY step per control tick (meters)
    "vel_lookahead_sec": 0.25,    # kept for compatibility; we won't "teleport" with it

    # ORCA (inter-drone only)
    "orca": {
        "enabled": True,
        "agent_radius": 0.33,
        "neighbor_dist": 4.0,
        "time_horizon": 4.0,
        "orca_planning_speed_mps": 5.5,  # will be clamped to exec_max_speed in code
    },

    # Boundary repulsion
    "boundary": {
        "enabled": True,
        "margin": 1.4,
        "gain": 1.0,
        "cap": 2.5,
    },

    # LIDAR (2D rays)
    "lidar": {
        "enabled": True,
        "num_rays": 28,
        "max_range": 4.2,
        "z_offset": 0.6,
        "influence_radius": 3.0,
        "gain": 1.05,
        "cap": 3.0,
    },

    # LIDAR safety brake zone
    "lidar_safety": {
        "enabled": True,
        "d_stop": 0.80,
        "d_slow": 1.80,
        "slow_scale": 0.08,
    },

    # Victim
    "victim": {
        "pos": np.array([0.0, 0.0, 0.0], dtype=float),
        "type": "multi",  # "multi", "urdf", "mjcf"
        "urdf": "humanoid/humanoid.urdf",
        "use_fixed_base": True,
        "mjcf": "mjcf/humanoid.xml",
        "scale": 0.12,
        "z_offset": 0.02,
        "upright_euler": [1.5708, 0.0, 0.0],
        "yaw_deg": 25.0,
        "rgba": [1.0, 0.0, 0.0, 1.0],
        "detect_radius_m": 1.0,
    },

    # Post detection
    "rescue": {"hover_altitude": 1.2},
    "support": {
        "loiter_radius_min": 3.0,
        "loiter_radius_max": 4.2,
        "loiter_altitude": 1.8,
        "angular_speed": 0.3
    },

    # Exploration jitter
    "exploration": {"dist_threshold": 0.06, "jitter_scale": 0.35},

    # Leader-Follower formation
    "leader_follower": {
        "enabled": True,
        "leader_idx": 0,
        "formation_radius": 2.8,
        "formation_rotation_rate": 0.0,
        "leader_waypoint_margin": 1.1,
        "leader_regoal_threshold": 0.35,

        # Break/soften formation when close to obstacles
        "break_on_obstacles": True,
        "break_dmin": 1.25,                 # near obstacles, soften formation strongly
        "leader_pull_when_break": 0.35,     # mild drift toward leader when breaking
    },

    # Attitude safety governor
    "safety": {
        "enabled": True,
        "tilt_soft_rad": 0.45,
        "tilt_hard_rad": 0.75,
        "xy_soft_scale": 0.30,
        "xy_hard_scale": 0.00,
        "z_boost": 0.70
    },

    # SMOOTHING
    "motion": {
        "goal_lpf_alpha": 0.88,
        "vel_lpf_alpha": 0.84,
        "accel_limit": 4.0,
    },

    # Safe spawn
    "spawn": {
        "enabled": True,
        "max_tries": 6000,
        "boundary_margin": 1.2,
        "min_interdrone_dist": 1.3,
        "obstacle_clearance": 1.35,
        "victim_clearance": 1.1,
    },

    # FOREST SCENE
    "forest": {
        "enabled": True,
        "seed": 7,
        "terrain": {
            "enabled": True,
            "cell_size": 0.25,
            "height_scale": 0.18,
            "base_height": 0.0,
        },
        "obstacles": {
            "num_trees": 16,
            "num_rocks": 14,
            "num_logs": 10,
            "num_bushes": 12,
            "min_spacing": 1.45,
        }
    },

    # Logging
    "search_progress_every_s": 5.0,
}

# -----------------------------
# Defaults / CLI
# -----------------------------
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
DEFAULT_DEBUG_LOGS = True
DEFAULT_LOG_EVERY_S = 2.0
DEFAULT_LOG_PER_DRONE = False

AREA_X = (CONFIG["area"]["x_min"], CONFIG["area"]["x_max"])
AREA_Y = (CONFIG["area"]["y_min"], CONFIG["area"]["y_max"])
DRONE_Z = float(CONFIG["drone_z_start"])
SEARCH_ALTITUDE = float(CONFIG["search_altitude"])
MAX_STEP = float(CONFIG["max_xy_step"])
VICTIM_POS = CONFIG["victim"]["pos"]


# ==============================================================================
# Small vector utils
# ==============================================================================
def _norm2(v):
    return float(np.linalg.norm(v))

def _cap_vec(v, cap):
    n = _norm2(v)
    if cap is not None and cap > 0.0 and n > cap:
        return (v / n) * cap
    return v

def _rand_xy(rng, x_range, y_range, margin=1.0):
    return np.array([
        rng.uniform(x_range[0] + margin, x_range[1] - margin),
        rng.uniform(y_range[0] + margin, y_range[1] - margin),
    ], dtype=float)


# ==============================================================================
# Victim helpers (MULTI-PIECE "human" + URDF + MJCF)
# ==============================================================================
def _weld(client_id, parent_id, child_id, parent_frame_pos, child_frame_pos):
    p.createConstraint(
        parentBodyUniqueId=parent_id,
        parentLinkIndex=-1,
        childBodyUniqueId=child_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 1],
        parentFramePosition=parent_frame_pos,
        childFramePosition=child_frame_pos,
        physicsClientId=client_id
    )

def add_multi_piece_victim(client_id, pos_xyz, rgba=(1, 0, 0, 1), yaw_deg=0.0):
    x, y, z = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
    yaw = np.deg2rad(float(yaw_deg))
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    torso_r = 0.14
    torso_h = 0.55
    torso_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=torso_r, height=torso_h, physicsClientId=client_id)
    torso_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=torso_r, length=torso_h, rgbaColor=list(rgba), physicsClientId=client_id)
    torso_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=torso_col,
        baseVisualShapeIndex=torso_vis,
        basePosition=[x, y, z + 0.65],
        baseOrientation=orn,
        physicsClientId=client_id
    )

    head_r = 0.12
    head_col = p.createCollisionShape(p.GEOM_SPHERE, radius=head_r, physicsClientId=client_id)
    head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=head_r, rgbaColor=list(rgba), physicsClientId=client_id)
    head_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=head_col,
        baseVisualShapeIndex=head_vis,
        basePosition=[x, y, z + 1.05],
        baseOrientation=orn,
        physicsClientId=client_id
    )
    _weld(client_id, torso_id, head_id, parent_frame_pos=[0, 0, 0.40], child_frame_pos=[0, 0, 0.0])

    legs_r = 0.11
    legs_h = 0.55
    legs_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=legs_r, height=legs_h, physicsClientId=client_id)
    legs_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=legs_r, length=legs_h, rgbaColor=list(rgba), physicsClientId=client_id)
    legs_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=legs_col,
        baseVisualShapeIndex=legs_vis,
        basePosition=[x, y, z + 0.25],
        baseOrientation=orn,
        physicsClientId=client_id
    )
    _weld(client_id, torso_id, legs_id, parent_frame_pos=[0, 0, -0.42], child_frame_pos=[0, 0, 0.0])

    return torso_id

def set_humanoid_standing_pose(client_id, humanoid_id):
    nJ = p.getNumJoints(humanoid_id, physicsClientId=client_id)
    for j in range(nJ):
        try:
            p.resetJointState(humanoid_id, j, targetValue=0.0, physicsClientId=client_id)
        except Exception:
            pass

    def safe_set(j, val):
        if 0 <= j < nJ:
            try:
                p.resetJointState(humanoid_id, j, targetValue=val, physicsClientId=client_id)
            except Exception:
                pass

    for j in range(nJ):
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

def add_urdf_victim(client_id, victim_cfg):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

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
            physicsClientId=client_id
        )
    except Exception as e:
        print(f"[WARN] Could not load URDF '{urdf_path}' ({e}). Falling back to multi-piece victim.")
        return add_multi_piece_victim(client_id, victim_cfg["pos"], rgba=rgba, yaw_deg=victim_cfg.get("yaw_deg", 0.0))

    set_humanoid_standing_pose(client_id, victim_id)

    try:
        for link in range(-1, p.getNumJoints(victim_id, physicsClientId=client_id)):
            p.changeVisualShape(victim_id, link, rgbaColor=rgba, physicsClientId=client_id)
    except Exception:
        pass

    return victim_id

def add_mjcf_victim(client_id, victim_cfg):
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    mjcf_path = str(victim_cfg.get("mjcf", "mjcf/humanoid.xml"))
    rgba = victim_cfg.get("rgba", [1.0, 0.0, 0.0, 1.0])

    pos = victim_cfg["pos"].astype(float).tolist()
    pos[2] = float(pos[2]) + float(victim_cfg.get("z_offset", 0.0)) + 0.02
    yaw = np.deg2rad(float(victim_cfg.get("yaw_deg", 0.0)))
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])

    try:
        bodies = p.loadMJCF(mjcf_path, physicsClientId=client_id)
        victim_id = int(bodies[0])
        p.resetBasePositionAndOrientation(victim_id, pos, orn, physicsClientId=client_id)

        try:
            for bid in bodies:
                for link in range(-1, p.getNumJoints(bid, physicsClientId=client_id)):
                    p.changeVisualShape(bid, link, rgbaColor=rgba, physicsClientId=client_id)
        except Exception:
            pass

        return victim_id
    except Exception as e:
        print(f"[WARN] Could not load MJCF '{mjcf_path}' ({e}). Falling back to multi-piece victim.")
        return add_multi_piece_victim(client_id, victim_cfg["pos"], rgba=rgba, yaw_deg=victim_cfg.get("yaw_deg", 0.0))

def add_victim(client_id, victim_cfg, safe_visuals=False):
    vtype = str(victim_cfg.get("type", "multi")).lower()
    rgba = victim_cfg.get("rgba", [1.0, 0.0, 0.0, 1.0])
    yaw_deg = float(victim_cfg.get("yaw_deg", 0.0))

    if safe_visuals:
        return add_multi_piece_victim(client_id, victim_cfg["pos"], rgba=rgba, yaw_deg=yaw_deg)

    if vtype == "urdf":
        return add_urdf_victim(client_id, victim_cfg)
    if vtype == "mjcf":
        return add_mjcf_victim(client_id, victim_cfg)

    return add_multi_piece_victim(client_id, victim_cfg["pos"], rgba=rgba, yaw_deg=yaw_deg)

def victim_found_by_distance(drone_xy, victim_id, client_id, radius_m):
    vpos, _ = p.getBasePositionAndOrientation(victim_id, physicsClientId=client_id)
    vxy = np.array([vpos[0], vpos[1]], dtype=float)
    d = float(np.linalg.norm(np.asarray(drone_xy, dtype=float) - vxy))
    return (d <= float(radius_m)), vxy


# ==============================================================================
# Boundary repulsion
# ==============================================================================
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

    return _cap_vec(np.array([fx, fy], dtype=float), cap)


# ==============================================================================
# LIDAR
# ==============================================================================
def lidar_scan_2d(drone_pos, pyb_client, cfg):
    if not cfg["enabled"]:
        return []

    num_rays = int(cfg["num_rays"])
    max_range = float(cfg["max_range"])
    z = float(drone_pos[2]) + float(cfg["z_offset"])

    ray_from, ray_to = [], []
    for k in range(num_rays):
        angle = 2.0 * np.pi * k / num_rays
        dx, dy = np.cos(angle), np.sin(angle)
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

    return _cap_vec(offset, cap)

def min_lidar_distance(drone_xy, hits_xy):
    if not hits_xy:
        return np.inf
    dmin = np.inf
    for hp in hits_xy:
        d = np.linalg.norm(drone_xy - np.asarray(hp, dtype=float))
        dmin = min(dmin, d)
    return dmin


# ==============================================================================
# FOREST SCENE (SAFE VISUALS)
# ==============================================================================
def _add_box_ground_visual(client_id, area_x, area_y):
    hx = (area_x[1] - area_x[0]) * 0.5
    hy = (area_y[1] - area_y[0]) * 0.5
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, 0.01],
        rgbaColor=[0.30, 0.40, 0.28, 1.0],
        physicsClientId=client_id
    )
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                      basePosition=[0, 0, 0.0], physicsClientId=client_id)

def _add_tree_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    trunk_h = 1.1 + 0.5 * rng.random()
    trunk_r = 0.10 + 0.05 * rng.random()
    col1 = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h, physicsClientId=client_id)
    vis1 = p.createVisualShape(p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h,
                               rgbaColor=[0.45, 0.28, 0.12, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col1, vis1, basePosition=[x, y, trunk_h * 0.5], physicsClientId=client_id)

    crown_r = 0.30 + 0.18 * rng.random()
    col2 = p.createCollisionShape(p.GEOM_SPHERE, radius=crown_r, physicsClientId=client_id)
    vis2 = p.createVisualShape(p.GEOM_SPHERE, radius=crown_r,
                               rgbaColor=[0.12, 0.42, 0.14, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col2, vis2, basePosition=[x, y, trunk_h + crown_r * 1.2], physicsClientId=client_id)

def _add_rock_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    r = 0.18 + 0.22 * rng.random()
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.35, 0.35, 0.35, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col, vis, basePosition=[x, y, r], physicsClientId=client_id)

def _add_log_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    length = 1.0 + 0.8 * rng.random()
    radius = 0.10 + 0.06 * rng.random()
    yaw = float(rng.uniform(0, 180))
    orn = p.getQuaternionFromEuler([0, 0, np.deg2rad(yaw)])
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                              rgbaColor=[0.40, 0.26, 0.12, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col, vis, basePosition=[x, y, radius], baseOrientation=orn, physicsClientId=client_id)

def _add_bush_primitive(client_id, xy, rng):
    x, y = float(xy[0]), float(xy[1])
    r = 0.25 + 0.18 * rng.random()
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client_id)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.15, 0.45, 0.15, 1.0], physicsClientId=client_id)
    p.createMultiBody(0, col, vis, basePosition=[x, y, r], physicsClientId=client_id)

def build_forest_scene(client_id, area_x, area_y, forest_cfg):
    rng = np.random.default_rng(int(forest_cfg.get("seed", 0)))
    obstacle_centers = []

    terrain_cfg = forest_cfg.get("terrain", {})
    if terrain_cfg.get("enabled", True):
        cell = float(terrain_cfg.get("cell_size", 0.25))
        hx = int(max(16, round((area_x[1] - area_x[0]) / cell)))
        hy = int(max(16, round((area_y[1] - area_y[0]) / cell)))

        noise = rng.normal(0.0, 1.0, size=(hy, hx))
        for _ in range(3):
            noise = (
                noise +
                np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
                np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1)
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
            physicsClientId=client_id
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, -0.02],
            physicsClientId=client_id
        )
        _add_box_ground_visual(client_id, area_x, area_y)

    obst_cfg = forest_cfg.get("obstacles", {})
    min_spacing = float(obst_cfg.get("min_spacing", 0.95))
    placed = []

    def place_xy():
        for _ in range(6000):
            xy = _rand_xy(rng, area_x, area_y, margin=1.0)
            if _norm2(xy - np.array([0.0, 0.0])) < 1.4:
                continue
            if all(_norm2(xy - q) >= min_spacing for q in placed):
                placed.append(xy)
                return xy
        xy = _rand_xy(rng, area_x, area_y, margin=1.0)
        placed.append(xy)
        return xy

    def add_center(xy):
        obstacle_centers.append(np.array([xy[0], xy[1]], dtype=float))

    for _ in range(int(obst_cfg.get("num_trees", 16))):
        xy = place_xy()
        _add_tree_primitive(client_id, xy, rng)
        add_center(xy)

    for _ in range(int(obst_cfg.get("num_rocks", 14))):
        xy = place_xy()
        _add_rock_primitive(client_id, xy, rng)
        add_center(xy)

    for _ in range(int(obst_cfg.get("num_logs", 10))):
        xy = place_xy()
        _add_log_primitive(client_id, xy, rng)
        add_center(xy)

    for _ in range(int(obst_cfg.get("num_bushes", 12))):
        xy = place_xy()
        _add_bush_primitive(client_id, xy, rng)
        add_center(xy)

    return obstacle_centers


# ==============================================================================
# ORCA (self-contained, 2D)
# ==============================================================================
class OrcaLine:
    __slots__ = ("point", "direction")
    def __init__(self, point, direction):
        self.point = np.asarray(point, dtype=float)
        self.direction = np.asarray(direction, dtype=float)

def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

def _norm(a):
    return float(np.linalg.norm(a))

def _normalize(a, eps=1e-12):
    n = _norm(a)
    if n < eps:
        return np.zeros_like(a)
    return a / n

def _linear_program_1(lines, line_no, radius, opt_vel, direction_opt):
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
        denom = _cross2(line.direction, other.direction)
        numer = _cross2(other.direction, line.point - other.point)

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

def _linear_program_2(lines, radius, opt_vel, direction_opt):
    if direction_opt:
        result = _normalize(opt_vel) * radius
    else:
        result = _normalize(opt_vel) * radius if _norm(opt_vel) > radius else opt_vel.copy()

    for i, line in enumerate(lines):
        if _cross2(line.direction, result - line.point) < 0.0:
            success, tmp = _linear_program_1(lines, i, radius, opt_vel, direction_opt)
            if not success:
                return result, i
            result = tmp

    return result, len(lines)

def _linear_program_3(lines, begin_line, radius, result):
    distance = 0.0
    for i in range(begin_line, len(lines)):
        line = lines[i]
        if _cross2(line.direction, result - line.point) < distance:
            proj_lines = []
            for j in range(i):
                other = lines[j]
                denom = _cross2(line.direction, other.direction)
                if abs(denom) < 1e-12:
                    continue
                t = _cross2(other.direction, line.point - other.point) / denom
                pt = line.point + t * line.direction
                dir_ = _normalize(other.direction - line.direction)
                proj_lines.append(OrcaLine(pt, dir_))

            opt_dir = np.array([-line.direction[1], line.direction[0]])
            new_result, fail = _linear_program_2(proj_lines, radius, opt_dir, direction_opt=True)
            if fail == len(proj_lines):
                result = new_result

            distance = _cross2(line.direction, result - line.point)
    return result

def orca_step_2d(positions, velocities, pref_velocities, agent_radius, neighbor_dist, time_horizon, max_speed):
    N = positions.shape[0]
    new_vels = np.zeros((N, 2), dtype=float)

    inv_time_h = 1.0 / max(time_horizon, 1e-6)
    combined_radius = 2.0 * agent_radius
    combined_radius_sq = combined_radius * combined_radius

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
                    if _cross2(rel_pos, w) > 0.0:
                        direction = np.array([
                            rel_pos[0] * leg - rel_pos[1] * combined_radius,
                            rel_pos[0] * combined_radius + rel_pos[1] * leg
                        ]) / max(dist_sq, 1e-12)
                    else:
                        direction = -np.array([
                            rel_pos[0] * leg + rel_pos[1] * combined_radius,
                            -rel_pos[0] * combined_radius + rel_pos[1] * leg
                        ]) / max(dist_sq, 1e-12)
                    direction = _normalize(direction)
                    u = (np.dot(rel_vel, direction) - np.dot(inv_time_h * rel_pos, direction)) * direction - rel_vel
            else:
                inv_time_step = 1.0 / 0.05
                #inv_time_step = 1.0 / max(dt, 1e-6)
                w = rel_vel - inv_time_step * rel_pos
                w_len = _norm(w)
                unit_w = w / max(w_len, 1e-12)
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_step - w_len) * unit_w

            line_point = v_i + 0.5 * u
            lines.append(OrcaLine(line_point, direction))

        pref = pref_velocities[i].copy()
        if _norm(pref) > max_speed:
            pref = _normalize(pref) * max_speed

        result, failed = _linear_program_2(lines, max_speed, pref, direction_opt=False)
        if failed < len(lines):
            result = _linear_program_3(lines, failed, max_speed, result)

        new_vels[i] = result

    return new_vels


# ==============================================================================
# Safe spawn (rejection sampling)
# ==============================================================================
def sample_safe_initial_xy(num_drones, rng, area_x, area_y, spawn_cfg,
                          obstacle_centers, obstacle_clear,
                          victim_centers, victim_clear):
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

        ok = True
        for prev in pts:
            if _norm2(cand - prev) < min_d:
                ok = False
                break
        if not ok:
            continue

        for oc in obstacle_centers:
            if _norm2(cand - oc) < obstacle_clear:
                ok = False
                break
        if not ok:
            continue

        for vc in victim_centers:
            if _norm2(cand - vc) < victim_clear:
                ok = False
                break
        if not ok:
            continue

        pts.append(cand)

    if len(pts) < num_drones:
        pts = []
        for _ in range(num_drones):
            pts.append(np.array([rng.uniform(bx0 * 0.7, bx1 * 0.7),
                                 rng.uniform(by0 * 0.7, by1 * 0.7)], dtype=float))
    return np.stack(pts, axis=0)


# ==============================================================================
# Main
# ==============================================================================
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
    lidar_every=3,
    no_sync=True,
    safe_visuals=True,
    debug_logs=DEFAULT_DEBUG_LOGS,
    log_every_s=DEFAULT_LOG_EVERY_S,
    log_per_drone=DEFAULT_LOG_PER_DRONE,
):
    rng = np.random.default_rng()

    motion_cfg = CONFIG.get("motion", {})
    goal_alpha = float(motion_cfg.get("goal_lpf_alpha", 0.88))
    vel_alpha = float(motion_cfg.get("vel_lpf_alpha", 0.84))

    if not fast:
        render_every = 1
        lidar_every = 1
        no_sync = False

    # preview obstacles for safe spawn
    forest_cfg = CONFIG["forest"]
    rng_spawn = np.random.default_rng(int(forest_cfg.get("seed", 0)))
    obst_cfg = forest_cfg.get("obstacles", {})
    n_total = (
        int(obst_cfg.get("num_trees", 16)) +
        int(obst_cfg.get("num_rocks", 14)) +
        int(obst_cfg.get("num_logs", 10)) +
        int(obst_cfg.get("num_bushes", 12))
    )
    min_spacing = float(obst_cfg.get("min_spacing", 0.95))

    obstacle_centers_preview = []
    placed = []

    def place_xy_preview():
        for _ in range(6000):
            xy = _rand_xy(rng_spawn, AREA_X, AREA_Y, margin=1.0)
            if _norm2(xy - np.array([0.0, 0.0])) < 1.4:
                continue
            if all(_norm2(xy - q) >= min_spacing for q in placed):
                placed.append(xy)
                return xy
        xy = _rand_xy(rng_spawn, AREA_X, AREA_Y, margin=1.0)
        placed.append(xy)
        return xy

    for _ in range(n_total):
        obstacle_centers_preview.append(place_xy_preview())

    victim_center_for_spawn = [np.array([float(VICTIM_POS[0]), float(VICTIM_POS[1])], dtype=float)]

    spawn_cfg = CONFIG["spawn"]
    obstacle_clear = float(spawn_cfg["obstacle_clearance"])
    victim_clear = float(spawn_cfg["victim_clearance"])

    INIT_XYZS = np.zeros((num_drones, 3))
    if spawn_cfg["enabled"]:
        init_xy = sample_safe_initial_xy(
            num_drones, rng, AREA_X, AREA_Y, spawn_cfg,
            obstacle_centers_preview, obstacle_clear,
            victim_center_for_spawn, victim_clear
        )
    else:
        init_xy = np.stack([
            np.array([rng.uniform(AREA_X[0] * 0.7, AREA_X[1] * 0.7),
                      rng.uniform(AREA_Y[0] * 0.7, AREA_Y[1] * 0.7)], dtype=float)
            for _ in range(num_drones)
        ], axis=0)

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
        user_debug_gui=user_debug_gui
    )
    PYB_CLIENT = env.getPyBulletClient()

    if gui:
        p.addUserDebugLine([AREA_X[0], AREA_Y[0], 0.01], [AREA_X[1], AREA_Y[0], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[1], AREA_Y[0], 0.01], [AREA_X[1], AREA_Y[1], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[1], AREA_Y[1], 0.01], [AREA_X[0], AREA_Y[1], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)
        p.addUserDebugLine([AREA_X[0], AREA_Y[1], 0.01], [AREA_X[0], AREA_Y[0], 0.01], [0, 1, 0], physicsClientId=PYB_CLIENT)

    obstacle_centers = []
    if CONFIG["forest"].get("enabled", True):
        obstacle_centers = build_forest_scene(PYB_CLIENT, AREA_X, AREA_Y, CONFIG["forest"])

    victim_id = add_victim(PYB_CLIENT, CONFIG["victim"], safe_visuals=safe_visuals)
    victim_detect_radius = float(CONFIG["victim"].get("detect_radius_m", 1.0))

    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    comm = {
        "victim_found": False,
        "rescuer_idx": None,
        "victim_pos_est": None,
        "min_dist_ever": float("inf"),
        "next_progress_print": 0.0
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
    lf_cfg = CONFIG.get("leader_follower", {})

    dt = env.CTRL_TIMESTEP
    exec_max_speed = MAX_STEP / max(dt, 1e-6)

    # IMPORTANT FIX: ORCA speed must not exceed what we can actually execute
    orca_speed_cfg = float(orca_cfg.get("orca_planning_speed_mps", exec_max_speed))
    orca_max_speed = min(orca_speed_cfg, 0.9 * exec_max_speed)

    # Leader/Follower state
    leader_idx = int(lf_cfg.get("leader_idx", 0))
    leader_idx = max(0, min(num_drones - 1, leader_idx))
    leader_goal = _rand_xy(rng, AREA_X, AREA_Y, margin=float(lf_cfg.get("leader_waypoint_margin", 1.0)))
    formation_r = float(lf_cfg.get("formation_radius", 2.8))
    rot_rate = float(lf_cfg.get("formation_rotation_rate", 0.0))
    ring_angles = np.linspace(0.0, 2.0 * np.pi, max(num_drones - 1, 1), endpoint=False)

    START = time.time()
    print("=== START ===")
    print(f"fast={fast} gui={gui} safe_visuals={safe_visuals} num_drones={num_drones}")
    print(f"CTRL dt={dt:.4f}s | MAX_STEP={MAX_STEP:.2f}m | exec_max_speed≈{exec_max_speed:.2f}m/s")
    print(f"ORCA max_speed(used)={orca_max_speed:.2f}m/s")
    print("Spawn XY:\n", INIT_XYZS[:, :2])
    print(f"Victim type={CONFIG['victim'].get('type','multi')} | detect_radius={victim_detect_radius:.2f}m")

    progress_every_s = float(CONFIG.get("search_progress_every_s", 5.0))

    lidar_hits_per_drone = [[] for _ in range(num_drones)]
    goals_xy_filt = None
    cmd_vel = np.zeros((num_drones, 2), dtype=float)

    targets = np.zeros((num_drones, 3))
    goals_xy = np.zeros((num_drones, 2))
    pref_vels = np.zeros((num_drones, 2))
    dmin_by_drone = np.full((num_drones,), np.inf, dtype=float)
    speed_by_drone = np.zeros((num_drones,), dtype=float)

    # Debug logging state
    next_log_t = 0.0

    def _maybe_log(t_sim):
        nonlocal next_log_t
        if not debug_logs:
            return
        if t_sim + 1e-9 < next_log_t:
            return
        next_log_t = t_sim + float(log_every_s)
        status = f"FOUND by {comm.get('rescuer_idx', None)}" if comm.get("victim_found", False) else "SEARCHING"
        min_d = comm.get("min_dist_ever", float("inf"))
        print(f"[t={t_sim:.1f}s] {status} | best_min_dist={min_d:.2f}m | mean_speed={float(np.mean(speed_by_drone)):.2f}m/s")
        if log_per_drone:
            for j in range(num_drones):
                z = float(obs[j][2])
                roll = float(obs[j][3]); pitch = float(obs[j][4])
                tilt = max(abs(roll), abs(pitch))
                print(f"  drone{j}: z={z:.2f} tilt={tilt:.2f} dmin={dmin_by_drone[j]:.2f} v={speed_by_drone[j]:.2f}")

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Step with previous action to get new obs
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

        # progress: how close are we getting to the victim
        if not comm["victim_found"]:
            vpos, _ = p.getBasePositionAndOrientation(victim_id, physicsClientId=PYB_CLIENT)
            vxy = np.array([vpos[0], vpos[1]], dtype=float)
            dists = np.linalg.norm(drone_xy - vxy[None, :], axis=1)
            min_dist = float(np.min(dists))
            comm["min_dist_ever"] = min(comm["min_dist_ever"], min_dist)
            if t_sim >= comm["next_progress_print"]:
                print(f"[t={t_sim:.1f}s] searching... min_dist={min_dist:.2f}m (best={comm['min_dist_ever']:.2f}m)")
                comm["next_progress_print"] = t_sim + progress_every_s

        if lidar_cfg["enabled"] and (i % max(1, int(lidar_every)) == 0):
            lidar_hits_per_drone = [lidar_scan_2d(obs[j][0:3], PYB_CLIENT, lidar_cfg) for j in range(num_drones)]

        # victim detection
        if not comm["victim_found"]:
            for j in range(num_drones):
                found, vxy = victim_found_by_distance(drone_xy[j], victim_id, PYB_CLIENT, victim_detect_radius)
                if found:
                    comm["victim_found"] = True
                    comm["rescuer_idx"] = j
                    comm["victim_pos_est"] = np.array([vxy[0], vxy[1], float(CONFIG["rescue"]["hover_altitude"])], dtype=float)
                    print(f"\n*** Drone {j} FOUND HUMAN at t={t_sim:.2f}s | victim_xy={vxy} ***")
                    break

        # ---------------------------------------------------------------------
        # GOALS (Leader-Follower) — FIXED: soften formation near obstacles, no tug-of-war
        # ---------------------------------------------------------------------
        if not comm["victim_found"]:
            leader_xy = drone_xy[leader_idx].copy()
            if np.linalg.norm(leader_goal - leader_xy) < float(lf_cfg.get("leader_regoal_threshold", 0.35)):
                leader_goal = _rand_xy(rng, AREA_X, AREA_Y, margin=float(lf_cfg.get("leader_waypoint_margin", 1.1)))

            # base ring offsets (unit circle), scaled later by formation_r_eff
            offsets_unit = np.zeros((num_drones, 2), dtype=float)
            if num_drones > 1:
                ang0 = rot_rate * t_sim
                for k in range(num_drones):
                    if k == leader_idx:
                        continue
                    kk = k - 1 if k > leader_idx else k
                    ang = ring_angles[kk] + ang0
                    offsets_unit[k, :] = np.array([np.cos(ang), np.sin(ang)], dtype=float)

            for j in range(num_drones):
                current_xy = drone_xy[j]
                dmin = min_lidar_distance(current_xy, lidar_hits_per_drone[j])
                dmin_by_drone[j] = float(dmin)

                # formation softening near obstacles (smoothly shrink radius)
                d_slow = float(lidar_safety["d_slow"])
                formation_weight = float(np.clip(dmin / max(d_slow, 1e-6), 0.0, 1.0))
                formation_r_eff = formation_r * formation_weight

                if j == leader_idx:
                    goal_xy = leader_goal.copy()
                else:
                    goal_xy = leader_xy + formation_r_eff * offsets_unit[j]

                # SINGLE obstacle avoidance injection (ONCE)
                avoid = compute_lidar_avoidance_offset(
                    drone_xy, j, lidar_hits_per_drone,
                    influence_radius=float(lidar_cfg["influence_radius"]),
                    gain=float(lidar_cfg["gain"]),
                    cap=float(lidar_cfg.get("cap", 3.0))
                )
                goal_xy += avoid

                # If very close to obstacles, optionally "break" formation behavior:
                # move primarily away from obstacles; and only gently drift toward leader.
                if (
                    bool(lf_cfg.get("break_on_obstacles", True))
                    and j != leader_idx
                    and dmin < float(lf_cfg.get("break_dmin", d_slow))
                ):
                    pull = leader_xy - current_xy

                    # FIX: disable pull when braking zone (prevents being dragged into obstacles)
                    if dmin < d_slow:
                        pull *= 0.0
                    else:
                        pull *= float(lf_cfg.get("leader_pull_when_break", 0.35))

                    goal_xy = current_xy + 1.1 * avoid + 0.10 * pull

                # jitter if stuck
                if np.linalg.norm(goal_xy - current_xy) < explore_cfg["dist_threshold"]:
                    goal_xy = current_xy + explore_cfg["jitter_scale"] * rng.standard_normal(2)

                # boundary repulsion
                if boundary_cfg["enabled"]:
                    goal_xy += boundary_repulsion(
                        current_xy, AREA_X, AREA_Y,
                        margin=float(boundary_cfg["margin"]),
                        gain=float(boundary_cfg["gain"]),
                        cap=float(boundary_cfg.get("cap", 2.5))
                    )

                goal_xy[0] = np.clip(goal_xy[0], AREA_X[0] + 0.35, AREA_X[1] - 0.35)
                goal_xy[1] = np.clip(goal_xy[1], AREA_Y[0] + 0.35, AREA_Y[1] - 0.35)

                goals_xy[j, :] = goal_xy
        else:
            goals_xy[:, :] = drone_xy[:, :]

        # goal low-pass filter
        if goals_xy_filt is None:
            goals_xy_filt = goals_xy.copy()
        else:
            goals_xy_filt = goal_alpha * goals_xy_filt + (1.0 - goal_alpha) * goals_xy

        # preferred velocity toward goal
        for j in range(num_drones):
            to_goal = goals_xy_filt[j] - drone_xy[j]
            d = np.linalg.norm(to_goal)
            if d < 1e-6:
                pref_vels[j] = 0.0
            else:
                pref_vels[j] = (to_goal / d) * min(exec_max_speed, orca_max_speed)

        # ORCA solve (inter-drone only)
        safe_vels = pref_vels.copy()
        if orca_cfg.get("enabled", True):
            safe_vels = orca_step_2d(
                positions=drone_xy,
                velocities=drone_vel,
                pref_velocities=pref_vels,
                agent_radius=float(orca_cfg["agent_radius"]),
                neighbor_dist=float(orca_cfg["neighbor_dist"]),
                time_horizon=float(orca_cfg["time_horizon"]),
                max_speed=float(orca_max_speed),
            )

        # LIDAR safety brake
        if lidar_safety.get("enabled", True):
            d_stop = float(lidar_safety["d_stop"])
            d_slow = float(lidar_safety["d_slow"])
            slow_scale = float(lidar_safety["slow_scale"])
            for j in range(num_drones):
                dmin = dmin_by_drone[j]
                if dmin < d_stop:
                    safe_vels[j] *= 0.0
                elif dmin < d_slow:
                    safe_vels[j] *= slow_scale

        # velocity low-pass + clamp
        for j in range(num_drones):
            cmd_vel[j] = vel_alpha * cmd_vel[j] + (1.0 - vel_alpha) * safe_vels[j]
            spd = float(np.linalg.norm(cmd_vel[j]))
            speed_by_drone[j] = spd
            if spd > exec_max_speed:
                cmd_vel[j] = (cmd_vel[j] / max(spd, 1e-9)) * exec_max_speed

        # ---------------------------------------------------------------------
        # POSITION TARGETS — FIXED: integrate with dt (no lookahead teleport)
        # ---------------------------------------------------------------------
        z_boost = float(safety_cfg.get("z_boost", 0.70))
        tilt_soft = float(safety_cfg.get("tilt_soft_rad", 0.45))

        for j in range(num_drones):
            roll = float(obs[j][3])
            pitch = float(obs[j][4])
            tilt = max(abs(roll), abs(pitch))

            # FIX: integrate using dt (stable)
            new_xy = drone_xy[j] + cmd_vel[j] * dt

            # still enforce MAX_STEP as a hard cap
            step_vec = new_xy - drone_xy[j]
            step_len = np.linalg.norm(step_vec)
            if step_len > MAX_STEP:
                new_xy = drone_xy[j] + (step_vec / max(step_len, 1e-9)) * MAX_STEP

            # altitude logic
            if comm["victim_found"]:
                if j == comm["rescuer_idx"]:
                    z_cmd = float(CONFIG["rescue"]["hover_altitude"])
                    comm_xy = comm["victim_pos_est"][:2].copy()
                    new_xy = 0.90 * new_xy + 0.10 * comm_xy
                else:
                    rad = 0.5 * (float(support_cfg["loiter_radius_min"]) + float(support_cfg["loiter_radius_max"]))
                    ang = loiter_phases[j] + float(support_cfg["angular_speed"]) * t_sim
                    new_xy = comm["victim_pos_est"][:2] + np.array([rad * np.cos(ang), rad * np.sin(ang)])
                    z_cmd = float(support_cfg["loiter_altitude"])
            else:
                z_cmd = SEARCH_ALTITUDE

            if safety_cfg.get("enabled", True) and tilt >= tilt_soft:
                z_cmd = max(z_cmd, SEARCH_ALTITUDE + z_boost)

            Z_MIN = 1.3
            Z_MAX = 3.2
            z_cmd = float(np.clip(z_cmd, Z_MIN, Z_MAX))

            targets[j, :] = np.array([new_xy[0], new_xy[1], z_cmd], dtype=float)

        # controller -> action
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=dt,
                state=obs[j],
                target_pos=targets[j, :],
                target_rpy=np.zeros(3),
            )

        _maybe_log(t_sim)

        if gui and not no_sync:
            sync(i, START, env.CTRL_TIMESTEP)

    if not comm["victim_found"]:
        print(f"\n!!! NOT FOUND within {duration_sec:.1f}s (best min_dist={comm['min_dist_ever']:.2f}m, detect_radius={victim_detect_radius:.2f}m) !!!\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest SAR Leader-Follower (fixed)")

    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics)

    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool)
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool)

    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int)

    parser.add_argument('--debug_logs', default=DEFAULT_DEBUG_LOGS, type=str2bool,
                        help='Enable extra debug logging.')
    parser.add_argument('--log_every_s', default=DEFAULT_LOG_EVERY_S, type=float,
                        help='How often to print debug summary lines, in seconds.')
    parser.add_argument('--log_per_drone', default=DEFAULT_LOG_PER_DRONE, type=str2bool,
                        help='If True, print per-drone telemetry on each debug summary.')

    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=str2bool)

    parser.add_argument('--fast', default=True, type=str2bool)
    parser.add_argument('--render_every', default=10, type=int)
    parser.add_argument('--lidar_every', default=3, type=int)
    parser.add_argument('--no_sync', default=True, type=str2bool)

    parser.add_argument('--safe_visuals', default=True, type=str2bool,
                        help="Use primitive-based visuals (prevents texture crashes).")

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
        no_sync=args.no_sync,
        safe_visuals=args.safe_visuals,
        debug_logs=args.debug_logs,
        log_every_s=args.log_every_s,
        log_per_drone=args.log_per_drone,
    )
