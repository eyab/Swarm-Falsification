# examples/test_algorithems/orca_geom.py
import numpy as np

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
    positions = np.asarray(positions, float)
    velocities = np.asarray(velocities, float)
    pref_velocities = np.asarray(pref_velocities, float)

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
