import numpy as np
import copy

""" 
 
    Assumes a states as coming from SimpleInteractions :
    [
    4, position and speed
    + 2 * (n) (positions relative to landmarks)
    2 * (n - 1) ( positions relative to agents ) 
    + n ( landmark flags )
    + 2 (position relative to finish zone)
    + 1 ( finish_zone_radius )
    ]
"""
def state_to_teacher_state(state, landmarks, landmarks_flags):
    n_agents = state.shape[0]

    positions = np.tile(state[:, 2:4], (1, n_agents))
    relative_landmarks = np.tile(landmarks, (1, n_agents)) - positions
    relative_agents = state[:, 4 + 2 * (n_agents): 4 + 2 * (n_agents) + 2 * (n_agents - 1)]
    flags = np.tile(landmarks_flags, (n_agents, 1))

    finish_zone_position, finish_zone_radius = compute_finish_zone(state[:, 2:4])

    finish_zone_infos = np.tile(np.hstack((finish_zone_position, finish_zone_radius)), (3, 1))

    return copy.deepcopy(np.hstack((state[:, :4], relative_agents, relative_landmarks, flags, finish_zone_infos)))

def add_phase_to_state(state, phase_flag):
    phase = np.array([[phase_flag] for _ in range(len(state))])
    return np.hstack((state, phase))

"""
    Assumes a state in teacher state structure
"""
def state_to_set_bases_state(state_init, state):
    return np.hstack((state_init.flatten(), state.flatten()))

def state_to_set_finish_zone_state(state_init, state, landmarks):
    return np.hstack((state_to_set_bases_state(state_init, state), landmarks))
"""
    Assumes a state in teacher state structure
"""
def state_to_stop_state(state_init, state, landmarks):
    return np.hstack((state_to_set_bases_state(state_init, state), landmarks))

def __compute_finish_zone(positions):
    center = np.array([np.mean(positions[:, 0]), np.mean(positions[:, 1])])
    tmp = np.tile(center, (positions.shape[0], 1))
    distances = np.linalg.norm(positions - tmp, axis=1)

    return center, max(np.max(distances), 0.3)

def compute_finish_zone(positions):

    ax, ay = positions[0] - positions[1]
    bx, by = positions[1] - positions[2]
    cx, cy = positions[2] - positions[0]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d

    d = np.sqrt((ux - positions[0][0]) ** 2 + (uy - positions[0][1]) ** 2)

    return np.array([ux, uy]), max(d, 0.3)

def _compute_finish_zone(positions):
    b = positions[0]
    c = positions[1]
    d = positions[2]

    temp = c[0] ** 2 + c[1] ** 2
    bc = (b[0] ** 2 + b[1] ** 2 - temp) / 2
    cd = (temp - d[0] ** 2 - d[1] ** 2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        return None

    # Center of circle
    cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((cx - b[0]) ** 2 + (cy - b[1]) ** 2) ** .5

    return np.array([cx, cy]), max(radius, 0.3)