"""
==========================================================================
                MOBILE_1.PY - Husky + UR5 Mobile Robot Navigation
==========================================================================
Grid-world navigation using Value Iteration. A Husky mobile base with UR5
arm navigates from start to goal while avoiding obstacles.

Dependencies: pybullet, numpy, utils.py
Usage: python mobile_1.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from utils import MobileGridEnv, value_iteration


def state_to_position(state, rows, cols, grid_size=0.6, center=(0, 0), z=0.05):
    """Convert grid state index to 3D world position (x, y, z)."""
    row = state // cols
    col = state % cols
    x = center[0] + (col - cols / 2 + 0.5) * grid_size
    y = center[1] + (row - rows / 2 + 0.5) * grid_size
    return [x, y, z]


def draw_mobile_grid(rows, cols, grid_size=0.6, center=(0, 0), z=0.01):
    """Draw grid lines on the ground."""
    line_color = [0.2, 0.2, 0.2]
    x_start = center[0] - (cols / 2) * grid_size
    x_end = center[0] + (cols / 2) * grid_size
    y_start = center[1] - (rows / 2) * grid_size
    y_end = center[1] + (rows / 2) * grid_size

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, 1)
    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, 1)


def hold_ur5_pose(ur5_id, joint_targets, num_joints, force=1500, velocity_gain=1.0):
    """Apply position control to hold UR5 in target pose."""
    for j in range(num_joints):
        p.setJointMotorControl2(
            ur5_id, j, p.POSITION_CONTROL,
            targetPosition=joint_targets[j],
            force=force,
            velocityGain=velocity_gain,
        )


def main():
    # -------------------------------------------------------------------------
    # 1. Parameters
    # -------------------------------------------------------------------------
    ROWS, COLS = 8, 8
    GRID_SIZE = 0.6
    CENTER = [0, 0]
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    NUM_OBSTACLES = 12

    # UR5 arm pose (straight up, stable during motion)
    UR5_ARM_UP = [0, -np.pi / 2, 0, -np.pi / 2, 0, 0]
    JOINT_FORCE = 1500
    VELOCITY_GAIN = 1.0
    SIM_DT = 1.0 / 240

    # Block: 0.05 cube, scale 2.0 → half-height 0.05. Center at z=0.05 → top at 0.1.
    BLOCK_TOP_Z = 0.05 + 0.05  # center + half-height
    ROBOT_BASE_Z = BLOCK_TOP_Z + 1.0  # fly 1m above blocks

    # -------------------------------------------------------------------------
    # 2. Generate obstacles
    # -------------------------------------------------------------------------
    total_states = ROWS * COLS
    available = list(set(range(total_states)) - {START, GOAL})
    num_obs = min(NUM_OBSTACLES, len(available))
    obstacle_states = [int(x) for x in np.random.choice(available, num_obs, replace=False)]

    print(f"Grid: {ROWS}x{COLS}, Start: {START}, Goal: {GOAL}")
    print(f"Obstacles: {obstacle_states}")

    # -------------------------------------------------------------------------
    # 3. Run Value Iteration
    # -------------------------------------------------------------------------
    env = MobileGridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacles=obstacle_states)

    print("Running Value Iteration...")
    t0 = time.time()
    policy, V, _ = value_iteration(env, gamma=GAMMA)
    elapsed = time.time() - t0
    print(f"Converged in {elapsed:.4f}s. V(start)={V[env.start]:.2f}, V(goal)={V[env.goal]:.2f}")

    path = env.get_optimal_path(policy)
    if not path or path[-1] != GOAL:
        print("No valid path to goal!")
        return

    print(f"Path: {path}")

    # -------------------------------------------------------------------------
    # 4. PyBullet setup
    # -------------------------------------------------------------------------
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    draw_mobile_grid(ROWS, COLS, GRID_SIZE, CENTER)

    # Obstacles
    obstacle_urdf = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")
    for obs_state in obstacle_states:
        pos = state_to_position(obs_state, ROWS, COLS, GRID_SIZE, CENTER)
        p.loadURDF(obstacle_urdf, pos, globalScaling=2.0)

    # Husky + UR5 (loaded above blocks)
    husky_id = p.loadURDF("husky/husky.urdf", [0, 0, ROBOT_BASE_Z])
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_id = p.loadURDF(ur5_path, [0, 0, ROBOT_BASE_Z + 0.2])

    # Fixed constraint: Husky top to UR5 base
    p.createConstraint(husky_id, -1, ur5_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.2], [0, 0, 0])

    ur5_num_joints = p.getNumJoints(ur5_id)
    num_controlled = min(6, ur5_num_joints)

    # Offset for moving both bodies together (reduces glitching)
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky_id)
    ur5_pos, _ = p.getBasePositionAndOrientation(ur5_id)
    husky_to_ur5_offset = np.array(ur5_pos) - np.array(husky_pos)

    # -------------------------------------------------------------------------
    # 5. Settle arm before moving
    # -------------------------------------------------------------------------
    for _ in range(120):
        hold_ur5_pose(ur5_id, UR5_ARM_UP, num_controlled, JOINT_FORCE, VELOCITY_GAIN)
        p.stepSimulation()
        time.sleep(SIM_DT)

    # -------------------------------------------------------------------------
    # 6. Execute path
    # -------------------------------------------------------------------------
    prev_pos = state_to_position(path[0], ROWS, COLS, GRID_SIZE, CENTER)
    steps_per_move = 50

    for state in path[1:]:
        target_pos = state_to_position(state, ROWS, COLS, GRID_SIZE, CENTER)
        p.addUserDebugLine(prev_pos, target_pos, [0, 1, 0], 5)

        current_pos, current_orn = p.getBasePositionAndOrientation(husky_id)

        for s in range(steps_per_move):
            hold_ur5_pose(ur5_id, UR5_ARM_UP, num_controlled, JOINT_FORCE, VELOCITY_GAIN)

            alpha = (s + 1) / steps_per_move
            new_x = current_pos[0] * (1 - alpha) + target_pos[0] * alpha
            new_y = current_pos[1] * (1 - alpha) + target_pos[1] * alpha
            new_husky = [new_x, new_y, ROBOT_BASE_Z]
            new_ur5 = (np.array(new_husky) + husky_to_ur5_offset).tolist()

            p.resetBasePositionAndOrientation(husky_id, new_husky, current_orn)
            p.resetBasePositionAndOrientation(ur5_id, new_ur5, current_orn)
            p.stepSimulation()
            time.sleep(SIM_DT)

        prev_pos = target_pos

    print("Mobile navigation complete.")

    # -------------------------------------------------------------------------
    # 7. Idle loop
    # -------------------------------------------------------------------------
    while True:
        hold_ur5_pose(ur5_id, UR5_ARM_UP, num_controlled, JOINT_FORCE, VELOCITY_GAIN)
        p.stepSimulation()
        time.sleep(SIM_DT)


if __name__ == "__main__":
    main()
