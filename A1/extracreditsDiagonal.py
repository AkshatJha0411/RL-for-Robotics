import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from utils2 import MobileGridEnv, value_iteration

def state_to_position(state, rows, cols, grid_size=0.6, center=[0, 0]):
    row = state // cols
    col = state % cols
    x = center[0] + (col - cols/2 + 0.5) * grid_size
    y = center[1] + (row - rows/2 + 0.5) * grid_size
    return [x, y, 0.05] 

def draw_mobile_grid(rows, cols, grid_size=0.6, center=[0, 0]):
    line_color = [0.2, 0.2, 0.2]
    z = 0.01 
    x_start, x_end = center[0] - (cols/2) * grid_size, center[0] + (cols/2) * grid_size
    y_start, y_end = center[1] - (rows/2) * grid_size, center[1] + (rows/2) * grid_size

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, 1)
    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, 1)

if __name__ == "__main__":
   # 1. Setup Parameters
    ROWS, COLS = 8, 8
    GRID_SIZE = 0.6
    START, GOAL = 0, (ROWS * COLS - 1)
    GAMMA=0.99
    
    # 2. Generate Obstacles (similar to main.py logic)
    total_states = ROWS * COLS
    available_states = list(set(range(total_states)) - {START, GOAL})
    num_obstacles = 8 # Increased for the larger mobile grid
    obstacle_states = [int(x) for x in np.random.choice(available_states, num_obstacles, replace=False)]

    # 3. Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    # 4. Initialize Environment
    env = MobileGridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacles=obstacle_states)
    draw_mobile_grid(ROWS, COLS, GRID_SIZE)

    # 5. Load 3D Obstacles into the scene
    # Using the same cube URDF as the main assignment
    obstacle_urdf_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")
    for obs_state in obstacle_states:
        # Scale z_offset and potentially the cube size for the larger mobile base
        obs_pos = state_to_position(obs_state, ROWS, COLS, GRID_SIZE)
        p.loadURDF(obstacle_urdf_path, obs_pos, globalScaling=2.0, useFixedBase=True) # Scaled up for Husky

    # 6. Load Husky + UR5
    husky_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_id = p.loadURDF(ur5_path, [0, 0, 0.3])
    p.createConstraint(husky_id, -1, ur5_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.1], [0, 0, 0.2])
    # Run DP Algorithm
    print("Running Value Iteration...")
    start_time = time.time()
    
    policy, V, delta = value_iteration(env, gamma=GAMMA)
        
    end_time = time.time()
    print(f"Converged in {end_time - start_time:.4f} seconds.")
    print(f"V(start): {V[env.start]:.4f}, V(goal): {V[env.goal]:.4f}")

    path = env.get_optimal_path(policy)

    # Execution with Path Visualization
    if path:
        prev_pos = state_to_position(path[0], ROWS, COLS, GRID_SIZE)
        
        for state in path[1:]:
            target_pos = state_to_position(state, ROWS, COLS, GRID_SIZE)
            
            # Draw green trail segment (Simulation Demonstration requirement)
            p.addUserDebugLine(prev_pos, target_pos, [0, 1, 0], 5)
            
            # Move the base
            steps = 30
            current_base_pos, current_orn = p.getBasePositionAndOrientation(husky_id)
            for s in range(steps):
                alpha = (s + 1) / steps
                ix = current_base_pos[0] * (1-alpha) + target_pos[0] * alpha
                iy = current_base_pos[1] * (1-alpha) + target_pos[1] * alpha
                p.resetBasePositionAndOrientation(husky_id, [ix, iy, 0.1], current_orn)
                p.stepSimulation()
                time.sleep(1./240)
            
            prev_pos = target_pos

    print("Mobile navigation complete.")
    while True: p.stepSimulation(); time.sleep(1./240)