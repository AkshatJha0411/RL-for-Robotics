"""
==========================================================================
                    MAIN.PY - UR5 GRID NAVIGATION
==========================================================================
Students implement DP algorithms in utils.py and run this to see results.

Dependencies:
    - pybullet
    - numpy
    - utils.py

Usage:
    python main.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt

from utils2 import (
    GridEnv,
    policy_iteration,
    value_iteration
)



def state_to_position(state, rows, cols, grid_size=0.10, 
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols
    
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    
    return [x, y, z]


def draw_grid_lines(rows, cols, grid_size=0.10, table_center=[0, -0.3, 0.65]):

    line_color = [0, 0, 0]
    line_width = 2
    z = table_center[2] + 0.001
    
    x_start = table_center[0] - (cols/2) * grid_size
    x_end = table_center[0] + (cols/2) * grid_size
    y_start = table_center[1] - (rows/2) * grid_size
    y_end = table_center[1] + (rows/2) * grid_size
    

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, line_width)

    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, line_width)





if __name__ == "__main__":
    
    ROWS = 5
    COLS = 6
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    
    # 1. Generate Obstacles First
    total_states = ROWS * COLS
    all_states = set(range(total_states))
    available_states = list(all_states - {START, GOAL})
    
    # Increase number of obstacles for testing robustness
    num_obstacles = min(5, len(available_states))
    obstacle_states = []
    
    if len(available_states) >= num_obstacles:
        obstacle_states = np.random.choice(available_states, num_obstacles, replace=False)
        # Convert to standard python ints for safety
        obstacle_states = [int(x) for x in obstacle_states]
        
    print(f"Goal State: {GOAL}")
    print(f"Obstacles: {obstacle_states}")

    # 2. Initialize Environment with obstacles
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacles=obstacle_states)

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    p.loadURDF("plane.urdf")
    
    table_path = os.path.join("assest", "table", "table.urdf")
    p.loadURDF(table_path, [0, -0.3, 0], globalScaling=2.0)
    
    stand_path = os.path.join("assest", "robot_stand.urdf")
    p.loadURDF(stand_path, [0, -0.8, 0], useFixedBase=True)
    
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    # Use fixed base to prevent falling
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    ur5_num_joints = p.getNumJoints(ur5_id)
    ee_index = ur5_num_joints - 1 

    sys.stderr = old_stderr
    
    draw_grid_lines(env.rows, env.cols)
    
    # Draw Obstacles
    obstacle_urdf_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")
    for obs_state in obstacle_states:
        obs_pos = state_to_position(obs_state, env.rows, env.cols, z_offset=0.025)
        p.loadURDF(obstacle_urdf_path, obs_pos)

    grid_size = 0.10
    half = grid_size / 2 * 0.8


    start_pos = state_to_position(env.start, env.rows, env.cols, z_offset=0.005)
    yellow = [1, 1, 0]
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    
   
    goal_pos = state_to_position(env.goal, env.rows, env.cols, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    
    # ============================================================
    # DP ALGORITHM EXECUTION
    # ============================================================
    
    # 1. Choose algorithm (Policy Iteration or Value Iteration)
    
    
    
    ##Value iteration
    print("Running Value Iteration...")
    start_time = time.time()
    
    policy, V, vi_deltas = value_iteration(env, gamma=GAMMA)
        
    end_time = time.time()
    vi_time=end_time - start_time
    print(f"Converged in {vi_time:.4f} seconds.")
    print(f"V(start): {V[env.start]:.4f}, V(goal): {V[env.goal]:.4f}")


    ##Policy Iteration
    print("Running Policy Iteration...")
    start_time_ = time.time()
    
    policy_, V_, pi_deltas = policy_iteration(env, gamma=GAMMA)
    
    end_time_ = time.time()
    pi_time=end_time_-start_time_
    print(f"Converged in {pi_time:.4f} seconds.")
    print(f"V(start): {V_[env.start]:.4f}, V(goal): {V_[env.goal]:.4f}")

    #VI vs PI comparison
    plt.figure(figsize=(10, 6))
    plt.plot(vi_deltas, label=f'Value Iteration ({len(vi_deltas)} iterations)', color='blue', linewidth=2)
    plt.plot(pi_deltas, label=f'Policy Iteration ({len(pi_deltas)} total sweeps)', color='orange', linewidth=2)
    
    plt.yscale('log') # Use log scale to see convergence clearly
    plt.xlabel('Number of Iterations (Sweeps over States)')
    plt.ylabel('Max Delta ($\Delta$) [Log Scale]')
    plt.title('Convergence Comparison: Policy Iteration vs Value Iteration')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Save the plot
    plt.savefig('convergence_comparison.png')
    print(f"Convergence plot saved. VI Time: {vi_time:.4f}s, PI Time: {pi_time:.4f}s")
    
    # 2. Extract Optimal Path
    path = env.get_optimal_path(policy)
    print(f"Optimal Path Code: {path}")
    
    if not path or path[-1] != GOAL:
        print("Failed to find a path to the goal!")
    else:
        print("Path found! Executing on robot...")
        
        # 3. Visualize Robot Movement along the path
        # Increase Z offset to make sure it doesn't clip through table
        # using a safe height for movement.
        MOVEMENT_Z_OFFSET = 0.20
        
        # Move to start first
        start_pos = state_to_position(path[0], ROWS, COLS, z_offset=MOVEMENT_Z_OFFSET)
        
        print("Moving to start position...")
        for _ in range(50):
            joint_poses = p.calculateInverseKinematics(
                ur5_id, 
                ee_index, 
                start_pos,
                targetOrientation=ur5_start_orn
            )
            for i, val in enumerate(joint_poses):
                p.setJointMotorControl2(ur5_id, i, p.POSITION_CONTROL, targetPosition=val, force=500)
            p.stepSimulation()
            time.sleep(1./240)
            
        prev_pos = start_pos
        
        print("Executing path...")
        for state_idx in range(1, len(path)):
            state = path[state_idx]
            
            # Calculate target position for this step
            target_pos = state_to_position(state, ROWS, COLS, z_offset=MOVEMENT_Z_OFFSET)
            
            # Draw trail
            p.addUserDebugLine(prev_pos, target_pos, [0, 1, 0], 4, 0)
            
            # INTERPOLATION for smooth movement
            # Generate 20 intermediate waypoints between prev_pos and target_pos
            steps = 20
            for s in range(steps):
                alpha = (s + 1) / steps
                interp_pos = [
                    prev_pos[0] * (1 - alpha) + target_pos[0] * alpha,
                    prev_pos[1] * (1 - alpha) + target_pos[1] * alpha,
                    prev_pos[2] * (1 - alpha) + target_pos[2] * alpha
                ]
                
                # Servo to waypoint
                # We perform IK at every fractional step to keep the robot stable
                for _ in range(3): # Small sub-steps for physics stability
                    joint_poses = p.calculateInverseKinematics(
                        ur5_id, 
                        ee_index, 
                        interp_pos,
                        targetOrientation=ur5_start_orn
                    )
                    
                    # Apply joint positions - Key Fix: iterate 0 to N-1
                    if joint_poses:
                        for i, val in enumerate(joint_poses):
                            p.setJointMotorControl2(ur5_id, i, p.POSITION_CONTROL, targetPosition=val, force=500)
                            
                    p.stepSimulation()
                    time.sleep(1./240)
            
            # Pause briefly at the grid center
            time.sleep(0.05)
            prev_pos = target_pos
        
        print("Goal Reached!")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except:
        pass
