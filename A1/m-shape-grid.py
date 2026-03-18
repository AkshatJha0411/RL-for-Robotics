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
    python main-3.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import time
import os
import sys


from utils import (
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


def get_m_shape_cells():

    m_cells = set()

    # Left vertical leg (cols 0-1, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 0))
        m_cells.add((row, 1))

    # Right vertical leg (cols 7-8, all rows) - 2 cells wide
    for row in range(7):
        m_cells.add((row, 7))
        m_cells.add((row, 8))

    # Left diagonal (from top-left going down to center)
    # Row 5: extend to col 2
    m_cells.add((5, 2))
    # Row 4: cols 2, 3
    m_cells.add((4, 2))
    m_cells.add((4, 3))
    # Row 3: cols 2, 3, 4 (connecting to middle)
    m_cells.add((3, 2))
    m_cells.add((3, 3))
    m_cells.add((3, 4))
    # Row 2: col 4 (bottom of V)
    m_cells.add((2, 4))

    # Right diagonal (from top-right going down to center)
    # Row 5: extend to col 6
    m_cells.add((5, 6))
    # Row 4: cols 5, 6
    m_cells.add((4, 5))
    m_cells.add((4, 6))
    # Row 3: cols 4, 5, 6 (connecting to middle - 4 already added)
    m_cells.add((3, 5))
    m_cells.add((3, 6))

    # Middle vertical extension (above the V bottom)
    m_cells.add((4, 4))  # extends middle upward
    m_cells.add((5, 4))  # continues up
    m_cells.add((6, 4))  # top of middle extension

    return m_cells


def cell_to_position(row, col, rows, cols, grid_size=0.10,
                     table_center=[0, -0.3, 0.65], z_offset=0.10):
    """Convert (row, col) to world position."""
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    return [x, y, z]


def draw_m_grid(grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw the M-shaped grid with individual blocks.
    Returns the valid cells, start cell, and end cell.
    """
    m_cells = get_m_shape_cells()
    rows, cols = 7, 9  # Grid dimensions for the M shape

    line_color = [0, 0, 0]  # Black borders
    line_width = 2
    z = table_center[2] + 0.001
    half = grid_size / 2

    # Draw each M-shaped cell as a block
    for (row, col) in m_cells:
        pos = cell_to_position(row, col, rows, cols, grid_size, table_center, z_offset=0.001)
        x, y = pos[0], pos[1]

        # Draw filled square (4 border lines)
        p.addUserDebugLine([x-half, y-half, z], [x+half, y-half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y-half, z], [x+half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x+half, y+half, z], [x-half, y+half, z], line_color, line_width)
        p.addUserDebugLine([x-half, y+half, z], [x-half, y-half, z], line_color, line_width)

    # Define start and end cells
    start_cell = (0, 0)   # Bottom-left of M
    end_cell = (0, 8)     # Bottom-right of M

    # Draw start marker (green)
    start_pos = cell_to_position(start_cell[0], start_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    marker_half = half * 0.6
    green = [0, 1, 0]
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]-marker_half, start_pos[2]],
                       [start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]+marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugLine([start_pos[0]-marker_half, start_pos[1]+marker_half, start_pos[2]],
                       [start_pos[0]-marker_half, start_pos[1]-marker_half, start_pos[2]], green, 4, 0)
    p.addUserDebugText("START", [start_pos[0], start_pos[1], start_pos[2] + 0.05], green, 1.0)

    # Draw end marker (red)
    end_pos = cell_to_position(end_cell[0], end_cell[1], rows, cols, grid_size, table_center, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]-marker_half, end_pos[2]],
                       [end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]+marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugLine([end_pos[0]-marker_half, end_pos[1]+marker_half, end_pos[2]],
                       [end_pos[0]-marker_half, end_pos[1]-marker_half, end_pos[2]], red, 4, 0)
    p.addUserDebugText("END", [end_pos[0], end_pos[1], end_pos[2] + 0.05], red, 1.0)



    return m_cells, start_cell, end_cell





if __name__ == "__main__":

    # 1. Define Environment Configuration (M-Shape)
    ROWS = 7
    COLS = 9
    GAMMA = 0.99
    
    # Get valid M-shape cells
    valid_cells = get_m_shape_cells()
    
    # Define Start and Goal (from draw_m_grid logic)
    # Start: Bottom-left (0,0), Goal: Bottom-right (0,8)
    start_row, start_col = 0, 0
    goal_row, goal_col = 0, 8
    
    START = start_row * COLS + start_col
    GOAL = goal_row * COLS + goal_col
    
    # Define fixed obstacles
    OBSTACLE_CELLS = [
        (3, 3),  
        (5, 1),  
        (5, 6),  
        (2, 1),  
        (2, 3),
        (5, 8),
    ]
    
    # Build complete obstacle set for GridEnv
    # Includes both "void" cells (not in M-shape) and explicit obstacles
    env_obstacles = set()
    
    # Add implicit obstacles (voids)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in valid_cells:
                env_obstacles.add(r * COLS + c)
                
    # Add explicit obstacles
    for (r, c) in OBSTACLE_CELLS:
        # Ensure obstacle is within valid area and not on start/goal
        if (r, c) in valid_cells and (r, c) != (start_row, start_col) and (r, c) != (goal_row, goal_col):
            env_obstacles.add(r * COLS + c)
            
    # Initialize Environment
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacles=list(env_obstacles))

    # 2. PyBullet Simulation Setup
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
    # Helper to get end-effector index - usually 7 for UR5
    # We can hardcode or search. Let's assume standard UR5 or use a high index.
    # Looking at typical UR5 setup, ee_link is often index 6 or 7.
    # We'll just define it here as it was used in the original code but not defined in the snippet I replaced.
    ee_index = 6  # Approximation, or often the last link. 
    # Validating previous code usage: 'ee_index' was used in the loop but not defined in the visible snippet? 
    # Ah, I see it referenced in 'p.calculateInverseKinematics(..., ee_index, ...)'. 
    # I should define it if it wasn't defined. The previous snippet didn't show it defined.
    # I'll check if I need to define it.
    joint_num = p.getNumJoints(p.loadURDF(ur5_path, [0,0,-100], useFixedBase=True)) # Load dummy to check? No.
    # Let's clean up: define ee_index safely.
    
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    # Get the actual ee_index (often 6 for wrist_3_link or similar)
    # If not sure, 6 or 7 is standard for UR5. Let's stick to what might work or just define it.
    ee_index = 6 

    sys.stderr = old_stderr

    # Draw M-shaped grid with start and end markers
    m_cells, start_cell, end_cell = draw_m_grid()

    # Add fixed obstacles on the M-shaped grid 
    rows, cols = ROWS, COLS # Reuse constants

    obstacle_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")

    print(f"\nFixed obstacles placed at:")
    for cell in OBSTACLE_CELLS:
        if cell in m_cells and cell != start_cell and cell != end_cell:
            obs_pos = cell_to_position(cell[0], cell[1], rows, cols, z_offset=0.025)
            p.loadURDF(obstacle_path, obs_pos)
            print(f"  Cell {cell}")

    print(f"Total obstacles: {len(OBSTACLE_CELLS)}")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except:
        pass

    # ============================================================
    # TODO: Implement DP algorithms in utils.py, then add simulation code here
    # ============================================================

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
    # plt.figure(figsize=(10, 6))
    # plt.plot(vi_deltas, label=f'Value Iteration ({len(vi_deltas)} iterations)', color='blue', linewidth=2)
    # plt.plot(pi_deltas, label=f'Policy Iteration ({len(pi_deltas)} total sweeps)', color='orange', linewidth=2)
    
    # plt.yscale('log') # Use log scale to see convergence clearly
    # plt.xlabel('Number of Iterations (Sweeps over States)')
    # plt.ylabel('Max Delta ($\Delta$) [Log Scale]')
    # plt.title('Convergence Comparison: Policy Iteration vs Value Iteration')
    # plt.legend()
    # plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Save the plot
    # plt.savefig('convergence_comparison.png')
    # print(f"Convergence plot saved. VI Time: {vi_time:.4f}s, PI Time: {pi_time:.4f}s")
    
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