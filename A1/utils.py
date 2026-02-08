"""
==========================================================================
                        UTILS.PY - STUDENT IMPLEMENTATION
==========================================================================
Students must implement the Dynamic Programming algorithms below.

Author: Assignment 1 - AR525
==========================================================================
"""

import numpy as np

class GridEnv:
    
    def __init__(self, rows=5, cols=6, start=0, goal=29, obstacles=None):
   
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):

        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):

        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):

        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_next_state(self, state, action):
  
        row, col = self._state_to_pos(state)
        
        if action == 0:    # LEFT
            col -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # UP
            row -= 1
        
        # Check boundary
        if not self._is_valid_pos(row, col):
            return state
            
        next_state = self._pos_to_state(row, col)
        
        # Check obstacle collision - treat as wall
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def _build_dynamics(self):

        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                # If state itself is an obstacle or goal (terminal), usually no transition or self-loop
                # But for standard grid world, we usually simulate moving *from* it if we somehow start there.
                # However, obstacles should be blocked.
                
                next_state = self._get_next_state(state, action)
                
                reward = -1.0
                done = False
                
                if next_state == self.goal:    #should off grid actions be penalized?
                    reward = 100.0
                    done = True
                
                P[state][action] = [(1.0, next_state, reward, done)]
        
        return P
    
    def get_optimal_path(self, policy):

        path = [self.start]
        current_state = self.start
        
        # Max steps to prevent infinite loops
        for _ in range(self.rows * self.cols * 2):
            if current_state == self.goal:
                break
            
            action = policy[current_state]
            next_state = self._get_next_state(current_state, action)
            
            path.append(next_state)
            current_state = next_state
        
        return path


#for the extra credits:
class MobileGridEnv(GridEnv):
   
    def __init__(self, rows=10, cols=10, start=0, goal=99, obstacles=None):     #larger scale for mobile base
        super().__init__(rows, cols, start, goal, obstacles)
       

# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS
# ==========================================================================

def policy_evaluation(env, policy, gamma=1.0, theta=1e-8):       #algo pg 92
    V = np.zeros(env.nS)
    deltas = []
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            # For multiple outcomes P[s][a] -> [(prob, next_state, reward, done)]
            # In deterministic grid world, only one tuple usually.
            a = policy[s]
            for prob, next_state, reward, done in env.P[s][a]:
                # If done, next value is 0 (terminal)
                v_next = V[next_state] if not done else 0.0      #if next state is the goal, v_next is 0
                v += prob * (reward + gamma * v_next)            #bellman equation update step
            
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        deltas.append(delta)
        if delta < theta:
            break
            
    return np.array(V), deltas


def q_from_v(env, V, s, gamma=1.0):
    q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            v_next = V[next_state] if not done else 0.0
            q[a] += prob * (reward + gamma * v_next)
            
    return q


def policy_improvement(env, V, gamma=1.0):
    policy = np.zeros(env.nS, dtype=int)
    
    for s in range(env.nS):
        q_values = q_from_v(env, V, s, gamma)
        best_a = np.argmax(q_values)
        policy[s] = best_a
        
    return policy


def policy_iteration(env, gamma=1.0, theta=1e-8):         #algo pg 97
   # 1. Initialization
    policy = np.zeros(env.nS, dtype=int)
    all_deltas=[]
    
    while True:
        # 2. Policy Evaluation
        V, deltas = policy_evaluation(env, policy, gamma, theta)
        all_deltas.extend(deltas)
        
        # 3. Policy Improvement
        policy_stable = True
        for s in range(env.nS):
            old_action = policy[s]
            
            # Get best action based on current V
            q_values = q_from_v(env, V, s, gamma)
            best_action = np.argmax(q_values)
            
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            return policy, V, all_deltas


def value_iteration(env, gamma=1.0, theta=1e-8):   #algo pg 101
    V = np.zeros(env.nS)
    delta_history=[]
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            
            # One step lookahead to find max action value
            q_values = q_from_v(env, V, s, gamma)
            best_value = np.max(q_values)
            
            V[s] = best_value
            delta = max(delta, np.abs(best_value - v))
        
        delta_history.append(delta)
        if delta < theta:
            break
            
    
    # Extract optimal policy
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, delta_history