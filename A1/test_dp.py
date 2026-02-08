import numpy as np
from utils import GridEnv, policy_evaluation, q_from_v, policy_improvement, policy_iteration, value_iteration

def test_policy_evaluation():
    print("Testing Part 1: Policy Evaluation...")
    env = GridEnv(rows=3, cols=3, start=0, goal=8)
    # Policy: Always go RIGHT (Action 2)
    policy = np.full(env.nS, 2) 
    gamma = 0.9
    
    V = policy_evaluation(env, policy, gamma=gamma, theta=1e-6)
    
    # Check Bellman Expectation for state 0
    # Next state of 0 (RIGHT) is 1. Reward -1.
    # V(0) should be -1 + gamma * V(1)
    
    expected_v0 = -1 + gamma * V[1]
    assert np.isclose(V[0], expected_v0, atol=1e-4), f"V(0) {V[0]} != -1 + gamma*V(1) {expected_v0}"
    print("PASS: Policy Evaluation consistency check.")

def test_q_computation():
    print("Testing Part 2 : Q-value Computation...")
    env = GridEnv(rows=3, cols=3)
    V = np.zeros(env.nS)
    V[1] = 100 # Fake high value at state 1
    
    # State 0. Action RIGHT (2) -> State 1. Reward -1.
    s = 0
    gamma = 0.9
    q = q_from_v(env, V, s, gamma)
    
    # Q(0, RIGHT) = 1.0 * (-1 + 0.9 * 100) = -1 + 90 = 89
    expected_q_right = -1.0 + gamma * 100.0
    
    assert np.isclose(q[2], expected_q_right), f"Q(0, RIGHT) {q[2]} != {expected_q_right}"
    print("PASS: Q-value Computation.")

def test_policy_improvement():
    print("Testing Part 3: Policy Improvement...")
    env = GridEnv(rows=3, cols=3)
    V = np.zeros(env.nS)
    # Make state 1 very attractive
    V[1] = 100 
    
    # State 0 neighbors: 1 (Right), 3 (Down). 
    # Right goes to 1 (Value 100). Down goes to 3 (Value 0).
    # Greedy policy at 0 should be RIGHT (2).
    
    policy = policy_improvement(env, V, gamma=0.9)
    assert policy[0] == 2, f"Policy at state 0 should be 2 (RIGHT), got {policy[0]}"
    print("PASS: Policy Improvement.")

def test_convergence_consistency():
    print("Testing Part 4 & 5: PI and VI Consistency...")
    # Use a slightly complex grid to ensure non-trivial convergence
    env = GridEnv(rows=4, cols=4, start=0, goal=15)
    gamma = 0.99
    
    print("Running Policy Iteration...")
    pi_policy, pi_v = policy_iteration(env, gamma)
    
    print("Running Value Iteration...")
    vi_policy, vi_v = value_iteration(env, gamma)
    
    # Check Values match
    diff = np.max(np.abs(pi_v - vi_v))
    assert diff < 1e-4, f"PI and VI Value functions differ by {diff}"
    
    # Check for non-trivial convergence (values should be propagated)
    # The max value should be around 100 (reward at goal)
    max_val = np.max(vi_v)
    assert max_val > 0, f"Value function did not propagate positive rewards! Max value: {max_val}"
    
    print(f"PASS: PI and VI Values match (max diff {diff:.2e})")
    print("Successfully converged.")

if __name__ == "__main__":
    try:
        test_policy_evaluation()
        test_q_computation()
        test_policy_improvement()
        test_convergence_consistency()
        print("\nSUCCESS: All Part 1-5 verifications passed!")
    except AssertionError as e:
        print(f"\nFAILURE: Verification failed.\n{e}")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred.\n{e}")
