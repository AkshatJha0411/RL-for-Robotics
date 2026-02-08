# Grid Navigation using Dynamic Programming

## Overview
This project implements Dynamic Programming algorithms for robotic path planning. **Policy Iteration** and **Value Iteration** are used to compute optimal policies for navigating a UR5 robotic manipulator on a grid world with obstacles.

The simulation uses **PyBullet** for 3D visualization and physics. The UR5 navigates on an m×n grid with randomly placed obstacles, following the optimal path computed by the DP algorithms.

<p align="center">
  <img src="env.png" alt="Simulation Environment" width="600"/>
</p>

## Installation

```bash
pip install numpy matplotlib seaborn pybullet
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Code

**Main simulation** (Policy Iteration vs Value Iteration, convergence plot, UR5 grid navigation):
```bash
python main.py
```

**Unseen environment** (varied obstacles and grid for robustness testing):
```bash
python main_unseen.py
```

**Extra credits – Mobile base** (Husky + UR5 on scaled grid, flying above obstacles):
```bash
python extracredit.py
```

## Video Demonstrations

- **Basic demo**: `basic_video.mp4`
- **Extra credits (mobile base)**: `extraCredits.webm`

## Code Structure

```
├── main.py           # Main execution: VI vs PI, convergence plot, UR5 navigation
├── main_unseen.py    # Unseen environment test
├── extracredit.py    # Mobile base (Husky + UR5) navigation
├── utils.py          # DP algorithms: policy_evaluation, q_from_v, policy_improvement,
│                     #               policy_iteration, value_iteration, GridEnv, MobileGridEnv
├── assest/           # 3D models (UR5, table, obstacles, Husky)
├── basic_video.mp4   # Basic simulation video
├── extraCredits.webm # Mobile base video
└── README.md
```

## Implemented Algorithms (`utils.py`)

- **policy_evaluation** – Iterative policy evaluation (Bellman expectation)
- **q_from_v** – Q-values from V-values
- **policy_improvement** – Greedy policy from value function
- **policy_iteration** – Alternating evaluation and improvement
- **value_iteration** – Bellman optimality updates

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 4: Dynamic Programming)
- [OpenAI Spinning Up - Value Iteration](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)


