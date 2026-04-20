# Reinforcement Learning - Assignments

Course assignments for Semester 6 RL for Robotics. Each assignment has its own folder with detailed instructions and running commands.

---

## Assignment 1: Grid Navigation using Dynamic Programming

### Team
1. [Akshat Jha](https://github.com/AkshatJha0411)
2. [Kusum Agrawal](https://github.com/AgrawalKusum)

### Demo of Husky:
<p align="center">
  <img src="A1-extraCredts.gif" alt="Extra credits – Mobile base (Husky + UR5) navigation" width="600"/>
</p>

Implements **Policy Iteration** and **Value Iteration** for robotic path planning. A UR5 robotic manipulator navigates on a grid world with obstacles. Uses PyBullet for 3D visualization. Includes main simulation, unseen environment tests, and extra credits (mobile base with Husky + mounted UR5).

*See [A1/README.md](A1/README.md) for details of Assignment 1 and it's installation and running instructions.*

---

## Assignment 2: Drone Hovering using Model-Free Control

### Team
1. [Akshat Jha](https://github.com/AkshatJha0411)
2. [Kusum Agrawal](https://github.com/AgrawalKusum)

<p align="center">
  <img src="A2.gif" alt="Extra credits – Mobile base (Husky + UR5) navigation" width="600"/>
</p>

Implements **Monte Carlo Control** and **Q-Learning (TD)** to teach a simulated drone to hover at a fixed 3D target position using the `gym-pybullet-drones` HoverAviary environment. The continuous position state is discretized into a 10-bin-per-axis Q-table, and a 1D thrust action space (`-1`, `0`, `+1`) is used. Also includes three bonus challenges — **SARSA**, **Double Q-Learning**, and **Experience Replay** — all of which were completed for full bonus credit.

**Final Score: 85 / 85 + 20 / 20 bonus points**

*See [A2/README.md](A2/README.md) for implementation details, results, and running instructions.*

---

## Assignment 3: Biped Platform Jump with Deep RL

### Team
1. [Akshat Jha](https://github.com/AkshatJha0411)
2. [Pujan Purohit](https://github.com/PujanPurohit)

<p align="center">
  <img src="A3.gif" alt="Biped platform jump using SAC" width="600"/>
</p>

Trains a 6-DoF biped robot to **jump off a 1 m platform and land upright** using **SAC (Soft Actor-Critic)** in PyBullet. The environment features a multi-phase reward function covering edge approach, controlled flight, and stable landing. A lightweight landing-assist controller bootstraps early learning by blending a safe posture near the ground. Hyperparameters were tuned across three configurations with the best achieving ~10% fall rate.

```bash
cd A3 && python main.py --mode test --model_path diff-configs/models-jump/config3/sac_biped_final --render --episodes 5
```

*See [A3/README.md](A3/README.md) for implementation details, hyperparameter tuning, evaluation metrics, and analysis.*
