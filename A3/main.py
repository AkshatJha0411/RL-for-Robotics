"""
main.py — Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage examples
--------------
# View the environment (biped + stair in GUI, no model needed):
    python main.py --mode view

# Train SAC (timesteps set in utils.py):
    python main.py --mode train

# Train SAC for a custom number of steps:
    python main.py --mode train --timesteps 500000

# Evaluate the best saved checkpoint (10 episodes, headless):
    python main.py --mode test

# Evaluate with GUI rendering:
    python main.py --mode test --render --episodes 5

# Evaluate a specific model file:
    python main.py --mode test --model_path "models/sac_best/best_model"
"""

import argparse
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from tqdm.auto import tqdm

from utils import (
    BipedJumpEnv, RewardPlotCallback,
    TOTAL_TIMESTEPS, EVAL_FREQ, MAX_EPISODE_STEPS,
    SAC_CONFIG,
    EVAL_EPISODES, ROBOT_MASS_KG,
)

# ── Algorithm registry ────────────────────────────────────────────────────────
# TODO: Register the SAC algorithm with its config from utils.py.
ALGO_MAP = {
    "sac": (SAC, SAC_CONFIG),
}


def _resolve_num_envs() -> int:
    cpu_count = os.cpu_count() or 1
    target = int(cpu_count * 0.9)
    if cpu_count >= 32:
        return min(28, cpu_count)
    return max(1, target)


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env(rank: int, render: bool = False, monitor_path: str | None = None):
    def _init():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        env = BipedJumpEnv(render=render)
        if monitor_path is not None:
            return Monitor(env, monitor_path)
        return env
    return _init


class EpisodeProgressCallback(BaseCallback):
    """Shows an episode-based tqdm bar with ETA during training."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.total_episodes = max(1, math.ceil(total_timesteps / MAX_EPISODE_STEPS))
        self.completed_episodes = 0
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_episodes, desc="Episodes", unit="ep")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        completed_now = sum(1 for info in infos if isinstance(info, dict) and "episode" in info)
        if completed_now and self.pbar is not None:
            self.completed_episodes += completed_now
            avg_steps_per_episode = max(1.0, self.num_timesteps / max(1, self.completed_episodes))
            estimated_total_episodes = max(1, math.ceil(self.total_timesteps / avg_steps_per_episode))
            if estimated_total_episodes != self.total_episodes:
                self.total_episodes = estimated_total_episodes
                self.pbar.total = estimated_total_episodes
                self.pbar.refresh()
            self.pbar.update(completed_now)
            self.pbar.set_postfix_str(
                f"cur_ep={self.completed_episodes}/{self.total_episodes} "
                f"timesteps={self.num_timesteps}/{self.total_timesteps}"
            )
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


# ── Environment Preview ────────────────────────────────────────────────────────
def view():
    """Spawns the biped + stair in GUI mode. Press Ctrl+C to quit."""
    import pybullet_data

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    assest = os.path.join(os.path.dirname(__file__), "assest")
    p.loadURDF(os.path.join(assest, "biped_.urdf"), [0, 0, 0.81],
               useFixedBase=False, physicsClientId=cid)
    p.loadURDF(os.path.join(assest, "stair.urdf"),  [0, 2, 0],
               p.getQuaternionFromEuler([0, 0, -3.1416]),
               useFixedBase=True, physicsClientId=cid)

    print("[view] Biped + stair spawned. Press Ctrl+C to quit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    p.disconnect(cid)


# ── Training ──────────────────────────────────────────────────────────────────
def train(timesteps: int, render: bool = False, init_model_path: str | None = None):
    """
    Trains a SAC agent on the 1 m platform jump task and saves the model.

    Steps
    -----
    1. Create training and evaluation environments (wrapped in Monitor).
    2. Instantiate SAC with SAC_CONFIG from utils.py.
    3. Set up EvalCallback (saves best model) and RewardPlotCallback.
    4. Call model.learn() and handle KeyboardInterrupt for crash-saves.
    5. Save the final model and plot the reward curve.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/sac_best", exist_ok=True)
    os.makedirs("logs/sac_goal", exist_ok=True)
    os.makedirs("logs/sac_eval", exist_ok=True)

    num_envs = 1 if render else _resolve_num_envs()
    device = _resolve_device()
    torch.set_num_threads(max(1, _resolve_num_envs()))

    if render:
        train_env = DummyVecEnv([_make_env(rank=0, render=True, monitor_path="logs/sac_monitor.csv")])
        train_env = VecMonitor(train_env)
    else:
        train_env = SubprocVecEnv([_make_env(rank=i, render=False) for i in range(num_envs)])
        train_env = VecMonitor(train_env, filename="logs/sac_monitor.csv")

    eval_env = DummyVecEnv([_make_env(rank=0, render=False)])
    eval_env = VecMonitor(eval_env)

    algo_cls, algo_config = ALGO_MAP["sac"]
    if init_model_path:
        model = algo_cls.load(init_model_path, env=train_env, device=device)
        model.tensorboard_log = "logs/sac_goal/"
        print(f"[train] Resuming from {init_model_path}")
    else:
        model = algo_cls(
            env=train_env,
            tensorboard_log="logs/sac_goal/",
            device=device,
            **algo_config,
        )

    reward_cb = RewardPlotCallback()
    progress_cb = EpisodeProgressCallback(total_timesteps=timesteps)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/sac_best",
        log_path="logs/sac_eval",
        eval_freq=max(1, EVAL_FREQ // num_envs),
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES,
    )
    callbacks = CallbackList([reward_cb, progress_cb, eval_cb])

    print(f"[train] device={device}  num_envs={num_envs}  timesteps={timesteps}")
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        crash_path = "models/sac_biped_crashsave"
        model.save(crash_path)
        print(f"[train] Interrupted. Crash checkpoint saved to {crash_path}.zip")
    finally:
        final_path = "models/sac_biped_goal"
        model.save(final_path)
        reward_cb.plot_rewards("reward_curve_sac.png")
        train_env.close()
        eval_env.close()
        print(f"[train] Final model saved to {final_path}.zip")


# ── Evaluation ────────────────────────────────────────────────────────────────
def _test_single_model(model_path: str, episodes: int, render: bool):
    """
    Loads a trained SAC model and evaluates it for a given number of episodes.

    Metrics reported per episode
    ----------------------------
    - Steps taken
    - Total reward
    - Energy consumed  (sum of |torque × velocity| × dt)
    - Distance travelled (Euclidean, spawn → landing)

    Summary metrics printed at the end
    -----------------------------------
    - Average reward
    - Fall rate  (%)
    - Average distance (m)
    - Average energy (J)
    - Cost of Transport (CoT) = Energy / (mass × g × distance)
    """
    DT = 1.0 / 50.0   # simulation timestep (must match utils.py)

    env = BipedJumpEnv(render=render)
    model = SAC.load(model_path, env=env, device=_resolve_device())

    joint_indices = env.get_joint_indices()
    total_energy = 0.0
    total_distance = 0.0
    total_reward = 0.0
    fall_count = 0

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        start_pos = np.array(env.robot_initial_position(), dtype=np.float32)

        done = False
        truncated = False
        episode_reward = 0.0
        episode_energy = 0.0
        steps = 0
        success = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            joint_states = p.getJointStates(
                env.robot_id,
                joint_indices,
                physicsClientId=env.physics_client,
            )
            episode_energy += sum(abs(state[3] * state[1]) for state in joint_states) * DT
            success = success or bool(info.get("success", False))

            base_pos = env.robot_current_position()
            joint_angles = [state[0] for state in joint_states]
            joint_angle_str = ", ".join(f"{angle:.3f}" for angle in joint_angles)
            print(
                f"[trace] ep={episode}/{episodes} step={steps:03d} "
                f"pos=({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}) "
                f"roll={info.get('roll', 0.0):.3f} pitch={info.get('pitch', 0.0):.3f} "
                f"L={int(bool(info.get('left_ground_contact', False)))} "
                f"R={int(bool(info.get('right_ground_contact', False)))} "
                f"touchdown={int(bool(info.get('touchdown', False)))} "
                f"stable_steps={int(info.get('stable_steps', 0))} "
                f"joint_angles=[{joint_angle_str}]"
            )

        end_pos = np.array(env.robot_current_position(), dtype=np.float32)
        distance = float(np.linalg.norm(end_pos - start_pos))
        total_reward += episode_reward
        total_energy += episode_energy
        total_distance += distance
        if not success:
            fall_count += 1

        print(
            f"[test] ep={episode}/{episodes} "
            f"steps={steps} reward={episode_reward:.2f} "
            f"distance={distance:.3f}m energy={episode_energy:.3f}J "
            f"success={success}"
        )

    n = float(episodes)
    avg_reward = total_reward / n
    fall_rate = 100.0 * fall_count / n
    avg_distance = total_distance / n
    avg_energy = total_energy / n
    cot = total_energy / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print(f"\n=== Evaluation Summary: {model_path} ===")
    print(f"Average Reward   : {avg_reward:.3f}")
    print(f"Fall Rate (%)    : {fall_rate:.2f}")
    print(f"Average Distance : {avg_distance:.3f} m")
    print(f"Average Energy   : {avg_energy:.3f} J")
    print(f"Cost of Transport: {cot:.6f}")

    env.close()


def test(model_paths, episodes: int, render: bool):
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    for idx, model_path in enumerate(model_paths, start=1):
        print(f"\n[test] Evaluating model {idx}/{len(model_paths)}: {model_path}")
        _test_single_model(model_path, episodes, render)


# ── CLI entry-point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 — Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode",       choices=["view", "train", "test"], required=True,
                        help="view: preview env  |  train: train SAC  |  test: evaluate")
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Override TOTAL_TIMESTEPS from utils.py")
    parser.add_argument("--init_model_path", type=str, default=None,
                        help="Optional saved model to resume training from in --mode train")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model (.zip) for --mode test")
    parser.add_argument("--episodes",   type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--render",     action="store_true",
                        help="Enable PyBullet GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view()

    else:
        if args.mode == "train":
            ts = args.timesteps if args.timesteps else TOTAL_TIMESTEPS
            train(ts, args.render, args.init_model_path)
        elif args.mode == "test":
            model_paths = (
                [args.model_path]
                if args.model_path
                else ["models/sac_best/best_model", "models/sac_biped_goal"]
            )
            test(model_paths, args.episodes, args.render)


if __name__ == "__main__":
    main()
