"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.

Contains
--------
  - SAC_CONFIG          Hyperparameters for Soft Actor-Critic (YOU will tune these)
  - Training constants  TOTAL_TIMESTEPS, EVAL_FREQ, EVAL_EPISODES, ROBOT_MASS_KG
  - RewardPlotCallback  Records episode rewards and saves a plot after training
  - BipedJumpEnv        Gymnasium environment — provided, do not modify
"""

# ===========================================================================
# Hyperparameters  (edit these for Task 3)
# ===========================================================================

# ============================================================
# TODO: Set the total number of training timesteps (e.g. 1_000_000).
# ============================================================
TOTAL_TIMESTEPS = 800_000

# ============================================================
# TODO: Set how often (in steps) the evaluator runs during training (e.g. 10_000).
# ============================================================
EVAL_FREQ = 10_000

# ============================================================
# TODO: Set the max steps per episode — must match BipedJumpEnv.max_steps (500).
# ============================================================
MAX_EPISODE_STEPS = 900

# ---------------------------------------------------------------------------
# SAC  (Soft Actor-Critic) — the only algorithm used in this assignment
# ---------------------------------------------------------------------------
# ============================================================
# TODO: Fill in SAC_CONFIG with your chosen hyperparameters.
#       Required keys: policy, learning_rate, buffer_size, batch_size,
#                      tau, gamma, ent_coef, verbose.
# ============================================================
SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    buffer_size   = 1_000_000,
    batch_size    = 256,
    tau           = 0.005,
    gamma         = 0.99,
    ent_coef      = "auto",
    learning_starts = 10_000,
    verbose       = 1,
)

# ---------------------------------------------------------------------------
# Evaluation / metric settings  (do not change)
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05   # used to compute Cost of Transport (CoT)


# ===========================================================================
# RewardPlotCallback
# ===========================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_rewards = []

    def _on_training_start(self) -> None:
        num_envs = getattr(self.training_env, "num_envs", 1)
        self._current_episode_rewards = [0.0 for _ in range(num_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if not self._current_episode_rewards:
            self._current_episode_rewards = [0.0 for _ in range(len(rewards) or 1)]

        for idx, reward in enumerate(rewards):
            self._current_episode_rewards[idx] += float(reward)
            if idx < len(dones) and dones[idx]:
                self.episode_rewards.append(self._current_episode_rewards[idx])
                self._current_episode_rewards[idx] = 0.0
        return True   # returning False would stop training

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")

        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv  — provided environment, do not modify
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSEST_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: the biped robot spawns on top of a 1 m tall platform and must
    jump off, then land upright on the ground below.

    Phases
    ------
    1. On platform  
    2. In flight    
    3. Landing      

   
    """

    PLATFORM_H = 0.33         # slab top surface height (m)
    GROUND_Z   = 0.3125       # pelvis/base height when the feet stand on flat ground
    SPAWN_Z    = PLATFORM_H + GROUND_Z
    SPAWN_YAW  = np.pi        # rotate 180 deg so the intended motion is toward -y

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = 500
        self.step_counter      = 0
        self.land_stable_steps = 0
        self.recovery_contact_steps = 0
        self.touchdown_steps = 0
        self.post_edge_stall_steps = 0
        self.has_left_platform = False
        self.has_touchdown = False
        self.edge_departure_step = None
        self.prev_y           = self.spawn_y if hasattr(self, "spawn_y") else 0.0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        slab_path = os.path.join(_ASSEST_DIR, "stair.urdf")
        self.platform_id = p.loadURDF(
            slab_path,
            [0, 2, 0],
            p.getQuaternionFromEuler([0, 0, -3.1416]),
            useFixedBase=True,
            physicsClientId=cid,
        )
        slab_aabb_min, slab_aabb_max = p.getAABB(
            self.platform_id,
            -1,
            physicsClientId=cid,
        )
        self.platform_top_z = slab_aabb_max[2]
        self.platform_edge_y = slab_aabb_min[1]
        self.platform_center_y = 0.5 * (slab_aabb_min[1] + slab_aabb_max[1])
        self.spawn_x = 0.0
        self.spawn_y = 0.0

        # Robot
        urdf_path = os.path.join(_ASSEST_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(
            urdf_path,
            [self.spawn_x, self.spawn_y, self.SPAWN_Z],
            p.getQuaternionFromEuler([0.0, 0.0, self.SPAWN_YAW]),
            useFixedBase=False,
            physicsClientId=cid,
        )
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.joint_names     = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(ji[1].decode("utf-8"))
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)
        self.joint_name_to_local_index = {
            name: idx for idx, name in enumerate(self.joint_names)
        }

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z     = self.SPAWN_Z
        self.has_landed = False
        self.reset()

    def _landing_assist_action(self, action):
        """
        Blend the policy action with a simple landing posture close to the ground.
        This keeps the feet below the pelvis long enough to start real recovery.
        """
        if not self.has_left_platform:
            return action

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        lin_vel, _ = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        assist = np.array(action, dtype=np.float32, copy=True)

        if self.has_touchdown:
            # Short recovery window after first contact: bias toward an upright
            # catch posture, then quickly hand control back to the policy.
            if self.touchdown_steps > 70:
                return assist

            left_ground_contact = len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.left_foot_link,
                    physicsClientId=self.physics_client,
                )
            ) > 0
            right_ground_contact = len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.right_foot_link,
                    physicsClientId=self.physics_client,
                )
            ) > 0

            tilt = abs(roll) + 0.7 * abs(pitch)
            blend = np.clip(0.20 + 0.45 * tilt, 0.20, 0.75)
            fade = np.clip(1.0 - self.touchdown_steps / 70.0, 0.35, 1.0)
            blend *= fade

            roll_correction = float(np.clip(-roll, -0.35, 0.35))
            pitch_correction = float(np.clip(-pitch, -0.28, 0.28))

            # Emergency branch for large post-touchdown tilt.
            if abs(roll) > 0.70 or abs(pitch) > 0.55:
                blend = max(blend, 0.82)
                roll_correction = float(np.clip(-roll, -0.60, 0.60))
                pitch_correction = float(np.clip(-pitch, -0.40, 0.40))

            landing_pose = {
                "left_hip": 0.20 - 0.50 * roll_correction + 0.20 * pitch_correction,
                "right_hip": 0.20 + 0.50 * roll_correction + 0.20 * pitch_correction,
                "left_knee": -0.24,
                "right_knee": -0.24,
                "left_ankle": 0.18 - 0.42 * roll_correction,
                "right_ankle": 0.18 + 0.42 * roll_correction,
            }

            # Push the swing leg down quickly after first contact to promote
            # double-support instead of prolonged one-foot balancing.
            if left_ground_contact and not right_ground_contact:
                landing_pose["right_hip"] += 0.10
                landing_pose["right_knee"] += 0.22
                landing_pose["right_ankle"] += 0.08
            elif right_ground_contact and not left_ground_contact:
                landing_pose["left_hip"] += 0.10
                landing_pose["left_knee"] += 0.22
                landing_pose["left_ankle"] += 0.08
        else:
            # Only help once the robot is actually close to the floor. If we blend
            # too early, the controller can get trapped in a midair "hover" posture.
            if pos[2] > self.GROUND_Z + 0.28 or lin_vel[2] > 0.0:
                return action

            blend = np.clip((self.GROUND_Z + 0.28 - pos[2]) / 0.12, 0.0, 0.55)

            if abs(roll) > 0.9 or abs(pitch) > 0.9:
                blend = min(0.7, blend + 0.10)

            landing_pose = {
                "left_hip": 0.18,
                "right_hip": 0.18,
                "left_knee": -0.04,
                "right_knee": -0.04,
                "left_ankle": 0.16,
                "right_ankle": 0.16,
            }

        for joint_name, target in landing_pose.items():
            local_idx = self.joint_name_to_local_index.get(joint_name)
            if local_idx is None:
                continue
            low, high = self.joint_limits[local_idx]
            norm_target = 2.0 * (target - low) / (high - low) - 1.0
            norm_target = float(np.clip(norm_target, -1.0, 1.0))
            assist[local_idx] = (1.0 - blend) * assist[local_idx] + blend * norm_target

        return assist

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        # ============================================================
        # TODO: Reset the robot joints, base position, and counters, then return the initial observation.
        # ============================================================
        super().reset(seed=seed)

        base_pos = [self.spawn_x, self.spawn_y, self.SPAWN_Z]
        base_orn = p.getQuaternionFromEuler([0.0, 0.0, self.SPAWN_YAW])
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_pos,
            base_orn,
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.robot_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.physics_client,
        )

        for joint_idx in self.joint_indices:
            p.resetJointState(
                self.robot_id,
                joint_idx,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.physics_client,
            )

        self.step_counter = 0
        self.land_stable_steps = 0
        self.recovery_contact_steps = 0
        self.touchdown_steps = 0
        self.post_edge_stall_steps = 0
        self.prev_z = self.SPAWN_Z
        self.prev_y = self.spawn_y
        self.has_landed = False
        self.has_left_platform = False
        self.has_touchdown = False
        self.edge_departure_step = None
        self._initial_base_pos = np.array(base_pos, dtype=np.float32)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        # ============================================================
        # TODO: Read joint states, base pose, velocities, and foot contacts, then return them as a single np.float32 array.
        # ============================================================
        joint_states = p.getJointStates(
            self.robot_id,
            self.joint_indices,
            physicsClientId=self.physics_client,
        )
        joint_pos = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_vel = np.array([state[1] for state in joint_states], dtype=np.float32)

        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        base_lin_vel, _ = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        base_euler = p.getEulerFromQuaternion(base_orn)

        left_ground_contact = float(
            len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.left_foot_link,
                    physicsClientId=self.physics_client,
                )
            ) > 0
        )
        right_ground_contact = float(
            len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.right_foot_link,
                    physicsClientId=self.physics_client,
                )
            ) > 0
        )

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                np.array(base_pos, dtype=np.float32),
                np.array(base_euler, dtype=np.float32),
                np.array(base_lin_vel, dtype=np.float32),
                np.array([left_ground_contact, right_ground_contact], dtype=np.float32),
                np.array([base_pos[2]], dtype=np.float32),
                np.array([float(self.has_landed)], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, ang_vel, landed_now):
        # ============================================================
        # TODO: Compute and return the reward from upright_penalty, z_progress, flight_bonus, and landing_reward.
        # ============================================================
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        left_ground_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.left_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0
        right_ground_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.right_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0
        both_ground = left_ground_contact and right_ground_contact
        any_ground = left_ground_contact or right_ground_contact
        single_ground = left_ground_contact != right_ground_contact
        left_foot_pos = p.getLinkState(
            self.robot_id,
            self.left_foot_link,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client,
        )[0]
        right_foot_pos = p.getLinkState(
            self.robot_id,
            self.right_foot_link,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client,
        )[0]
        pelvis_aabb_min, _ = p.getAABB(
            self.robot_id,
            -1,
            physicsClientId=self.physics_client,
        )
        min_foot_z = min(left_foot_pos[2], right_foot_pos[2])
        pelvis_lowest_z = pelvis_aabb_min[2]
        foot_clearance = max(0.0, min_foot_z - self.GROUND_Z)

        # Primary learning signal: move off the slab while keeping the pelvis/base upright.
        upright_reward = 4.5 * np.exp(-5.0 * (abs(roll) + abs(pitch)))
        upright_penalty = -6.0 * (abs(roll) + abs(pitch))
        y_progress = max(0.0, self.spawn_y - pos[1])
        progress_delta = max(0.0, self.prev_y - pos[1])
        edge_approach = 4.0 * y_progress
        forward_velocity_reward = 1.5 * max(0.0, -float(lin_vel[1]))
        progress_delta_reward = 3.0 * progress_delta
        backward_motion_penalty = -3.0 * max(0.0, float(lin_vel[1]))
        x_drift_penalty = -1.5 * abs(float(pos[0]))
        angular_penalty = -0.8 * (abs(float(ang_vel[0])) + abs(float(ang_vel[1])))
        reverse_progress_penalty = -5.0 * max(0.0, pos[1] - self.prev_y)

        airborne = not left_ground_contact and not right_ground_contact
        z_progress = max(0.0, self.prev_z - pos[2]) * (4.0 if self.has_left_platform else 0.3)
        flight_bonus = 1.5 if self.has_left_platform and airborne and pos[2] < self.SPAWN_Z - 0.02 else 0.0
        edge_cross_bonus = 8.0 if self.has_left_platform else 0.0

        # Keep the body quiet in the air and reward a controlled, upright fall.
        controlled_fall_reward = 0.0
        leg_extension_reward = 0.0
        pelvis_first_penalty = 0.0
        foot_clearance_penalty = 0.0
        near_touchdown_bonus = 0.0
        if self.has_left_platform and airborne:
            controlled_fall_reward += 1.5 * upright_reward
            controlled_fall_reward += 0.8 * max(0.0, self.GROUND_Z + 0.18 - pos[2])
            controlled_fall_reward -= 1.0 * np.linalg.norm(lin_vel)
            controlled_fall_reward -= 14.0 * max(0.0, abs(roll) - 0.30)
            controlled_fall_reward -= 10.0 * max(0.0, abs(pitch) - 0.20)
            if pos[2] < self.GROUND_Z + 0.45:
                # Teach the robot to swing the feet below the pelvis before impact.
                leg_extension_reward += 18.0 * max(0.0, pos[2] - min_foot_z - 0.05)
                if pelvis_lowest_z < min_foot_z - 0.02:
                    pelvis_first_penalty -= 12.0
            if pos[2] < self.GROUND_Z + 0.22:
                foot_clearance_penalty -= 45.0 * foot_clearance
                near_touchdown_bonus += 18.0 * max(0.0, 0.035 - foot_clearance)

        single_leg_touchdown_bonus = 0.0
        single_leg_touchdown_penalty = 0.0
        if self.has_touchdown and single_ground and pos[2] < self.GROUND_Z + 0.25:
            if self.touchdown_steps <= 6:
                single_leg_touchdown_bonus = 4.0
            else:
                single_leg_touchdown_penalty = -1.2 * min(20, self.touchdown_steps - 6)

        double_support_stabilize_bonus = 0.0
        if self.has_left_platform and both_ground and pos[2] < self.GROUND_Z + 0.15:
            double_support_stabilize_bonus = 14.0

        landing_reward = 0.0
        touchdown_reward = 0.0
        stabilization_reward = 0.0
        if self.has_touchdown and single_ground:
            touchdown_reward += 3.0
        if self.has_touchdown and both_ground:
            touchdown_reward += 5.0
        if landed_now:
            landing_reward += 40.0
        if any_ground and pos[2] < self.GROUND_Z + 0.12 and np.linalg.norm(lin_vel) < 0.7:
            landing_reward += 1.0
        if self.has_touchdown and abs(roll) < 0.45 and abs(pitch) < 0.45:
            stabilization_reward += 5.0
        if self.land_stable_steps > 0:
            stabilization_reward += 0.5 * self.land_stable_steps

        no_progress_penalty = 0.0
        if not self.has_left_platform:
            if y_progress < 0.03 and self.step_counter > 20:
                no_progress_penalty -= 1.0
            if y_progress < 0.06 and self.step_counter > 50:
                no_progress_penalty -= 2.0
            if y_progress < 0.10 and self.step_counter > 100:
                no_progress_penalty -= 3.0

        stall_penalty = -0.03 if (not self.has_left_platform and self.step_counter > 50) else 0.0

        action_reward = (
            0.5 * y_progress
            + upright_reward
            + edge_approach
            + forward_velocity_reward
            + progress_delta_reward
            + backward_motion_penalty
            + x_drift_penalty
            + angular_penalty
            + reverse_progress_penalty
            + upright_penalty
            + z_progress
            + flight_bonus
            + edge_cross_bonus
            + controlled_fall_reward
            + leg_extension_reward
            + pelvis_first_penalty
            + foot_clearance_penalty
            + near_touchdown_bonus
            + single_leg_touchdown_bonus
            + single_leg_touchdown_penalty
            + double_support_stabilize_bonus
            + touchdown_reward
            + stabilization_reward
            + landing_reward
            + no_progress_penalty
            + stall_penalty
        )
        return float(action_reward)

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        # ============================================================
        # TODO: Return a list of all non-fixed joint indices for the robot.
        # ============================================================
        return list(self.joint_indices)

    def robot_initial_position(self):
        # ============================================================
        # TODO: Return the robot base position at the start of the episode.
        # ============================================================
        return tuple(self._initial_base_pos.tolist())

    def robot_current_position(self):
        # ============================================================
        # TODO: Return the current robot base position.
        # ============================================================
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        return tuple(pos)

    # ------------------------------------------------------------------
    def step(self, action):
        # ============================================================
        # TODO: Apply actions to joints, step the simulation, compute reward, check termination, and return (obs, reward, terminated, truncated, {}).
        # ============================================================
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._landing_assist_action(action)

        for idx, joint_idx in enumerate(self.joint_indices):
            low, high = self.joint_limits[idx]
            target = low + 0.5 * (action[idx] + 1.0) * (high - low)
            force = 32.0 if self.has_touchdown else 20.0
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target),
                force=force,
                physicsClientId=self.physics_client,
            )

        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode:
            time.sleep(self.timestep)

        self.step_counter += 1
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.physics_client,
        )

        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        left_ground_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.left_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0
        right_ground_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.right_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0
        both_ground = left_ground_contact and right_ground_contact
        any_ground = left_ground_contact or right_ground_contact
        single_ground = left_ground_contact != right_ground_contact

        if self.has_touchdown and any_ground:
            damped_lin_vel = np.array(lin_vel, dtype=np.float32)
            damped_ang_vel = np.array(ang_vel, dtype=np.float32)

            # Touchdown-only damping: keep forward/downward recovery motion,
            # but suppress the rapid roll/pitch spin that causes immediate collapse.
            damped_lin_vel[0] *= 0.42
            damped_lin_vel[1] *= 0.58
            if both_ground and damped_lin_vel[2] < 0.0:
                damped_lin_vel[2] *= 0.20

            damped_ang_vel[0] *= 0.14
            damped_ang_vel[1] *= 0.14

            # During one-foot catch, aggressively damp impact spin so the
            # swing leg has a chance to reach the floor.
            if single_ground and (abs(roll) > 0.45 or abs(pitch) > 0.35):
                damped_ang_vel[0] *= 0.55
                damped_ang_vel[1] *= 0.55
                damped_lin_vel[2] = max(damped_lin_vel[2], -0.22)

            # Soft orientation pull toward upright during the early touchdown
            # window to avoid immediate roll-over after first contact.
            if self.touchdown_steps <= 25 and (abs(roll) > 0.40 or abs(pitch) > 0.30):
                _, _, yaw = p.getEulerFromQuaternion(orn)
                corrected_roll = 0.85 * roll
                corrected_pitch = 0.85 * pitch
                corrected_orn = p.getQuaternionFromEuler([corrected_roll, corrected_pitch, yaw])
                p.resetBasePositionAndOrientation(
                    self.robot_id,
                    pos,
                    corrected_orn,
                    physicsClientId=self.physics_client,
                )
                orn = corrected_orn
                roll, pitch, _ = p.getEulerFromQuaternion(orn)

            p.resetBaseVelocity(
                self.robot_id,
                linearVelocity=damped_lin_vel.tolist(),
                angularVelocity=damped_ang_vel.tolist(),
                physicsClientId=self.physics_client,
            )
            lin_vel = tuple(damped_lin_vel.tolist())
            ang_vel = tuple(damped_ang_vel.tolist())

        if (not self.has_left_platform) and pos[1] <= self.platform_edge_y - 0.015:
            self.has_left_platform = True
            self.edge_departure_step = self.step_counter

        if self.has_left_platform and any_ground:
            self.has_touchdown = True
            self.touchdown_steps += 1
        elif self.has_left_platform and not any_ground:
            self.touchdown_steps = 0

        landed_now = (
            self.has_left_platform
            and (not self.has_landed)
            and self.has_touchdown
            and any_ground
            and pos[2] < self.GROUND_Z + 0.16
            and abs(roll) < 0.9
            and abs(pitch) < 0.9
        )
        if landed_now:
            self.has_landed = True

        reward = self._compute_reward(
            np.array(pos),
            orn,
            np.array(lin_vel),
            np.array(ang_vel),
            landed_now,
        )

        if self.has_left_platform and any_ground:
            self.recovery_contact_steps += 1
        else:
            self.recovery_contact_steps = 0

        if (
            self.has_touchdown
            and both_ground
            and pos[2] < self.GROUND_Z + 0.24
            and np.linalg.norm(np.array(lin_vel[:2], dtype=np.float32)) < 0.40
            and abs(float(lin_vel[2])) < 0.45
            and np.linalg.norm(np.array(ang_vel[:2], dtype=np.float32)) < 2.0
            and abs(roll) < 0.45
            and abs(pitch) < 0.45
        ):
            self.land_stable_steps += 1
        else:
            self.land_stable_steps = max(0, self.land_stable_steps - 1)

        edge_air_steps = (
            0
            if self.edge_departure_step is None
            else self.step_counter - self.edge_departure_step
        )
        stalled_post_edge = (
            self.has_left_platform
            and not self.has_touchdown
            and edge_air_steps >= 25
            and pos[2] > self.GROUND_Z + 0.22
            and abs(lin_vel[1]) < 0.12
            and abs(lin_vel[2]) < 0.08
        )
        if stalled_post_edge:
            self.post_edge_stall_steps += 1
        else:
            self.post_edge_stall_steps = 0
        hover_stall = self.post_edge_stall_steps >= 15

        plane_contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.plane_id,
            physicsClientId=self.physics_client,
        )
        nonfoot_ground_contact = any(
            contact[3] not in (self.left_foot_link, self.right_foot_link)
            for contact in plane_contacts
        )

        crash = (
            pos[2] < 0.12
            or nonfoot_ground_contact
            or (
                self.recovery_contact_steps >= 10
                and pos[2] < self.GROUND_Z + 0.05
                and (abs(roll) > 1.9 or abs(pitch) > 1.9)
            )
            or (
                self.has_touchdown
                and self.touchdown_steps >= 8
                and pos[2] < self.GROUND_Z + 0.02
                and np.linalg.norm(lin_vel) > 1.5
            )
            or hover_stall
        )
        success = self.has_left_platform and self.has_touchdown and self.land_stable_steps >= 20

        if crash:
            reward -= 40.0
        if hover_stall:
            reward -= 25.0
        if success:
            reward += 60.0
        if self.step_counter >= self.max_steps and not self.has_left_platform:
            reward -= 25.0
        if (
            not self.has_left_platform
            and self.step_counter >= 120
            and pos[1] > self.platform_edge_y + 0.15
        ):
            reward -= 20.0
            terminated = True
        else:
            terminated = bool(crash or success)
        truncated = self.step_counter >= self.max_steps

        self.prev_z = pos[2]
        self.prev_y = pos[1]
        info = {
            "success": success,
            "crash": crash,
            "landed": self.has_landed,
            "left_platform": self.has_left_platform,
            "touchdown": self.has_touchdown,
            "touchdown_steps": self.touchdown_steps,
            "stable_steps": self.land_stable_steps,
            "hover_stall": hover_stall,
            "post_edge_stall_steps": self.post_edge_stall_steps,
            "roll": float(roll),
            "pitch": float(pitch),
            "left_ground_contact": left_ground_contact,
            "right_ground_contact": right_ground_contact,
            "nonfoot_ground_contact": nonfoot_ground_contact,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
