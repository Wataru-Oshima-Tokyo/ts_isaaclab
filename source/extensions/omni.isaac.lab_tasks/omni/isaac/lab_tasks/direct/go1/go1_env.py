# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .go1_env_cfg import Go1FlatEnvCfg, Go1RoughEnvCfg
import numpy as np

class Go1Env(DirectRLEnv):
    cfg: Go1FlatEnvCfg | Go1RoughEnvCfg

    def __init__(self, cfg: Go1FlatEnvCfg | Go1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "phase_contact",
                "contact_no_vel",
                "hip_pos"
            ]
        }
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._robot.find_joints(".*hip_joint")
        self._calf_ids, _ = self._robot.find_joints(".*calf_joint")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh", "trunk")
        self.last_contacts = torch.zeros(self.num_envs, len(self._feet_ids), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.debug_print()
        self._init_foot()




    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go1RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        # self.update_feet_state()
        self.phase = (self.episode_length_buf *  self.step_dt) % self.step_period / self.step_period
        self.phase_FL_RR = self.phase  # Front-left (FL) and Rear-right (RR) in sync
        self.phase_FR_RL = (self.phase + self.step_offset) % 1  # Front-right (FR) and Rear-left (RL) offset

        # Assign phases to legs based on their indices (FL, FR, RL, RR) order matters
        self.leg_phase = torch.cat([
            self.phase_FL_RR.unsqueeze(1),  # FL
            self.phase_FR_RL.unsqueeze(1),  # FR
            self.phase_FR_RL.unsqueeze(1),  # RL
            self.phase_FL_RR.unsqueeze(1)   # RR
        ], dim=-1)

    def _init_foot(self):
        # Initialize step-related parameters
        self.feet_num = len(self._feet_ids)
        self.step_period = 0.8  # self.cfg.asset.step_period
        self.step_offset = 0.5  # self.cfg.asset.step_offset
        self.step_height = 0.1  # self.cfg.asset.step_height
        self.phase = (self.episode_length_buf *  self.step_dt) % self.step_period / self.step_period
        self.phase_FL_RR = self.phase  # Front-left (FL) and Rear-right (RR) in sync
        self.phase_FR_RL = (self.phase + self.step_offset) % 1  # Front-right (FR) and Rear-left (RL) offset

        # Assign phases to legs based on their indices (FL, FR, RL, RR) order matters
        self.leg_phase = torch.cat([
            self.phase_FL_RR.unsqueeze(1),  # FL
            self.phase_FR_RL.unsqueeze(1),  # FR
            self.phase_FR_RL.unsqueeze(1),  # RL
            self.phase_FL_RR.unsqueeze(1)   # RR
        ], dim=-1)

    def debug_print(self):
        print(f"feet ids {self._feet_ids}")
        print(f"hip_ids {self._hip_ids}")



    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # print("stepping")
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras




    def _apply_action(self):

        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        height_data = None
        # if isinstance(self.cfg, Go1RoughEnvCfg):
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_com_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    sin_phase,
                    cos_phase
                )
                if tensor is not None
            ],
            dim=-1,
        )
        privileged_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_com_lin_vel_b,
                    self._robot.data.root_com_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    sin_phase,
                    cos_phase
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs, "critic": privileged_obs}
        return observations


    #rewards -----------

    def _rewards_lin_vel_error(self):
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_com_lin_vel_b[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / 0.25)
         
    def _rewards_yaw_rate_error(self):
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_com_ang_vel_b[:, 2])
        return torch.exp(-yaw_rate_error / 0.25)

    def _rewards_feet_air_time(self):
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        return air_time
    def _rewards_z_vel_error(self):
        return torch.square(self._robot.data.root_com_lin_vel_b[:, 2])

    def _rewards_ang_vel_error(self):
        return torch.sum(torch.square(self._robot.data.root_com_ang_vel_b[:, :2]), dim=1)

    def _rewards_joint_torques(self):
        return torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

    def _rewards_joint_accel(self):
        return torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

    def _rewards_action_rate(self):
        return torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

    def _rewards_collision(self):
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        return torch.sum(is_contact, dim=1)

    def _rewards_orientation(self):
        return  torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self._robot.data.joint_pos[:, self._hip_ids]), dim=1)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Iterate over legs (order: FL, FR, RL, RR)
        for i in range(self.feet_num):
            # Determine if the current phase indicates a stance phase (< 0.55)
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if the foot is in contact with the ground
            contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids[i], 2] > 1

            # Reward correct contact behavior (stance matches contact)
            res += ~(contact ^ is_stance)  # XOR for mismatch, negate for correct match

        return res

    def _reward_contact_no_vel(self):
        # print(self._robot.data.joint_pos.data)
        # print(f"calf ids {self._calf_ids}")
        # Penalize contact with no velocity
        # contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids[i], 2] > 1
        contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :3], dim=2) > 1.
        
        # print(contact)
        # contact = contact[:, :len(self._calf_ids)]  # Filter for calf IDs
        # print("Joint velocities shape:", self._robot.data.joint_vel.shape)
        # print("Contact shape before filter:", contact.shape)
        # print("Filtered contact shape:", contact[:, :len(self._calf_ids)].shape)

        # contact_feet_vel = self._robot.data.joint_vel.data[:,self._calf_ids] * contact.unsqueeze(-1)
        contact_feet_vel = self._robot.data.joint_vel[:, self._calf_ids] * contact.float()

        # # print(contact_feet_vel)
        # penalize = torch.square(contact_feet_vel[:, :, :3])
        # return torch.sum(penalize, dim=(1,2))
        # Penalize square of velocities for feet in contact
        penalize = torch.square(contact_feet_vel)
        # print("Penalize shape:", penalize.shape)

        return torch.sum(penalize, dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self._robot.data.root_link_state_w[:, 2]
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    """ should be implimented later


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)





    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.step_height) * ~contact
        return torch.sum(pos_error, dim=(1))
    """


    #----------------------

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error_mapped = self._rewards_lin_vel_error()
        # yaw rate tracking
        yaw_rate_error_mapped = self._rewards_yaw_rate_error()
        # z velocity tracking
        z_vel_error = self._rewards_z_vel_error()
        # angular velocity x/y
        ang_vel_error = self._rewards_ang_vel_error()
        # joint torques
        joint_torques = self._rewards_joint_torques()
        # joint acceleration
        joint_accel = self._rewards_joint_accel()
        # action rate
        action_rate = self._rewards_action_rate()

        feet_air_time = self._rewards_feet_air_time()
        # undersired contacts
        undersired_contacts = self._rewards_collision()
        # flat orientation
        flat_orientation = self._rewards_orientation()
        #contact phase
        phase_contact = self._reward_contact()

        contact_no_vel = self._reward_contact_no_vel()

        hip_pos = self._reward_hip_pos()

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undersired_contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "phase_contact": phase_contact * self.cfg.phase_contact_reward_scale * self.step_dt,
            "contact_no_vel": contact_no_vel * self.cfg.contact_no_vel_reward_scale * self.step_dt,
            "hip_pos": hip_pos * self.cfg.hip_pos_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)