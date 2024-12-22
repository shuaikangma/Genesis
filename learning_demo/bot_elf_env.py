import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from collections import deque

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BotElfEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_single_obs = obs_cfg["num_single_obs"]
        self.num_single_privileged_obs = obs_cfg["num_single_privileged_obs"]
        self.frame_stack = obs_cfg["frame_stack"]
        self.c_frame_stack = obs_cfg["c_frame_stack"]
        self.num_obs = self.num_single_obs * obs_cfg["frame_stack"]
        self.num_privileged_obs = self.num_single_privileged_obs * obs_cfg["c_frame_stack"]
        # self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.01  # control frequence on real robot is 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.cycle_time = reward_cfg["cycle_time"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=10),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="learning_demo/bot_elf/urdf/bot_elf_ess_collision_fixarm.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        #TODO 修正关于dof的索引
        #修正关于link的索引问题
        # #motor_dofs[6, 8, 10, 12, 14, 16, 7, 9, 11, 13, 15, 17]
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp(self.env_cfg["kp"], self.motor_dofs)
        self.robot.set_dofs_kv(self.env_cfg["kd"], self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        
        self.obs_buf = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_single_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.obs_history = deque(maxlen=self.frame_stack)
        self.privileged_obs_history = deque(maxlen=self.c_frame_stack)
        for _ in range(self.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.num_single_obs, dtype=gs.tc_float, device=self.device))
        for _ in range(self.c_frame_stack):
            self.privileged_obs_history.append(torch.zeros(
                self.num_envs, self.num_single_privileged_obs, dtype=gs.tc_float, device=self.device))
        self.obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.privileged_obs_buf_all = torch.stack([self.privileged_obs_history[i]
                                          for i in range(self.privileged_obs_history.maxlen)], dim=1)
        self.obs_buf_all = self.obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf_all = self.privileged_obs_buf_all.reshape(self.num_envs, -1)
        
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dofs_force = torch.zeros_like(self.actions)
        self.last_dofs_force = torch.zeros_like(self.actions)
        self.dofs_force_limits = self.robot.get_dofs_force_range()[1][self.motor_dofs]
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        
        self.links_pos = torch.zeros((self.num_envs, len(self.robot.links), 3), device=self.device, dtype=gs.tc_float)
        self.links_vel = torch.zeros((self.num_envs, len(self.robot.links), 3), device=self.device, dtype=gs.tc_float)
        self.links_net_force = torch.zeros((self.num_envs, len(self.robot.links), 3), device=self.device, dtype=gs.tc_float)

        self.feet_indices = [self.robot.get_link(name).idx_local for name in self.env_cfg["foot_names"]]
        self.knee_indices = [self.robot.get_link(name).idx_local for name in self.env_cfg["knee_names"]]
        self.penalize_contacts_indices = [self.robot.get_link(name).idx_local for name in self.env_cfg["penalize_contacts_on"]]
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=gs.tc_float, device=self.device, requires_grad=False)
        self.last_feet_contact_xy = torch.zeros(self.num_envs, len(self.feet_indices), 2, dtype=gs.tc_float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dofs_force[:] = self.robot.get_dofs_force(self.motor_dofs)
        
        self.links_pos[:] = self.robot.get_links_pos()
        self.links_vel[:] = self.robot.get_links_vel()
        self.links_net_force[:] = self.robot.get_links_net_contact_force()

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self._get_phase()
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        #TODO 探究不同类型obs输入的区别
        self.obs_buf = torch.cat(
            [   torch.sin(2 * torch.pi * self.phase).unsqueeze(-1), #1
                torch.cos(2 * torch.pi * self.phase).unsqueeze(-1), #1
                self.commands * self.commands_scale,  # 3
                
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                
                (self.dof_pos - self.default_dof_pos - self.ref_dof_pos) * self.obs_scales["dof_pos"], # 12
                self.stance_mask, #2
                
                self.actions,  # 12
            ],
            axis=-1,
        )
        self.privileged_obs_buf = torch.cat(
            [
                self.obs_buf,
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
            ],
            axis=-1,
        )
        
        self.obs_history.append(self.obs_buf)
        self.privileged_obs_history.append(self.privileged_obs_buf)
        self.obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        
        self.privileged_obs_buf_all = torch.stack([self.privileged_obs_history[i]
                                          for i in range(self.privileged_obs_history.maxlen)], dim=1)
        self.obs_buf_all = self.obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf_all = self.privileged_obs_buf_all.reshape(self.num_envs, -1)
        
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dofs_force[:] = self.dofs_force[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]

        return self.obs_buf_all, self.privileged_obs_buf_all, self.rew_buf, self.reset_buf, self.extras

    def _get_phase(self):
        self.phase = (self.episode_length_buf * self.dt) / self.cycle_time
        sin_pos = torch.sin(2 * math.pi * self.phase)
        self.stance_mask = torch.zeros((self.num_envs,2), device=self.device, dtype=gs.tc_float)
        # left foot stance
        self.stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        self.stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        self.stance_mask[torch.abs(sin_pos) < 0.1] = 1
        
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        scale_1 = self.reward_cfg["target_joint_pos_scale"]
        scale_2 = 2*scale_1
        # left foot
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot 
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        

    def get_observations(self):
        return self.obs_buf_all

    def get_privileged_observations(self):
        return self.privileged_obs_buf_all

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.last_dof_vel[:] = 0.0
        self.last_dofs_force[:] = 0.0
        self.last_base_lin_vel[:] = 0.0
        self.last_contacts[:] = False

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf_all, None

    # ------------ reward functions----------------
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - self.default_dof_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.links_net_force[:, self.feet_indices, 2] > 5.
        reward = torch.where(contact == self.stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)
    
    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.links_net_force[:, self.feet_indices, 2] > 5.
        self.contact_filt = torch.logical_or(torch.logical_or(contact, self.stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        cur_feet_contact_xy = self.links_pos[:, self.feet_indices, :2]
        self.feet_contact_xy_error = torch.norm(torch.norm( \
            (cur_feet_contact_xy - self.last_feet_contact_xy) * first_contact.unsqueeze(2), dim=2),dim=1)
        self.feet_contact_xy_error = torch.exp(-self.feet_contact_xy_error * 50)
        self.last_feet_contact_xy[self.contact_filt]=cur_feet_contact_xy[self.contact_filt]
        self.feet_air_time += self.dt
        air_time = torch.sum(self.feet_air_time.clamp(0, 0.5) * first_contact, dim=1)
        # air_time *= self.not_stand.squeeze() #no reward for stand
        self.feet_air_time *= ~self.contact_filt
        return air_time
    
    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.links_net_force[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.links_vel[:, self.feet_indices, :2], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    #TODO 根据地面高度求出接触高度
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.links_net_force[:, self.feet_indices, 2] > 5.
        # Get the z-position of the feet and compute the change in z-position
        self.feet_height = self.links_pos[:, self.feet_indices, 2]
        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.reward_cfg["feet_height_target"]) < 0.01
        rew_pos = torch.sum(rew_pos * (1-self.stance_mask), dim=1)
        self.feet_height *= ~contact
        return rew_pos
       
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_l = self.links_pos[:, self.feet_indices[0], :]
        foot_r = self.links_pos[:, self.feet_indices[1], :]
        foot_dist = torch.norm(foot_l - foot_r, dim=1)
        min_fd = self.reward_cfg["min_distance"]
        max_fd = self.reward_cfg["max_distance"]
        d_min = torch.clamp(foot_dist - min_fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    
    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_l = self.links_pos[:, self.knee_indices[0], :]
        knee_r = self.links_pos[:, self.knee_indices[1], :]
        knee_dist = torch.norm(knee_l - knee_r, dim=1)
        min_kd = self.reward_cfg["min_distance"]
        max_kf = self.reward_cfg["max_distance"]
        d_min = torch.clamp(knee_dist - min_kd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_kf, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
    
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) / (torch.norm(self.commands[:, :2], dim=1)+0.05)
        return torch.exp(-lin_vel_error * self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]) / (torch.abs(self.commands[:, 2])+0.02)
        return torch.exp(-ang_vel_error * self.reward_cfg["tracking_sigma"])
    
    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update
    
    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) / (torch.norm(self.commands[:, :2], dim=1)+0.02)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 5.)
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2]) / (torch.abs(self.commands[:, 2])+0.02)
        ang_vel_error_exp = torch.exp(-ang_vel_error * 5.)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
    
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
                  
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        measured_heights = torch.sum(
            self.links_pos[:, self.feet_indices, 2] * self.stance_mask, dim=1) / torch.sum(self.stance_mask, dim=1)
        base_height = self.base_pos[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.reward_cfg["base_height_target"]) * 100)
    
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_base_lin_vel - self.base_lin_vel
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew
        
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print(torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)))
        return torch.sum((torch.norm(self.links_net_force[:, self.feet_indices, :], dim=-1) - self.reward_cfg["max_contact_force"]).clip(0, 200).square(), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.dofs_force/self.dofs_force_limits), dim=1)
    
    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.links_net_force[:, self.penalize_contacts_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_torque_rate(self):
        """
        Penalizes the rate of change of the torques in the robot's joints. This encourages smooth and controlled
        movements by limiting the acceleration of the robot's joints.
        """
        return torch.sum(torch.square((self.last_dofs_force - self.dofs_force)/self.dofs_force_limits / self.dt), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(((torch.abs(self.dofs_force) - self.dofs_force_limits*self.reward_cfg["soft_torque_limit"])/self.dofs_force_limits).clip(min=0.), dim=1)

    
    