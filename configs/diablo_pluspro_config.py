"""
Description:
Version: 2.0
Author: Dandelion
Date: 2025-03-12 16:35:29
LastEditTime: 2025-03-13 20:52:41
FilePath: /tita_rl/configs/diablo_pluspro_config.py
"""

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DiabloPlusProCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 6 # 智能体在环境中可采取的动作数量

        n_scan = 187
        n_priv_latent = 30  # 3 + 2 + 1 + 4 + 1 + 1+ 6 + 6 + 6
        n_proprio = 27  # 3+3+3+6+6+6
        history_len = 10
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x, y, z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x, y, z [rad/s]
        default_joint_angles = {
            "left_hip_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"hip_joint": 40, "knee_joint": 40, "wheel": 10.0}  # [N*m/rad]
        damping = {"hip_joint": 1.0, "knee_joint": 1.0, "wheel": 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        action_scale_vel = 10
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5 #100Hz
        hip_scale_reduction = 0.5

        use_filter = True

    class commands(LeggedRobotCfg.control):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        use_random_height = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-1.57, 1.57]
            height = [0.26, 0.4]

    class asset(LeggedRobotCfg.asset):

        file = (
            "{ROOT_DIR}/resources/diablo_pluspro_stand/urdf/diablo_pluspro_stand.urdf"
        )
        foot_name = "wheel"
        name = "diablo_pluspro"
        penalize_contacts_on = ["hip_link", "knee_link", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        armature = 0.0422

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        foot_x_position_sigma = 0.01 #越小对x的约束越强

        class scales(LeggedRobotCfg.rewards.scales):
            torques = 0.0
            powers = -2e-5
            termination = -100
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.05
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 1.0
            feet_air_time = 0.0
            collision = -1.0
            stumble = 0.0
            action_rate = -0.01
            # action_smoothness = 0
            stand_still = -0.5 #要给负值才能有起到使其在0输入时关节少动的效果
            # foot_clearance = -0.0
            orientation = -10.0
            stand_nice = -0.2
            same_foot_x_position = 0.1
            inclination = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_restitution = True
        restitution_range = [0.0, 1.0] #恢复系数
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        # randomize_base_com = True
        # added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.9, 1.1]

        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        ##delay lag
        add_action_lag = True
        randomize_lag_timesteps = True
        lag_timesteps = 6
        lag_timesteps_range = [1, 6] # 1~10ms

        add_dof_lag = False
        randomize_dof_lag_timesteps = True
        dof_lag_timesteps_range = [0, 2] # 1~4ms

        add_imu_lag = False # 现在是euler，需要projected gravity 
        randomize_imu_lag_timesteps = True
        imu_lag_timesteps_range = [0, 2] # 实际10~22ms

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_com_displacement = True      #加到priv里的东西
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
        randomize_each_link = True
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)

        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        rand_interval = 10  # Randomization interval in seconds

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False       
        default_joint_friction = [0., 0., 0.01, 0., 0., 0.01]
        joint_friction_range = [0.8, 1.2]
        joint_1_friction_range = [0.9, 1.1] #系数
        joint_2_friction_range = [0.9, 1.1]
        joint_3_friction_range = [0.9, 1.1]
        joint_4_friction_range = [0.9, 1.1]
        joint_5_friction_range = [0.9, 1.1]
        joint_6_friction_range = [0.9, 1.1]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = True
        default_joint_damping = [0.6, 0.6, 0.0,\
                                 0.6, 0.6, 0.0,]
        joint_damping_range = [0.8, 1.2]
        joint_1_damping_range = [0.8, 1.2] #系数
        joint_2_damping_range = [0.8, 1.2]
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]

        randomize_joint_armature = True   
        randomize_joint_armature_each_joint = True
        joint_armature_range = [0.05, 0.10]    
        joint_1_armature_range = [0.05, 0.10]
        joint_2_armature_range = [0.05, 0.10]
        joint_3_armature_range = [0.003, 0.01]
        joint_4_armature_range = [0.05, 0.10]
        joint_5_armature_range = [0.05, 0.10]
        joint_6_armature_range = [0.003, 0.01]

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class costs:
        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            # vel_smoothness = 0.1
            acc_smoothness = 0.1
            # collision = 0.1
            feet_contact_forces = 0.1
            stumble = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # vel_smoothness = 0.0
            acc_smoothness = 0.0
            # collision = 0.0
            feet_contact_forces = 0.0
            stumble = 0.0

    class cost:
        num_costs = 6

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False  # 是否包含动作观察对缓冲区
        terrain_proportions = [0.1, 0.2, 0.35, 0.35, 0.0]
        # terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]


class DiabloPlusProCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = "lstm"
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 6

        teacher_act = True
        imi_flag = False

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "stair_with_capsule"
        experiment_name = "diablo_pluspro"
        policy_class_name = "ActorCriticBarlowTwins"
        runner_class_name = "OnConstraintPolicyRunner"
        algorithm_class_name = "NP3O"
        max_iterations = 10000
        num_steps_per_env = 24
        resume = False
        resume_path = ""
