"""
Description: Config for DDT B1 (8-DOF Wheeled-Legged Robot)
Version: 1.0
Author: Copilot
"""

from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DDTB1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 8 # 8 DOF (4 per leg)

        n_scan = 187
        n_priv_latent = 5
        n_proprio = 33  # 3+3+3+8+8+8 = 33
        history_len = 10
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent
        fail_to_terminal_time_s = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x, y, z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x, y, z [rad/s]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.79,
            "FL_calf_joint": -1.57,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.79,
            "FR_calf_joint": -1.57,
            "FR_foot_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"hip_joint": 40, "thigh_joint": 40, "calf_joint": 40, "foot_joint": 10}  # [N*m/rad]
        damping = {"hip_joint": 1.0, "thigh_joint": 1.0, "calf_joint": 1.0, "foot_joint": 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        action_scale_vel = 10
        decimation = 5 #100Hz
        hip_scale_reduction = 1.0
        use_filter = True

    class commands(LeggedRobotCfg.control):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 5
        resampling_time = 10.0
        heading_command = True
        global_reference = False
        use_random_height = False

        class ranges:
            lin_vel_x = [0, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1, 1]
            heading = [-1.57, 1.57]
            height = [0.45, 0.55]

    class asset(LeggedRobotCfg.asset):
        file = "{ROOT_DIR}/resources/ddt_b1/urdf/ddt_b1.urdf"
        foot_name = "foot"
        name = "ddt_b1"
        penalize_contacts_on = ["hip", "thigh", "calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        armature = 0.0

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2 
            gravity = 0.05 
            height_measurements = 0.02
            contact_states = 0.05

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        foot_x_position_sigma = 0.01
        tracking_sigma = 0.25
        stand_still_command_range = 0.2

        class scales(LeggedRobotCfg.rewards.scales):
            torques = 0.0
            powers = -2e-5
            termination = -100
            tracking_lin_vel = 5.0
            tracking_ang_vel = 2.5
            lin_vel_z = -0.0
            ang_vel_x = -0.1
            ang_vel_y = -0.05
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 2.0
            feet_air_time = 0.0
            collision = -10.0
            stumble = -100.0
            action_rate = -0.01
            action_smoothness = -0.005
            stand_still = -0.5
            orientation = -20.0
            stand_nice = -0.2
            com_feet = -0.4
            same_foot_x_position = 0.05
            inclination = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-5.0, 0.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.0]

        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        
        add_action_lag = True
        randomize_lag_timesteps = True
        lag_timesteps = 3
        lag_timesteps_range = [1, 6]

        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        dof_lag_timesteps_range = [0, 2]

        add_imu_lag = True
        randomize_imu_lag_timesteps = True
        imu_lag_timesteps_range = [0, 2]

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_each_link = True
        link_com_displacement_range_factor = 0.02

        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        rand_interval = 10

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = True       
        default_joint_friction = [0.002, 0.003, 0.002, 0.00, 0.002, 0.003, 0.002, 0.00] # Extended for 8 joints
        joint_friction_range = [0.8, 1.2]
        
        # Ranges for 8 joints
        joint_1_friction_range = [0.7, 1.3] 
        joint_2_friction_range = [0.8, 1.2]
        joint_3_friction_range = [0.8, 1.2]
        joint_4_friction_range = [0.7, 1.3] # Wheel?
        joint_5_friction_range = [0.7, 1.3] 
        joint_6_friction_range = [0.8, 1.2]
        joint_7_friction_range = [0.8, 1.2]
        joint_8_friction_range = [0.7, 1.3] # Wheel?

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = True
        default_joint_damping = [0.0] * 8
        joint_damping_range = [0.8, 1.2]
        
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2]
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]
        joint_7_damping_range = [0.8, 1.2]
        joint_8_damping_range = [0.8, 1.2]

        randomize_joint_armature = True   
        randomize_joint_armature_each_joint = True
        joint_armature_range = [0.03, 0.08]    
        
        joint_1_armature_range = [0.03, 0.06]
        joint_2_armature_range = [0.03, 0.06]
        joint_3_armature_range = [0.003, 0.01]
        joint_4_armature_range = [0.003, 0.01] # Wheel
        joint_5_armature_range = [0.03, 0.06]
        joint_6_armature_range = [0.03, 0.06]
        joint_7_armature_range = [0.003, 0.01]
        joint_8_armature_range = [0.003, 0.01] # Wheel

        randomize_coulomb_friction = True
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]
        angle = [-5, 5]
        update_interval = 1
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
            acc_smoothness = 0.1
            base_height = 0.2
            feet_contact_forces = 0.1
            stumble = 0.3

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            acc_smoothness = 0.0
            base_height = 0.0
            feet_contact_forces = 0.0
            stumble = 0.0

    class cost:
        num_costs = 7

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        include_act_obs_pair_buf = False
        static_friction = 0.6
        dynamic_friction = 0.5
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]

class DDTB1CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = []
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 512
        rnn_num_layers = 1
        tanh_encoder_output = False
        num_costs = 7
        teacher_act = True
        imi_flag = True

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "ddt_b1_test"
        experiment_name = "ddt_b1"
        policy_class_name = "ActorCriticBarlowTwins"
        runner_class_name = "OnConstraintPolicyRunner"
        algorithm_class_name = "NP3O"
        max_iterations = 6000
        num_steps_per_env = 24
        resume = False
        resume_path = ""
