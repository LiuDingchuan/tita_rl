# import cv2
import os

from isaacgym import gymapi
from envs import LeggedRobot
from envs import DiabloPlusPro
from modules import *
from configs import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def log_and_plot_states(env, env_cfg, obs, infos, actions, logger, i):
    #--- param used for plot and log states ---#
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    latent = None
    CoM_offset_compensate = False
    # vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    if i < stop_state_log:
        # print("step:", i, "env.dt:", env.dt)
        logger.log_states(
            {
                "dof_pos_target": actions[robot_index, joint_index].item()
                * env.cfg.control.action_scale
                + env.default_dof_pos[robot_index, joint_index].item(),
                "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                "dof_torque": env.torques[robot_index, joint_index].item(),
                "command_yaw": env.commands[robot_index, 1].item(),
                "command_height": env.commands[robot_index, 2].item(),
                # "base_height": env.base_height[robot_index].item(),
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                "contact_forces_z": env.contact_forces[
                    robot_index, env.feet_indices, 2
                ]
                .cpu()
                .numpy(),
            }
        )
        if CoM_offset_compensate:
            logger.log_states({"command_x": vel_cmd[robot_index].item()})
        else:
            logger.log_states({"command_x": env.commands[robot_index, 0].item()})
        if latent is not None:
            logger.log_states(
                {
                    "est_lin_vel_x": latent[robot_index, 0].item()
                    / env.cfg.normalization.obs_scales.lin_vel,
                    "est_lin_vel_y": latent[robot_index, 1].item()
                    / env.cfg.normalization.obs_scales.lin_vel,
                    "est_lin_vel_z": latent[robot_index, 2].item()
                    / env.cfg.normalization.obs_scales.lin_vel,
                }
            )
            if latent.shape[1] > 3 and env_cfg.noise.add_noise:
                logger.log_states(
                    {
                        "base_vel_yaw_obs": obs[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.ang_vel,
                        "dof_pos_obs": obs[robot_index, 9 + joint_index].item()
                        / env.cfg.normalization.obs_scales.dof_pos
                        + env.default_dof_pos[robot_index, joint_index].item(),
                        "dof_vel_obs": obs[robot_index, 15 + joint_index].item()
                        / env.cfg.normalization.obs_scales.dof_vel,
                    }
                )
                logger.log_states(
                    {
                        "base_vel_yaw_est": latent[robot_index, 3 + 2].item()
                        / env.cfg.normalization.obs_scales.ang_vel,
                        "dof_pos_est": latent[
                            robot_index, 3 + 9 + joint_index
                        ].item()
                        / env.cfg.normalization.obs_scales.dof_pos
                        + env.default_dof_pos[robot_index, joint_index].item(),
                        "dof_vel_est": latent[
                            robot_index, 3 + 15 + joint_index
                        ].item()
                        / env.cfg.normalization.obs_scales.dof_vel,
                    }
                )
    elif i == stop_state_log:
        print("START PLOTTING STATES")
        logger.plot_states()
    if 0 < i < stop_rew_log:
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)
    elif i == stop_rew_log:
        logger.print_rewards()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.control.use_filter = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    print(policy)
    #model_dict = torch.load(os.path.join(ROOT_DIR, 'model_4000_phase2_hip.pt'))
    #################新方法，从最新一次的路径直接加载模型，且会根据task名字自动检索#################
    train_cfg.runner.resume = True
    n3po_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = n3po_runner.alg.actor_critic.to(env.device)
    # policy = n3po_runner.get_inference_policy(device=env.device) #这个函数返回函数不太对，actor_critic被改动过
    if EXPORT_POLICY:
        path = os.path.join(
            ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        policy.save_torch_jit_policy(path, env.device) #保存模型为jit script，libtorch可以加载
        print("Exported policy as jit script to: ", path)

    #################老方法，从指定路径加载模型#################
    # model_dict = torch.load(os.path.join(ROOT_DIR, 'model_7100.pt'))
    # policy.load_state_dict(model_dict['model_state_dict'])
    # policy = policy.to(env.device)
    # policy.save_torch_jit_policy('model.pt',env.device)

    # clear images under frames folder
    # frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    # delete_files_in_directory(frames_path)

    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 40
    num_frames = int(video_duration / env.dt)# 40/0.01
    print(f'gathering {num_frames} frames')
    video = None

    #torch.sum(self.last_actions - self.actions, dim=1)
    # self.base_lin_vel[:, 2]
    #torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    logger = Logger(env.dt)
    for i in range(num_frames):
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        env.commands[:,0] = 1
        env.commands[:,1] = 0
        env.commands[:,2] = 0
        env.commands[:,3] = 0
        actions = policy.act_teacher(obs)
        # actions = torch.clamp(actions,-1.2,1.2)

        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)
        if RECORD_FRAMES:
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512,512,4))[:,:,:3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 

        log_and_plot_states(env, env_cfg, obs, infos, actions, logger, i)

    print("action rate:",action_rate/num_frames)
    print("z vel:",z_vel/num_frames)
    print("xy_vel:",xy_vel/num_frames)
    print("feet air reward",feet_air_time/num_frames)

    video.release()

if __name__ == '__main__':
    task_registry.register("tita",LeggedRobot,TitaConstraintHimRoughCfg(),TitaConstraintHimRoughCfgPPO())
    task_registry.register("titatit",LeggedRobot,TitatiConstraintHimRoughCfg(),TitatiConstraintHimRoughCfgPPO())
    task_registry.register(
        "diablo_pluspro", DiabloPlusPro, DiabloPlusProCfg(), DiabloPlusProCfgPPO()
    )
    RECORD_FRAMES = False
    EXPORT_POLICY = True
    args = get_args()
    play(args)
