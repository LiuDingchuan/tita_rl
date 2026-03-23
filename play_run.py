import argparse
import os

from isaacgym import gymapi
from envs import LeggedRobot
from envs import DiabloPlusPro,DDTB1
from modules import *
from configs import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im


def register_tasks():
	task_registry.register(
		"tita",
		LeggedRobot,
		TitaConstraintHimRoughCfg(),
		TitaConstraintHimRoughCfgPPO(),
	)
	task_registry.register(
		"diablo_pluspro",
		DiabloPlusPro,
		DiabloPlusProCfg(),
		DiabloPlusProCfgPPO(),
	)
	task_registry.register("ddt_b1", DDTB1, DDTB1Cfg(), DDTB1CfgPPO())


def parse_play_run_args():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument(
		"--policy_path",
		type=str,
		default=None,
		help="导出的 TorchScript 策略路径，例如 logs/ddt_b1/.../exported/policies/sim2sim.pt",
	)
	parser.add_argument("--duration", type=float, default=20.0, help="运行时长（秒）")
	parser.add_argument("--cmd_vx", type=float, default=0.8, help="x 方向速度指令")
	parser.add_argument("--cmd_vy", type=float, default=0.0, help="y 方向速度指令")
	parser.add_argument("--cmd_yaw", type=float, default=0.0, help="yaw 角速度指令")
	parser.add_argument("--cmd_heading", type=float, default=0.0, help="heading 指令")
	parser.add_argument("--cmd_height", type=float, default=0.45, help="机身高度指令")

	custom_args, remaining = parser.parse_known_args()

	# 让 Isaac Gym 的参数解析器处理其余参数
	sys.argv = [sys.argv[0]] + remaining
	args = get_args()

	for key, value in vars(custom_args).items():
		setattr(args, key, value)

	return args


def resolve_policy_path(args):
	if args.policy_path:
		if os.path.isabs(args.policy_path):
			return args.policy_path
		return os.path.join(ROOT_DIR, args.policy_path)

	if args.load_run is None:
		raise ValueError("请传入 --policy_path，或使用 --load_run 指定日志目录")

	if os.path.isabs(args.load_run):
		run_dir = args.load_run
	else:
		run_dir = os.path.join(ROOT_DIR, "logs", args.task, args.load_run)

	return os.path.join(run_dir, "exported", "policies", "sim2sim.pt")


def build_eval_env_cfg(task_name):
	env_cfg, _ = task_registry.get_cfgs(name=task_name)
	env_cfg.env.num_envs = min(env_cfg.env.num_envs, 30)
	env_cfg.terrain.mesh_type = "plane"
	env_cfg.terrain.curriculum = False
	env_cfg.noise.add_noise = False
	env_cfg.domain_rand.push_robots = False
	env_cfg.domain_rand.randomize_base_com = False
	env_cfg.domain_rand.randomize_base_mass = False
	env_cfg.domain_rand.randomize_motor = False
	env_cfg.domain_rand.randomize_lag_timesteps = False
	env_cfg.domain_rand.randomize_friction = False
	env_cfg.domain_rand.randomize_restitution = False
	env_cfg.domain_rand.add_action_lag = False
	env_cfg.domain_rand.add_dof_lag = False
	env_cfg.domain_rand.add_imu_lag = False
	return env_cfg


def run_policy(args):
	policy_path = resolve_policy_path(args)
	if not os.path.exists(policy_path):
		raise FileNotFoundError(f"策略文件不存在: {policy_path}")

	env_cfg = build_eval_env_cfg(args.task)
	env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
	obs = env.get_observations()

	policy = torch.jit.load(policy_path, map_location=env.device)
	policy.eval()
	print(f"Loaded policy: {policy_path}")

	num_steps = int(args.duration / env.dt)
	n_prop = env.cfg.env.n_proprio
	hist_len = env.cfg.env.history_len

	with torch.no_grad():
		for _ in range(num_steps):
			env.commands[:, 0] = 0.5
			env.commands[:, 1] = 0
			env.commands[:, 2] = 0
			env.commands[:, 3] = 0
			env.commands[:, 4] = 0.3


			obs_prop = obs[:, :n_prop]
			obs_hist = obs[:, -hist_len * n_prop :].view(-1, hist_len, n_prop)

			# try:
			# 	actions = policy(obs_prop, obs_hist)
			# except RuntimeError:
			# 	actions = policy(obs_prop)
			actions = policy(obs_prop, obs_hist)

			obs, _, _, _, _, _ = env.step(actions)

	print(f"Finished rollout: {num_steps} steps, dt={env.dt:.4f}, duration={num_steps * env.dt:.2f}s")


if __name__ == "__main__":
	register_tasks()
	args = parse_play_run_args()
	run_policy(args)
