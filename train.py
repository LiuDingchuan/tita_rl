import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import (
    TitaConstraintHimRoughCfg,
    TitaConstraintHimRoughCfgPPO,
)
# from configs.titati_constaint_config import (
#     TitatiConstraintHimRoughCfg,
#     TitatiConstraintHimRoughCfgPPO,
# )
from configs.diablo_pluspro_config import DiabloPlusProCfg, DiabloPlusProCfgPPO

from global_config import ROOT_DIR, ENVS_DIR
import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from envs import DiabloPlusPro
from utils.task_registry import task_registry


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    task_registry.register(
        "tita_constraint", LeggedRobot, TitaConstraintHimRoughCfg(), TitaConstraintHimRoughCfgPPO()
    )
    task_registry.register(
        "diablo_pluspro", DiabloPlusPro, DiabloPlusProCfg(), DiabloPlusProCfgPPO()
    )
    args = get_args()
    train(args)
