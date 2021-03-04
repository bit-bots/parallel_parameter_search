#!/usr/bin/env python3

import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler
import numpy as np

import rospy

from parallel_parameter_search.move_base_optimization import WolfgangMoveBaseOptimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao, Talos, reemc} ',
                    default='wolfgang', type=str, required=False)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default='pybullet',
                    type=str,
                    required=False)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--startup', help='Startup trials', default=1000,
                    type=int, required=True)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES}', default=10000,
                    type=str, required=True)

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
else:
    print("sampler not correctly specified")

study = optuna.create_study(study_name=args.name, storage=args.storage, direction='minimize',
                            sampler=sampler, load_if_exists=True)
study.set_user_attr("sampler", args.sampler)
if args.robot == "wolfgang":
    objective = WolfgangMoveBaseOptimization('worker', gui=args.gui, sim_type=args.sim)
else:
    print(f"robot type \"{args.robot}\" not known.")

# give known set of parameters as initial knowledge
study.enqueue_trial({"max_vel_x": 0.1,
                     "min_vel_x": -0.05,
                     "max_vel_y": 0.08,
                     "max_vel_theta": 0.7,
                     "acc_lim_x": 1.0,
                     "acc_lim_y": 1.0,
                     "acc_lim_theta": 4.0,
                     "acc_trans_limit": 1.0,
                     "path_distance_bias": 5,
                     "goal_distance_bias": 10.0,
                     "occdist_scale": 0.1,
                     "twirling_scale": 5.0})
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)
