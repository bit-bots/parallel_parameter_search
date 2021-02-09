#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler
import numpy as np

import rospy

from parallel_parameter_search.dynup_optimization import WolfgangOptimization, NaoOptimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao, Talos, reemc} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--startup', help='Startup trials', default=None,
                    type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES}', default=10000,
                    type=str, required=True)
parser.add_argument('--direction', help='Direction of standup {front, back} ', default=None, type=str,
                    required=True)
parser.add_argument('--externalforce',
                    help='How large the max random force applied to the robot should be (only in pybullet)',
                    default=0, type=float, required=False)

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
num_variables = 3
if args.startup:
    n_startup_trials = args.startup
else:
    n_startup_trials = num_variables * 11 - 1

multi_objective = False
if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
elif args.sampler == "MOTPE":
    sampler = MOTPESampler(n_startup_trials=n_startup_trials, seed=seed)
    multi_objective = True
else:
    print("sampler not correctly specified")
    exit(1)

if multi_objective:
    study = optuna.create_study(study_name=args.name, storage=args.storage, directions=["minimize"] * num_variables,
                            sampler=sampler, load_if_exists=True)
else:
    study = optuna.create_study(study_name=args.name, storage=args.storage, direction=["minimize"],
                                sampler=sampler, load_if_exists=True)
study.set_user_attr("sampler", args.sampler)

if args.robot == "wolfgang":
    objective = WolfgangOptimization('worker', gui=args.gui, direction=args.direction, externalforce=args.externalforce,
                                     sim_type=args.sim, multi_objective=multi_objective)
elif args.robot == "nao":
    objective = NaoOptimization('worker', gui=args.gui, direction=args.direction, externalforce=args.externalforce,
                                sim_type=args.sim, multi_objective=multi_objective)
else:
    print(f"robot type \"{args.robot}\" not known.")

# sanity check
# study.enqueue_trial({"foot_distance": 0.2,
#                     "leg_min_length": 0.21,
#                     "arm_side_offset": 0.05,
#                     "trunk_x": -0.05,
#                     "max_leg_angle": 60})

if False:
    study.enqueue_trial(
        {"arm_side_offset": 0.0625775539795593, "leg_min_length": 0.21363524122451, "max_leg_angle": 31.0011663418872,
         "rise_time": 0.504558481973029, "time_foot_close": 0.0354093518493812, "time_hands_front": 0.52809400629856,
         "time_hands_rotate": 0.262372840628522, "time_hands_side": 0.121878687748181,
         "time_to_squat": 0.46319617671523, "time_torso_45": 0.2232846740937,
         "trunk_overshoot_angle_front": -25.1782591936925, "trunk_x_front": 0.0478243133630788,
         "wait_in_squat_front": 0.607736653510198, "foot_distance": 0.2, "stabilizing": False, "trunk_height": 0.4,
         "trunk_pitch": 0, "trunk_x_final": 0})

study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)
