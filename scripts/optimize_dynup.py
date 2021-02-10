#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
import argparse
import importlib
import time

import optuna
#from optuna.integration.tensorboard import TensorBoardCallback
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler, RandomSampler
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
parser.add_argument('--stability',
                    help='Optimize stability',
                    default=False, type=bool, required=False)

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
num_variables = 5
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
elif args.sampler == "Random":
    sampler = RandomSampler(seed=seed)
else:
    print("sampler not correctly specified")
    exit(1)

if multi_objective:
    study = optuna.create_study(study_name=args.name, storage=args.storage, directions=["minimize"] * num_variables,
                                sampler=sampler, load_if_exists=True)
else:
    study = optuna.create_study(study_name=args.name, storage=args.storage, direction="minimize",
                                sampler=sampler, load_if_exists=True)
study.set_user_attr("sampler", args.sampler)

if args.robot == "wolfgang":
    objective = WolfgangOptimization('worker', gui=args.gui, direction=args.direction, sim_type=args.sim,
                                     multi_objective=multi_objective, stability=args.stability)
elif args.robot == "nao":
    objective = NaoOptimization('worker', gui=args.gui, direction=args.direction, sim_type=args.sim,
                                multi_objective=multi_objective, stability=args.stability)
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
        {"arm_side_offset": 0.072, "leg_min_length": 0.237, "max_leg_angle": 47.11, "rise_time": 0.554166666666667,
         "time_foot_close": 0.583333333333333, "time_hands_front": 0.770833333333333, "time_hands_rotate": 0.25,
         "time_hands_side": 0.495833333333333, "time_to_squat": 0.470833333333333, "time_torso_45": 0.0208333333333333,
         "trunk_overshoot_angle_front": -17.85, "trunk_pitch_d": 0.0, "trunk_pitch_p": 0.0,
         "trunk_x_front": -0.024, "wait_in_squat_front": 0.379166666666667, "foot_distance": 0.2, "stabilizing": False,
         "trunk_height": 0.4, "trunk_pitch": 0, "trunk_x_final": 0})

if False:
    study.enqueue_trial({"trunk_pitch_p": 0.0, "trunk_pitch_d": 0.0, "trunk_roll_p": 0.0, "trunk_roll_d": 0.0})

#tensorboard_callback = TensorBoardCallback("/tmp/tensorboard/", metric_name="value")
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)#, callbacks=[tensorboard_callback])
