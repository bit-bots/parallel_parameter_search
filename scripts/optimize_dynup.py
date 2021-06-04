#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
import argparse
import importlib
import time

import optuna
# from optuna.integration.tensorboard import TensorBoardCallback
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler, RandomSampler
import numpy as np

import rospy

from parallel_parameter_search.dynup_optimization import WolfgangOptimization, SigmabanOptimization, Op2Optimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, op2, sigmaban} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--startup', help='Startup trials', default=None, type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000, type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES}', default=10000, type=str, required=True)
parser.add_argument('--direction', help='Direction of standup {front, back} ', default=None, type=str,
                    required=True)
parser.add_argument('--stability', help='Optimize stability', action='store_true')
parser.add_argument('--real_robot', help='run on actual robot', action='store_true')
parser.add_argument('--repetitions', help='How often each trial is repeated while beeing evaluated', default=1,
                    type=int, required=False)
parser.add_argument('--score', help='Which score {SSH, SS, SSHP, SSP}', default="SSHP", type=str, required=True)
args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
if args.score == "SSHP":
    num_variables = 4
elif args.score in ["SSH", "SSP"]:
    num_variables = 3
elif args.score == "SS":
    num_variables = 2
n_startup_trials = args.startup

multi_objective = False
if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
elif args.sampler == "MOTPE":
    if n_startup_trials is None:
        n_startup_trials = num_variables * 11 - 1
    sampler = MOTPESampler(n_startup_trials=n_startup_trials, seed=seed)
    multi_objective = True
elif args.sampler == "Random":
    sampler = RandomSampler(seed=seed)
    multi_objective = True
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
                                     multi_objective=multi_objective, stability=args.stability,
                                     real_robot=args.real_robot, repetitions=args.repetitions, score=args.score)
elif args.robot == "op2":
    objective = Op2Optimization('worker', gui=args.gui, direction=args.direction, sim_type=args.sim,
                                multi_objective=multi_objective, stability=args.stability,
                                real_robot=args.real_robot, repetitions=args.repetitions, score=args.score)
elif args.robot == "sigmaban":
    objective = SigmabanOptimization('worker', gui=args.gui, direction=args.direction, sim_type=args.sim,
                                multi_objective=multi_objective, stability=args.stability,
                                real_robot=args.real_robot, repetitions=args.repetitions, score=args.score)
else:
    print(f"robot type \"{args.robot}\" not known.")

# save how this study was optimized
study.set_user_attr("direction", args.direction)
study.set_user_attr("robot", args.robot)
study.set_user_attr("real_robot", args.real_robot)
study.set_user_attr("stability", args.stability)
study.set_user_attr("repetitions", args.repetitions)
study.set_user_attr("score", args.score)

# sanity check
# study.enqueue_trial({"foot_distance": 0.2,
#                     "leg_min_length": 0.21,
#                     "arm_side_offset": 0.05,
#                     "trunk_x": -0.05,
#                     "max_leg_angle": 60})

if True:
    if args.direction == "front":
        # old sim params front
        study.enqueue_trial(
            {"arm_side_offset": 0.148, "leg_min_length_front": 0.244, "leg_min_length_back": 0.253, "max_leg_angle": 71.71,
             "rise_time": 0.84, "time_foot_close": 0.0,
             "time_foot_ground_front": 0.132, "time_hands_front": 0.396,
             "time_hands_rotate": 0.231, "time_hands_side": 0.132,
             "time_to_squat": 0.924, "time_torso_45": 0.462,
             "trunk_overshoot_angle_front": -10.54, "trunk_x_front": 0.091,
             "wait_in_squat_front": 0.165})
    elif args.direction == "back":
        study.enqueue_trial(
            {"arm_side_offset": 0.148, "rise_time": 0.84, "arms_angle_back": 120.36, "com_shift_1": 0.051,
             "com_shift_2": 0.0, "foot_angle": 51.76, "hands_behind_back_x": 0.162, "hands_behind_back_z": 0.183,
             "leg_min_length_back": 0.253,
             "time_foot_ground_back": 0.536, "time_full_squat_hands": 0.172, "time_full_squat_legs": 0.196,
             "time_legs_close": 0.068, "trunk_height_back": 0.17, "trunk_overshoot_angle_back": 5.95,
             "wait_in_squat_back": 0.6})

if args.stability:
    study.enqueue_trial(
        {"trunk_pitch_p": 0.0, "trunk_pitch_d": 0.0, "trunk_pitch_i": 0.0, "trunk_roll_p": 0.0, "trunk_roll_d": 0.0,
         "trunk_roll_i": 0.0})

# tensorboard_callback = TensorBoardCallback("/tmp/tensorboard/", metric_name="value")
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)  # , callbacks=[tensorboard_callback])
