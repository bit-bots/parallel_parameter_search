#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
from bitbots_quintic_walk import PyWalk
import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler, RandomSampler, NSGAIISampler
import numpy as np

import rospy

from parallel_parameter_search.walk_engine_optimization import OP2WalkEngine, WolfgangWalkEngine, OP3WalkEngine, \
    NaoWalkEngine

from parallel_parameter_search.walk_stabilization import WolfgangWalkStabilization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, op2, op3, nao} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--node', help="Run walking as extra node", action='store_true')
parser.add_argument('--type', help='Optimization type that should be used {engine, stabilization} ', default=None,
                    type=str, required=True)
parser.add_argument('--startup', help='Startup trials', default=None,
                    type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {MOTPE, TPE, CMAES, NSGA2, Random}', default=10000,
                    type=str, required=True)
parser.add_argument('--repetitions', help='How often each trial is repeated while beeing evaluated', default=1,
                    type=int, required=False)

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

num_variables = 2

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
elif args.sampler == "NSGA2":
    sampler = NSGAIISampler(seed=seed)
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
study.set_user_attr("robot", args.robot)
study.set_user_attr("type", args.type)
study.set_user_attr("repetitions", args.repetitions)

if args.type == "engine":
    if args.robot == "op2":
        objective = OP2WalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "wolfgang":
        objective = WolfgangWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim,
                                       repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "op3":
        objective = OP3WalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "nao":
        objective = NaoWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    else:
        print(f"robot type \"{args.robot}\" not known.")
elif args.type == "stabilization":
    if args.robot == "wolfgang":
        objective = WolfgangWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim,
                                              repetitions=args.repetitions, multi_objective=multi_objective)
    else:
        print(f"robot type \"{args.robot}\" not known.")
else:
    print(f"Optimization type {args.type} not known.")

if True:
    if len(study.get_trials()) == 0:
        # old params
        study.enqueue_trial({"double_support_ratio": 0.2, "first_step_swing_factor": 1.0,
                             "foot_distance": 0.2, "foot_rise": 0.06, "freq": 1.8, "trunk_height": 0.4,
                             "trunk_phase": -0.1, "trunk_pitch": 0.12,
                             "trunk_pitch_p_coef_forward": 1.2,
                             "trunk_pitch_p_coef_turn": -0.05, "trunk_swing": 0.4,
                             "trunk_x_offset": 0.0, "trunk_y_offset": 0.005,
                             "trunk_z_movement": 0.0,
                             })
        # best from first optimization
        study.enqueue_trial({"double_support_ratio": 0.205697020634591, "first_step_swing_factor": 1.43802181605666,
                             "foot_distance": 0.193167886146203, "foot_rise": 0.0634321629243628,
                             "freq": 1.8808567497048,
                             "trunk_height": 0.391032569052562, "trunk_phase": -0.225213829524181,
                             "trunk_pitch": 0.178415760885359, "trunk_pitch_p_coef_forward": 0.20532497579384,
                             "trunk_pitch_p_coef_turn": -0.323293138896975, "trunk_swing": 0.415852788166455,
                             "trunk_x_offset": -0.0256820805595823, "trunk_y_offset": 0.00563280748937276,
                             "trunk_z_movement": 0.00571131278337307, })

        # best from second optimization
        study.enqueue_trial({"double_support_ratio": 0.15,
                             "first_step_swing_factor": 1.0,
                             "foot_distance": 0.193,
                             "foot_rise": 0.08,
                             "freq": 1.881,
                             "trunk_height": 0.391,
                             "trunk_phase": -0.225,
                             "trunk_pitch": 0.178,
                             "trunk_pitch_p_coef_forward": 0.205,
                             "trunk_pitch_p_coef_turn": -0.323,
                             "trunk_swing": 0.3,
                             "trunk_x_offset": -0.0256,
                             "trunk_y_offset": 0.0056,
                             "trunk_z_movement": 0.0057})

study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)

# close simulator window
objective.sim.close()
