#!/usr/bin/env python3
import json
import sys

import argparse

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler
import numpy as np

import rospy

from parallel_parameter_search.kick_optimization import WolfgangKickEngineOptimization


parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--startup', help='Startup trials', default=1000,
                    type=int, required=True)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES, MOTPE}', default=10000,
                    type=str, required=True)
parser.add_argument('--tensorboard-log-dir', help='Directory for tensorboard logs', type=str)
parser.add_argument('--results-only', help="Do not optimize, just show results of an old study", action='store_true')
parser.add_argument('--json', help="Print best params in json", action='store_true')

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

multi_objective = False
if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
elif args.sampler == "MOTPE":
    # Fall?, Time, velocity, directional error
    directions = ['maximize', 'minimize', 'maximize', 'minimize']
    if n_startup_trials != 11 * len(directions) - 1:
        sys.exit(f"With MOTPE sampler, you should use {11 * len(directions) - 1} startup trials!")
    sampler = MOTPESampler(n_startup_trials=n_startup_trials, seed=seed)
    multi_objective = True
else:
    sys.exit("sampler not correctly specified")

if args.results_only:
    study = optuna.load_study(study_name=args.name, storage=args.storage, sampler=sampler)
else:
    if args.tensorboard_log_dir:
        from optuna.integration.tensorboard import TensorBoardCallback
        tensorboard_callback = TensorBoardCallback(args.tensorboard_log_dir, "cost")
        callbacks = [tensorboard_callback]
    else:
        callbacks = []


    if multi_objective:
        study = optuna.create_study(study_name=args.name, storage=args.storage, directions=directions,
                                    sampler=sampler, load_if_exists=True)
    else:
        study = optuna.create_study(study_name=args.name, storage=args.storage, direction='minimize',
                                    sampler=sampler, load_if_exists=True)
    study.set_user_attr("sampler", args.sampler)

    objective = WolfgangKickEngineOptimization('worker', gui=args.gui, sim_type=args.sim, multi_objective=multi_objective, kamikaze=True)

    # Enqueue some good parameters
    study.enqueue_trial({
        'foot_rise': 0.08,
        'foot_distance': 0.2,
        'kick_windup_distance': 0.5,
        'trunk_height': 0.41,
        'trunk_roll': 0.1,
        'trunk_pitch': -0.141,
        'trunk_yaw': 0.0611,
        'move_trunk_time': 0.59,
        'raise_foot_time': 0.22,
        'move_to_ball_time': 0.19,
        'kick_time': 0.27,
        'move_back_time': 0.21,
        'lower_foot_time': 0.06,
        'move_trunk_back_time': 0.08,
        'stabilizing_point_x': -0.02,
        'stabilizing_point_y': -0.02,
        'kick_speed': 1,
    })
    study.enqueue_trial({
        'foot_rise': 0.08,
        'foot_distance': 0.17,
        'kick_windup_distance': 0.23,
        'trunk_height': 0.36,
        'trunk_roll': -0.208,
        'trunk_pitch': 0.134,
        'trunk_yaw': -0.007,
        'move_trunk_time': 0.64,
        'raise_foot_time': 0.24,
        'move_to_ball_time': 0.13,
        'kick_time': 0.16,
        'move_back_time': 0.15,
        'lower_foot_time': 0.2,
        'move_trunk_back_time': 0.12,
        'stabilizing_point_x': -0.02,
        'stabilizing_point_y': 0.01,
        'kick_speed': 1,
    })
    study.enqueue_trial({
        'foot_rise': 0.08,
        'foot_distance': 0.2,
        'kick_windup_distance': 0.29,
        'trunk_height': 0.39,
        'trunk_roll': 0.101229096615671,
        'trunk_pitch': -0.136135681655558,
        'trunk_yaw': 0.0558505360638186,
        'move_trunk_time': 0.56,
        'raise_foot_time': 0.15,
        'move_to_ball_time': 0.19,
        'kick_time': 0.15,
        'move_back_time': 0.07,
        'lower_foot_time': 0.06,
        'move_trunk_back_time': 0.08,
        'stabilizing_point_x': -0.02,
        'stabilizing_point_y': -0.02,
        'kick_speed': 1,
    })

    study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True, callbacks=callbacks)

if multi_objective:
    if not args.json:
        print(f'Using MOTPE, cannot determine single best trial. Printing the best trials:')
        for trial in study.best_trials:
            print(f'Trial {trial.number}: Values {trial.values}')
            print(trial.params)
    else:
        results = {trial.number: trial.params for trial in study.best_trials}
        print(json.dumps(results, indent=4))
else:
    if not args.json:
        print(f'Best result was {study.best_value} in trial {study.best_trial.number} of {len(study.trials)}')
        print(study.best_params)
    else:
        result = {study.best_trial.number: study.best_trial.values}
        print(json.dumps(result, indent=4))
