#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
import sys

import argparse

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler
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
    sys.exit("sampler not correctly specified")

study = optuna.create_study(study_name=args.name, storage=args.storage, direction='minimize',
                            sampler=sampler, load_if_exists=True)
study.set_user_attr("sampler", args.sampler)

objective = WolfgangKickEngineOptimization('worker', gui=args.gui, sim_type=args.sim)
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)
