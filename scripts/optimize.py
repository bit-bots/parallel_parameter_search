#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
from bitbots_quintic_walk import PyWalk
import cProfile as profile

import argparse
import importlib

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

import rospy

from parallel_parameter_search.walk_optimization import WolfgangWalkOptimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--objective', help='Name of the objective class. Has to be of type AbstractRosOptimization.',
                    default=None, type=str, required=False)
# todo kwargs for optuna
args = parser.parse_args()

#importlib.import_module(args.objective)
#if not issubclass(args.objective, AbstractRosOptimization):
#    print('Objective class is not a subclass of AbstractRosOptimization.')
#    exit(1)

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = 10

sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
#pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=10)

study = optuna.create_study(study_name=args.name, storage=args.storage, direction='minimize',
                            sampler=sampler, load_if_exists=True)

#objective = args.objective()
pr = profile.Profile()
pr.enable()
try:
    objective = WolfgangWalkOptimization('worker', gui=True, walk_as_node=False)
    study.optimize(objective.objective, n_trials=1000, show_progress_bar=True)
finally:
    print("hi")
    pr.disable()
    pr.dump_stats('profile.pstat')