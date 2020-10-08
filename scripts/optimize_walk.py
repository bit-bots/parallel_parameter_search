#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
from bitbots_quintic_walk import PyWalk
import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler
import numpy as np

import rospy

from parallel_parameter_search.walk_optimization import DarwinWalkOptimization, WolfgangWalkOptimization, OP3WalkOptimization, NaoWalkOptimization, ReemcWalkOptimization, TalosWalkOptimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao, Talos, reemc} ', default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str, required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--node', help="Run walking as extra node", action='store_true')

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = 1000

sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=True)
#sampler = CmaEsSampler(seed=seed)

study = optuna.create_study(study_name=args.name, storage=args.storage, direction='minimize',
                            sampler=sampler, load_if_exists=True)

if args.robot == "darwin":
    objective = DarwinWalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
elif args.robot == "wolfgang":
    objective = WolfgangWalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
elif args.robot == "op3":
    objective = OP3WalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
elif args.robot == "nao":
    objective = NaoWalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
elif args.robot == "reemc":
    objective = ReemcWalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
elif args.robot == "talos":
    objective = TalosWalkOptimization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim)
else:
    print(f"robot type \"{args.robot}\" not known.")

study.optimize(objective.objective, n_trials=1000, show_progress_bar=True)
