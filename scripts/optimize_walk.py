#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
from bitbots_quintic_walk import PyWalk
import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler, RandomSampler
import numpy as np

import rospy

from parallel_parameter_search.walk_engine_optimization import DarwinWalkEngine, WolfgangWalkEngine, OP3WalkEngine, \
    NaoWalkEngine, ReemcWalkEngine, TalosWalkEngine, AtlasWalkEngine

from parallel_parameter_search.walk_stabilization import WolfgangWalkStabilization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao, Talos, reemc} ',
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
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES}', default=10000,
                    type=str, required=True)
parser.add_argument('--repetitions', help='How often each trial is repeated while beeing evaluated', default=1,
                    type=int, required=False)

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

num_variables = 3

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
study.set_user_attr("robot", args.robot)
study.set_user_attr("type", args.type)
study.set_user_attr("repetitions", args.repetitions)

if args.type == "engine":
    if args.robot == "darwin":
        objective = DarwinWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "wolfgang":
        objective = WolfgangWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "op3":
        objective = OP3WalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "nao":
        objective = NaoWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "reemc":
        objective = ReemcWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "talos":
        objective = TalosWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "atlas":
        objective = AtlasWalkEngine('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    else:
        print(f"robot type \"{args.robot}\" not known.")
elif args.type == "stabilization":
    if args.robot == "darwin":
        objective = DarwinWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "wolfgang":
        objective = WolfgangWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "op3":
        objective = OP3WalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "nao":
        objective = NaoWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "reemc":
        objective = ReemcWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "talos":
        objective = TalosWalkStabilization('worker', gui=args.gui, walk_as_node=args.node, sim_type=args.sim, repetitions=args.repetitions, multi_objective=multi_objective)
    else:
        print(f"robot type \"{args.robot}\" not known.")
else:
    print(f"Optimization type {args.type} not known.")
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True)
