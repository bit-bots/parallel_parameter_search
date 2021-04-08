#!/usr/bin/env python3

import os
import argparse

# script to run the same optimization multiple times

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--startup', help='Startup trials', default=None, type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=1000, type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {TPE, CMAES}', type=str, required=True)
parser.add_argument('--runs', help='How often should the study be performed', default=10, type=int, required=True)

args = parser.parse_args()

TARGET = "optimize_walk.py"

for i in range(args.runs):
    os.system(
        f"rosrun parallel_parameter_search {TARGET} --name {args.name}_{i}  --storage {args.storage} --robot {args.robot} --sim {args.sim} --type engine --startup {args.startup} --sampler {args.sampler} --trials {args.trials}")
