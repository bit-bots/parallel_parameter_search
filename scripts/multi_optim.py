#!/usr/bin/env python3

import os
import argparse

# script to run the same optimization multiple times
import subprocess
from time import sleep

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
parser.add_argument('--runs', help='How often should the study be performed', default=1, type=int, required=True)
parser.add_argument('--parallel', help='How often should the study be performed in parallel', default=1, type=int,
                    required=True)

args = parser.parse_args()

TARGET = "optimize_walk.py"

for i in range(args.runs):
    sim_procs = []
    for j in range(args.parallel):
        system_call = [f"rosrun parallel_parameter_search {TARGET}",
                       f"--name {args.name}_{i}",
                       f"--storage {args.storage}",
                       f"--robot {args.robot}",
                       f"--sim {args.sim}",
                       "--type engine",
                       f"--startup {args.startup}",
                       f"--sampler {args.sampler}",
                       f"--trials {args.trials}"]
        system_call = f"rosrun parallel_parameter_search {TARGET} --name {args.name}_{i}  --storage {args.storage} --robot {args.robot} --sim {args.sim} --type engine --startup {args.startup} --sampler {args.sampler} --trials {args.trials}"
        sim_procs.append(subprocess.Popen(system_call, stdout=subprocess.PIPE, shell=True))
        # sleep a bit to prevent some issues with workers taking same namespace
        sleep(10)
    print(sim_procs)
    # wait for all to finish
    for p in sim_procs:
        p.communicate()
