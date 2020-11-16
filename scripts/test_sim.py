#!/usr/bin/env python3

# this has to be first import, otherwise there will be an error
import rospkg
from bitbots_quintic_walk import PyWalk
import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler
import numpy as np

import rospy

from parallel_parameter_search.simulators import PybulletSim, WebotsSim

parser = argparse.ArgumentParser()
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, darwin, op3, nao, talos, reemc} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')

args = parser.parse_args()

rospy.init_node("sim_test")
rospack = rospkg.RosPack()
urdf_path = rospack.get_path(f'{args.robot}_description') + '/urdf/robot.urdf'

if args.sim == 'pybullet':
    if args.robot == "talos":
        print("talos")
        sim = PybulletSim("", args.gui, urdf_path, foot_link_names=['leg_left_6_link', 'leg_right_6_link'])
        sim.sim.start_position = [0.0, 0.0, 0.8]
        sim.set_joints_dict({"arm_left_4_joint": -1.57, "arm_right_4_joint": -1.57})
    else:
        sim = PybulletSim("/", args.gui, urdf_path)

elif args.sim == 'webots':
    sim = WebotsSim("/", args.gui)
else:
    print(f'sim type {args.sim} not known')

while not rospy.is_shutdown():
    sim.step_sim()

