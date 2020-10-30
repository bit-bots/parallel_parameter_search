#!/usr/bin/env python3

from bitbots_msgs.msg import DynUpAction
import bitbots_dynup

import math
import time

import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param
from parallel_parameter_search.simulators import PybulletSim
from sensor_msgs.msg import Imu, JointState


class AbstractDynupOptimization(AbstractRosOptimization):
    def __init__(self, namespace, gui, robot, sim_type, foot_link_names=()):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot)
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                           foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot)
        else:
            print(f'sim type {sim_type} not known')
        # load dynup params
        load_yaml_to_param(self.namespace, 'bitbots_dynup',
                           '/config/dynup_optimization.yaml',
                           self.rospack)
        self.dynup_node = roslaunch.core.Node('bitbots_dynup', 'DynupNode', 'dynup',
                                             namespace=self.namespace)
        self.launch.launch(self.dynup_node)
        self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'DynupNode', timeout=60)
        self.dynup_request_pub = rospy.Publisher(self.namespace + '/dynup', DynUpAction, queue_size=10)

        self.number_of_iterations = 10
        self.time_limit = 10
        self.time_difference = 0
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 90, 0]

    def objective(self, trial):
        raise NotImplementedError

    def run_attempt(self):
        start_time = self.sim.get_time()
        msg = DynUpAction()
        msg.front = 1
        self.dynup_request_pub.publish(msg)
        self.time_difference = self.sim.get_time() - start_time


    def compute_cost(self):
        return 1.0

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def reset(self):
        # reset simulation
        # let the robot do a few steps in the air to get correct walkready position
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.set_cmd_vel(0.1, 0, 0)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.complete_walking_step()
        self.set_cmd_vel(0, 0, 0)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.complete_walking_step()
        self.sim.set_gravity(True)
        self.reset_position()

class WolfgangOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(WolfgangOptimization, self).__init__(namespace, gui, 'wolfgang', sim_type)
        self.reset_height_offset = 0.005

def suggest_walk_params(self, trial):
    self._suggest_walk_params(trial, (0.38, 0.42), (0.15, 0.25), 0.1, 0.03, 0.03)

def load_robot_param(namespace, rospack, name):
    rospy.set_param(namespace + '/robot_type_name', name)
    set_param_to_file(namespace + "/robot_description", name + '_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml', rospack)


if __name__ == "__main__":
    optimizer = AbstractDynupOptimization("ns", "wolfgang")