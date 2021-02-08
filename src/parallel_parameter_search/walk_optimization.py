# THIS HAS TO BE IMPORTED FIRST!!! I don't know why
from bitbots_quintic_walk import PyWalk

import math
import time

import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState


class AbstractWalkOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, walk_as_node):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        # load walk params
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk',
                           '/config/walking_' + robot_name + '_optimization.yaml',
                           self.rospack)

        self.walk_as_node = walk_as_node
        self.current_speed = None
        self.last_time = 0
        if self.walk_as_node:
            self.walk_node = roslaunch.core.Node('bitbots_quintic_walk', 'WalkNode', 'walking',
                                                 namespace=self.namespace)
            self.walk_node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock")]
            self.launch.launch(self.walk_node)
            self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/engine', timeout=60)
            self.cmd_vel_pub = rospy.Publisher(self.namespace + '/cmd_vel', Twist, queue_size=10)
        else:
            load_yaml_to_param("/robot_description_kinematics", robot_name + '_moveit_config',
                               '/config/kinematics.yaml', self.rospack)
            # create walk as python class to call it later
            self.walk = PyWalk(self.namespace)

        self.number_of_iterations = 10
        self.time_limit = 10

        # needs to be specified by subclasses
        self.directions = None
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 0, 0]

    def objective(self, trial):
        raise NotImplementedError

    def suggest_walk_params(self, trial):
        raise NotImplementedError

    def correct_pitch(self, x, y, yaw):
        return self.trunk_pitch + self.trunk_pitch_p_coef_forward * x + self.trunk_pitch_p_coef_turn * yaw

    def evaluate_direction(self, x, y, yaw, trial: optuna.Trial, iteration, time_limit, cost_time=False,
                           start_speed=True):
        if time_limit == 0:
            time_limit = 1
        if start_speed:
            # start robot slowly
            self.set_cmd_vel(x * iteration / 4, y * iteration / 4, yaw * iteration / 4)
        else:
            self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
        print(F'cmd: {x * iteration} {y * iteration} {yaw * iteration}')
        start_time = self.sim.get_time()
        orientation_diff = 0.0
        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            passed_timesteps = passed_time / self.sim.get_timestep()
            if passed_timesteps == 0:
                # edge case with division by zero
                passed_timesteps = 1
            if start_speed:
                if passed_time > 2:
                    # use real speed
                    self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
                elif passed_time > 1:
                    self.set_cmd_vel(x * iteration / 2, y * iteration / 2, yaw * iteration / 2)
            if passed_time > time_limit:
                # reached time limit, stop robot
                self.set_cmd_vel(0, 0, 0)

            if passed_time > time_limit + 2:
                # robot should have stopped now, evaluate the fitness
                if cost_time:
                    return False, 0
                else:
                    # return self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                    early_term, cost = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                    return early_term, orientation_diff / passed_timesteps

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            orientation_diff += abs(rpy[0]) + abs(rpy[1] - self.correct_pitch(x, y, yaw))
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
                # add extra information to trial
                trial.set_user_attr('early_termination_at', (x * iteration, y * iteration, yaw * iteration))
                return True, 1 - min(1, (passed_time / (time_limit + 2)))

            if self.walk_as_node:
                # give time to other algorithms to compute their responses
                # use wall time, as ros time is standing still
                time.sleep(0.01)
            else:
                current_time = self.sim.get_time()
                joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                               self.sim.get_imu_msg(),
                                               self.sim.get_joint_state_msg(),
                                               self.sim.get_pressure_left(), self.sim.get_pressure_right())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
            self.sim.step_sim()

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def run_walking(self, duration):
        start_time = self.sim.get_time()
        while not rospy.is_shutdown() and (duration is None or self.sim.get_time() - start_time < duration):
            self.sim.step_sim()
            current_time = self.sim.get_time()
            joint_command = self.walk.step(current_time - self.last_time, self.current_speed, self.sim.get_imu_msg(),
                                           self.sim.get_joint_state_msg(), self.sim.get_pressure_left(),
                                           self.sim.get_pressure_right())
            self.sim.set_joints(joint_command)
            self.last_time = current_time

    def complete_walking_step(self, number_steps=1, fake=False):
        start_time = self.sim.get_time()
        for i in range(number_steps):
            # does a double step
            while not rospy.is_shutdown():
                current_time = self.sim.get_time()
                joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                               self.sim.get_imu_msg(),
                                               self.sim.get_joint_state_msg(), self.sim.get_pressure_left(),
                                               self.sim.get_pressure_right())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                if not fake:
                    self.sim.step_sim()
                phase = self.walk.get_phase()
                # phase will advance by the simulation time times walk frequency
                next_phase = phase + self.sim.get_timestep() * self.walk.get_freq()
                # do double step to always have torso at same position
                if (phase >= 0.5 and next_phase >= 1.0):
                    # next time the walking step will change
                    break
                if self.sim.get_time() - start_time > 5:
                    # if walking does not perform step correctly due to impossible IK problems, do a timeout
                    break

    def compute_cost(self, x, y, yaw):
        """
        x,y,yaw have to be either 1 for being good, -1 for being bad or 0 for making no difference
        """
        # factor to increase the weight of the yaw since it is a different unit then x and y
        yaw_factor = 5
        pos, rpy = self.sim.get_robot_pose_rpy()

        # 2D pose
        current_pose = [pos[0], pos[1], rpy[2]]
        correct_pose = [x * self.time_limit,
                        y * self.time_limit,
                        (yaw * self.time_limit) % (2 * math.pi)]
        cost = abs(current_pose[0] - correct_pose[0]) \
               + abs(current_pose[1] - correct_pose[1]) \
               + abs(current_pose[2] - correct_pose[
            2]) * yaw_factor
        # method doesn't work for going forward and turning at the same times
        if yaw != 0:
            # just give 0 cost for surviving
            cost = 0
        # test if robot moved at all for simple case
        early_term = False
        if yaw == 0 and ((x != 0 and abs(current_pose[0]) < 0.5 * abs(correct_pose[0])) or (
                y != 0 and abs(current_pose[1]) < 0.5 * abs(correct_pose[1]))):
            early_term = True
            print("didn't move")
        return early_term, cost

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

    def set_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = 0
        msg.angular.z = yaw
        if self.walk_as_node:
            self.cmd_vel_pub.publish(msg)
        else:
            self.current_speed = msg
