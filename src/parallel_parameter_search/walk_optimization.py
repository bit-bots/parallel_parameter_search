# THIS HAS TO BE IMPORTED FIRST!!! I don't know why
from bitbots_msgs.msg import JointCommand
from bitbots_quintic_walk import PyWalk

import math
import time
import numpy as np
import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState


class AbstractWalkOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, walk_as_node, config_name="_optimization"):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        # load walk params
        self.param_yaml_data = load_yaml_to_param(self.namespace, 'bitbots_quintic_walk',
                                                  '/config/walking_' + robot_name + config_name + '.yaml',
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

    def evaluate_direction(self, x, y, yaw, iteration, time_limit, start_speed=True):
        if time_limit == 0:
            time_limit = 1
        if start_speed:
            # start robot slowly
            self.set_cmd_vel(x * iteration / 4, y * iteration / 4, yaw * iteration / 4)
        else:
            self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
        print(F'cmd: {round(x * iteration, 2)} {round(y * iteration, 2)} {round(yaw * iteration, 2)}')
        start_time = self.sim.get_time()
        orientation_diff = 0.0
        angular_vel_diff = 0.0

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
            if passed_time > time_limit - 1:
                # decelerate
                self.set_cmd_vel(x * iteration / 2, y * iteration / 2, yaw * iteration / 2)

            if passed_time > time_limit:
                # reached time limit, stop robot
                self.set_cmd_vel(0, 0, 0)

            if passed_time > time_limit + 2:
                # robot should have stopped now, evaluate the fitness
                didnt_move, pose_cost, poses = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                return False, didnt_move, pose_cost, orientation_diff / passed_timesteps, \
                       angular_vel_diff / passed_timesteps, poses

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            # get orientation diff scaled to 0-1
            orientation_diff += min(1, (abs(rpy[0]) + abs(rpy[1] - self.correct_pitch(x, y, yaw))) * 0.5)
            imu_msg = self.sim.get_imu_msg()
            # get angular_vel diff scaled to 0-1. dont take yaw, since we might actually turn around it
            angular_vel_diff += min(1, (abs(imu_msg.angular_velocity.x) + abs(imu_msg.angular_velocity.y)) / 60)
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
                didnt_move, pose_cost, poses = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                return True, didnt_move, pose_cost, orientation_diff / passed_timesteps, \
                       angular_vel_diff / passed_timesteps, poses

            if self.walk_as_node:
                # give time to other algorithms to compute their responses
                # use wall time, as ros time is standing still
                time.sleep(0.01)  # todo would be better to just wait till a command from walking arrived, like in dynup
            else:
                current_time = self.sim.get_time()
                joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                               imu_msg,
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

    def compute_cost(self, v_x, v_y, v_yaw):
        def get_matrix(v_x, v_y, v_yaw, t):
            if v_yaw == 0:
                # prevent division by zero
                return (np.array([v_x * t, v_y * t]), 0)
            else:
                return (np.array([(v_x * math.sin(t * v_yaw) - v_y * (1 - math.cos(t * v_yaw))) / v_yaw,
                                  (v_y * math.sin(t * v_yaw) + v_x * (1 - math.cos(t * v_yaw))) / v_yaw]), v_yaw * t)

        def rotation(yaw):
            return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

        # 2D pose
        pos, rpy = self.sim.get_robot_pose_rpy()
        current_pose = [pos[0], pos[1], rpy[2]]
        t = self.time_limit

        # compute end pose. robot is basically walking on circles where radius=v_x/v_yaw and v_y/v_yaw
        # we need to include the acceleration and deceleration phases
        # 1s first accel, 1s second accel, time_limit -3 full speed, 1s deceleration
        # always need to rotate around current yaw
        after_first_accel = get_matrix(v_x / 4, v_y / 4, v_yaw / 4, 1)
        after_second_accel = get_matrix(v_x / 2, v_y / 2, v_yaw / 2, 1)
        after_second_accel = (after_first_accel[0] + np.dot(rotation(after_first_accel[1]), after_second_accel[0]),
                              after_first_accel[1] + after_second_accel[1])
        after_full_speed = get_matrix(v_x, v_y, v_yaw, self.time_limit - 3)
        after_full_speed = (after_second_accel[0] + np.dot(rotation(after_second_accel[1]), after_full_speed[0]),
                            after_second_accel[1] + after_full_speed[1])
        after_deceleration = get_matrix(v_x / 2, v_y / 2, v_yaw / 2, 1)
        after_deceleration = (after_full_speed[0] + np.dot(rotation(after_full_speed[1]), after_deceleration[0]),
                              after_full_speed[1] + after_deceleration[1])

        # back to x,y,yaw format
        correct_pose = (after_deceleration[0][0], after_deceleration[0][1], after_deceleration[1])

        # we need to handle targets cleverly or we will have divisions by 0
        if v_x == 0 and v_y == 0:
            trans_target = 1
        else:
            # Pythagoras
            trans_target = math.sqrt(correct_pose[0] ** 2 + correct_pose[1] ** 2)
        # always take tau as meassurement
        rot_target = math.tau

        # Pythagoras
        trans_error_abs = math.sqrt((correct_pose[0] - current_pose[0]) ** 2 + (correct_pose[1] - current_pose[1]) ** 2)
        # yaw is split in continuous sin and cos components
        rot_error_abs = abs(math.sin(correct_pose[2]) - math.sin(current_pose[2])) + abs(
            math.cos(correct_pose[2]) - math.cos(current_pose[2]))
        pose_cost = ((trans_error_abs / trans_target) + (rot_error_abs / rot_target)) / 2

        print(f"x goal {round(correct_pose[0], 2)} cur {round(current_pose[0], 2)}")
        print(f"y goal {round(correct_pose[1], 2)} cur {round(current_pose[1], 2)}")
        print(f"yaw goal {round(correct_pose[2], 2)} cur {round(current_pose[2], 2)}")

        # scale to [0-1]
        if pose_cost / 1 > 1:
            print("cutting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        pose_cost = min(1, pose_cost)

        # if error higher than 30% we will stop. there is also always some error from start and stop taking some time
        didnt_move = pose_cost > 0.30
        if didnt_move:
            print("didn't move")

        return didnt_move, pose_cost, (correct_pose, current_pose)

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
        #self.sim.set_self_collision(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.set_cmd_vel(0.1, 0, 0)
        # set arms correctly
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LElbow", "RElbow", "LShoulderPitch", "RShoulderPitch"]
        joint_command_msg.positions = [math.radians(35.86), math.radians(-36.10), math.radians(75.27),
                                       math.radians(-75.58)]
        self.sim.set_joints(joint_command_msg)
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
        #self.sim.set_self_collision(True)
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
