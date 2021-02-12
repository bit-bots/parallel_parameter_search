#!/usr/bin/env python3
import random
import time

import dynamic_reconfigure.client
from actionlib_msgs.msg import GoalID
from bitbots_msgs.msg import DynUpActionGoal, DynUpActionResult, JointCommand

import math

import roslaunch
import rospkg
import rospy
import tf

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState

from parallel_parameter_search.utils import fused_from_quat
from statistics import mean

class AbstractDynupOptimization(AbstractRosOptimization):
    def __init__(self, namespace, gui, robot, direction, sim_type, stability=False, foot_link_names=(),
                 multi_objective=False, real_robot=False, repetitions=1):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot)
        self.direction = direction
        self.multi_objective = multi_objective
        self.sim_type = sim_type
        self.stability = stability
        self.real_robot = real_robot
        self.repetitions = repetitions
        if not self.real_robot:
            if sim_type == 'pybullet':
                urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
                self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                       foot_link_names=foot_link_names, terrain=False, field=False)
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
        self.robot_state_publisher = roslaunch.core.Node('robot_state_publisher', 'robot_state_publisher',
                                                         'robot_state_publisher',
                                                         namespace=self.namespace)
        self.dynup_node.remap_args = [("/tf", "tf"), ("dynup_motor_goals", "DynamixelController/command"),
                                      ("/tf_static", "tf_static"), ("/clock", "clock")]
        self.robot_state_publisher.remap_args = [("/tf", "tf"), ("/tf_static", "tf_static"), ("/clock", "clock")]
        load_yaml_to_param("/robot_description_kinematics", robot + '_moveit_config',
                           '/config/kinematics.yaml', self.rospack)
        self.launch.launch(self.robot_state_publisher)
        self.launch.launch(self.dynup_node)

        self.dynup_request_pub = rospy.Publisher(self.namespace + '/dynup/goal', DynUpActionGoal, queue_size=1)
        self.dynup_cancel_pub = rospy.Publisher(self.namespace + '/dynup/cancel', GoalID, queue_size=1)
        self.dynamixel_controller_pub = rospy.Publisher(self.namespace + "/DynamixelController/command", JointCommand,
                                                        queue_size=1)
        self.number_of_iterations = 10
        self.time_limit = 30
        self.time_difference = 0
        self.reset_height_offset = None
        if self.direction == "front":
            self.reset_rpy_offset = [0, math.pi / 2, 0]
        elif self.direction == "back":
            self.reset_rpy_offset = [0, -math.pi / 2, 0]
        else:
            print(f"direction {self.direction}")
            exit(0)
        self.trunk_height = 0.4  # rosparam.get_param(self.namespace + "/dynup/trunk_height")
        self.trunk_pitch = 0.0

        self.result_subscriber = rospy.Subscriber(self.namespace + "/dynup/result", DynUpActionResult, self.result_cb)
        self.command_sub = rospy.Subscriber(self.namespace + "/DynamixelController/command", JointCommand,
                                            self.command_cb)
        if self.real_robot:
            self.imu_subscriber = rospy.Subscriber("/imu/data", Imu, self.imu_cb)
            self.current_imu_msg = None
            self.joint_subscrbier = rospy.Subscriber("/joint_states", JointState, self.joint_cb)
            self.current_joint_msg = None
        self.dynup_complete = False

        self.hand_ground_time = 0
        self.rise_phase_time = 0
        self.in_squat_time = 0
        self.total_trial_length = 0

        self.max_torque = 0
        self.torque_sum = 0
        self.torque_count = 0
        self.max_head_height = 0
        self.min_fused_pitch = 0
        self.imu_offset_sum = 0
        self.trial_duration = 0
        self.last_move_time = 0
        self.trial_running = False
        self.dynup_params = {}

        self.dynup_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'dynup/', timeout=60)
        self.trunk_pitch_client = dynamic_reconfigure.client.Client(
            self.namespace + '/' + 'dynup/pid_trunk_pitch/', timeout=60)
        self.trunk_roll_client = dynamic_reconfigure.client.Client(
            self.namespace + '/' + 'dynup/pid_trunk_roll/', timeout=60)

        self.dynup_step_done = False

        if self.real_robot:
            self.joint_names = ["LAnklePitch", "LAnkleRoll", "LHipPitch", "LHipRoll", "LHipYaw", "LKnee", "RAnklePitch",
                                "RAnkleRoll", "RHipPitch", "RHipRoll", "RHipYaw", "RKnee", "LShoulderPitch",
                                "LShoulderRoll", "LElbow", "RShoulderPitch", "RShoulderRoll", "RElbow", "HeadPan",
                                "HeadTilt"]
        else:
            self.joint_names = self.sim.get_joint_names()

    def imu_cb(self, msg):
        self.current_imu_msg = msg

    def joint_cb(self, msg):
        self.current_joint_msg = msg

    def result_cb(self, msg):
        if msg.result.successful:
            self.dynup_complete = True
            rospy.logerr("Dynup complete.")
        else:
            rospy.logerr("Dynup was cancelled.")

    def command_cb(self, msg):
        self.dynup_step_done = True

    def objective(self, trial):
        self.suggest_params(trial, self.stability)

        head_score = []
        fused_pitch_score = []
        percentage_left = []
        mean_imu_offsets = []
        success_sums = []
        speed_loss = []
        mean_torque = []
        # try multiple times to make sure its working stably
        for i in range(self.repetitions):
            self.imu_offset_sum = 0
            self.frames = 0
            self.max_torque = 0
            self.torque_sum = 0
            self.torque_count = 0

            if self.stability:
                # tries with different forces
                attempts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-2, 0, 0), (0, -2, 0), (4, 0, 0), (0, 4, 0)]
            else:
                # only force free
                attempts = [(0, 0, 0)]

            successes = 0
            for attempt in attempts:
                # self.sim.randomize_terrain(0.01)
                self.reset()
                success = self.run_attempt(attempt)
                # get time of trial. use last time the robot was moving as point
                self.trial_duration = self.last_move_time - self.start_time

                if success:
                    successes += 1
                else:
                    break

            # scores
            # better score for lifting the head higher, in cm
            head_score.append(100 - self.max_head_height * 100)
            print(f"head{100 - self.max_head_height * 100}")
            fused_pitch_score.append(self.min_fused_pitch)
            print(f"min pitch {self.min_fused_pitch}")
            percentage_left.append(100 * (self.total_trial_length - self.trial_duration) / (self.total_trial_length))

            # only divide by the frames we counted
            mean_imu_offset = 0
            if self.frames > 0:
                mean_imu_offset = math.degrees(self.imu_offset_sum / self.frames)
            mean_imu_offsets.append(mean_imu_offset)

            success_sums.append(100 * (len(attempts) - successes))
            # can not just take trial_duration, since would be short if robot fell
            # can not just take total_trial_length, since this does not include standing stable after rising
            speed_loss.append(max(self.total_trial_length, self.trial_duration))

            # mean_torque.append(self.torque_sum / self.torque_count)

            # todo this speeds up a lot but is it correct to do so?
            #if successes == 0:
            #    break

        head_score = mean(head_score)
        fused_pitch_score = mean(fused_pitch_score)
        percentage_left = mean(percentage_left)
        mean_imu_offset = mean(mean_imu_offsets)
        speed_loss = mean(speed_loss)
        # mean_torque = max(mean_torque)
        success_sum = mean(success_sums)
        print(f"Head height: {head_score}")
        print(f"imu offset: {mean_imu_offset}")
        print(f"percentage left: {percentage_left}")
        print(f"success loss: {success_sum}")
        print(f"speed loss {speed_loss}")
        print(f"mean torque {mean_torque}")

        if self.multi_objective:
            # score ob mans bis in die hocke geschafft hat (ist eigentlich ungefähr fused_ptich_score)
            # todo do we also find solutions without the first two scores?
            return [success_sum, speed_loss, head_score, fused_pitch_score]
            # return [success_sum, speed_loss]
        else:
            # todo vlt lieber die angular velocities nehmen statt imu offsets
            # todo falls er zu sehr auf zeit optimiert und das in der echten welt nicht mehr klappt, dann den zeit wert aus der score funktion nehmen oder kleiner machen
            return head_score + success_sum + speed_loss + mean_imu_offset  # + mean_y_offset + 200 * trial_failed_loss + speed_loss
            # return fused_ptich_score + success_loss + mean_imu_offset

    def run_attempt(self, force_vector):
        self.trial_running = True
        msg = DynUpActionGoal()
        msg.goal.direction = self.direction
        self.dynup_request_pub.publish(msg)
        self.start_time = self.get_time()
        end_time = self.start_time + self.total_trial_length
        # reset params
        self.max_head_height = 0
        self.min_fused_pitch = 90

        while not rospy.is_shutdown():
            if not self.real_robot:
                # apply force only while dynup is running and not in waiting time after end
                if self.get_time() - self.start_time < end_time:
                    self.sim.apply_force(-1, force_vector, [0, 0, 0])
                self.sim.step_sim()
                pos, quat = self.sim.get_robot_pose()
            else:
                quat = self.current_imu_msg.orientation
            fused_roll, fused_pitch, fused_yaw, hemi = fused_from_quat(quat)
            # only take values which are in positive hemi. otherwise we take values where the robot is tilted more than 90°
            if hemi == 1:
                self.min_fused_pitch = max(0, min(self.min_fused_pitch, math.degrees(abs(fused_pitch))))

            imu_frame_error = 0
            imu_frame_error_parts = 0
            # only account for pitch in the last phase
            if self.get_time() - self.start_time > self.rise_phase_time:
                imu_frame_error += abs(fused_pitch)
                imu_frame_error_parts += 1
            imu_frame_error += abs(fused_yaw)
            imu_frame_error += abs(fused_roll)
            imu_frame_error_parts += 2
            self.imu_offset_sum += imu_frame_error / imu_frame_error_parts
            self.frames += 1

            if not self.real_robot:
                head_position = self.sim.get_link_pose("head")
                self.max_head_height = max(self.max_head_height, head_position[2])

            # compute torques
            # torques = self.get_joint_torques()
            # self.max_torque = max(self.max_torque, max(torques))
            # self.torque_sum += sum(torques)
            # self.torque_count += len(torques)

            if self.real_robot:
                angular_vel = self.current_imu_msg.angular_velocity
            else:
                _, angular_vel = self.sim.get_robot_velocity()
            # print(angular_vel[1])
            # early termination if robot falls. detectable on gyro pitch
            if self.direction == "front":
                # only after initial arm movement and not after reaching squat
                if angular_vel[1] > 0 and self.get_time() - self.start_time > self.hand_ground_time +0.5\
                        and fused_pitch > math.radians(45):
                    print("gyro")
                    return False
                # detect falling after reaching squat
                if self.min_fused_pitch == 0 and fused_pitch < math.radians(-45):
                    print("orientation")
                    return False
            else:
                # only after initial arm movement and not after reaching squat
                if angular_vel[1] < 0 and self.get_time() - self.start_time > self.hand_ground_time +0.5\
                        and fused_pitch < math.radians(-45):
                    print("gyro")
                    return False
                # detect falling after reaching squat
                if self.min_fused_pitch == 0 and fused_pitch > math.radians(45):
                    print("orientation")
                    return False

            # early termination if robot falls, but not in first phase where head is always close to ground
            # (+ some time to move)
            if not self.real_robot and self.get_time() - self.start_time > self.hand_ground_time + 0.5 and \
                    head_position[2] < 0.2:
                print("head")
                return False

            # avoid simulator glitches
            if not self.real_robot and head_position[2] > 1:
                print("too high")
                return False

            # early abort if IK glitch occurs
            if abs(self.get_joint_position("RAnkleRoll")) > 0.9 or abs(self.get_joint_position("LHipYaw")) > 0.9:
                print("Ik bug")
                return False

            if self.dynup_complete:
                # dont waste time waiting for the time limit to arrive
                end_time = self.get_time()
                self.dynup_complete = False

            if self.get_time() - self.start_time > self.time_limit:
                return False

            if angular_vel[0] > 0.01 or angular_vel[1] > 0.01:
                self.last_move_time = self.get_time()
                # print(angular_vel)

            # wait for having no angular velocities for some time, to make sure the robot is not falling anymore
            if self.get_time() >= end_time and self.get_time() - self.last_move_time > 3:
                self.dynup_complete = False
                self.trial_running = False
                # also check if robot reached correct height
                if self.real_robot:
                    suc = input("Success? y/n")
                    return suc == "y"
                else:
                    if pos[2] > self.trunk_height * 0.8:
                        return True
                    else:
                        return False

        while not self.dynup_step_done:
            # give time to Dynup to compute its response
            # use wall time, as ros time is standing still
            time.sleep(0.0001)
        self.dynup_step_done = False

    def reset_position(self):
        height = self.reset_height_offset
        pitch = self.trunk_pitch

        if self.sim_type == "pybullet":
            (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                    self.reset_rpy_offset[1] + pitch,
                                                                    self.reset_rpy_offset[2])

            self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))
        else:
            angle = math.pi / 2
            x = 0
            y = 1
            z = 0
            self.sim.reset_robot_pose((0, 0, height), (angle, x, y, z))

    def reset(self):
        # reset Dynup. send emtpy message to just cancel all goals
        self.dynup_cancel_pub.publish(GoalID())

        try:
            if self.direction == "front":
                self.hand_ground_time = self.dynup_params["time_hands_side"] + \
                                        self.dynup_params["time_hands_rotate"] + \
                                        self.dynup_params["time_foot_close"] + \
                                        self.dynup_params["time_hands_front"] + \
                                        self.dynup_params["time_foot_ground_front"] + \
                                        self.dynup_params["time_torso_45"]
                self.rise_phase_time = self.hand_ground_time + \
                                       self.dynup_params["time_to_squat"]
                self.in_squat_time = self.rise_phase_time + \
                                     self.dynup_params["wait_in_squat_front"]
                self.total_trial_length = self.in_squat_time + \
                                          self.dynup_params["rise_time"]
            elif self.direction == "back":
                self.hand_ground_time = self.dynup_params["time_legs_close"] + \
                                        self.dynup_params["time_foot_ground_back"]
                self.rise_phase_time = self.hand_ground_time + \
                                       self.dynup_params["time_full_squat_hands"] + \
                                       self.dynup_params["time_full_squat_legs"]
                self.in_squat_time = self.rise_phase_time + \
                                     self.dynup_params["wait_in_squat_back"]
                self.total_trial_length = self.in_squat_time + \
                                          self.dynup_params["rise_time"]
            else:
                print(f"Direction {self.direction} not known")

            # trial continues for 3 sec after dynup completes
        except KeyError:
            rospy.logwarn("Parameter server not available yet, continuing anyway.")
            self.rise_phase_time = 100  # todo: hacky high value that should never be reached. But it works

        if self.real_robot:
            input("Please hold robot in the air and then press any key")
        else:
            # reset simulation
            self.sim.set_gravity(False)
            self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        time = self.get_time()

        while self.get_time() - time < 2:
            msg = JointCommand()
            msg.joint_names = ["HeadPan", "HeadTilt", "LElbow", "LShoulderPitch", "LShoulderRoll", "RElbow",
                               "RShoulderPitch", "RShoulderRoll", "LHipYaw", "LHipRoll", "LHipPitch", "LKnee",
                               "LAnklePitch", "LAnkleRoll", "RHipYaw", "RHipRoll", "RHipPitch", "RKnee", "RAnklePitch",
                               "RAnkleRoll"]
            if self.direction == "back":
                msg.positions = [0, 0, 0.79, 0, 0, -0.79, 0, 0, -0.01, 0.06, 0.47, 1.01, -0.45, 0.06, 0.01, -0.06,
                                 -0.47,
                                 -1.01, 0.45, -0.06]  # walkready
            elif self.direction == "front":
                msg.positions = [0, 0.78, 0.78, 1.36, 0, -0.78, -1.36, 0, 0.11, 0.07, -0.19, 0.23, -0.63, 0.07, 0.11,
                                 -0.07,
                                 0.19, -0.23, 0.63, -0.07]  # falling_front
            self.dynamixel_controller_pub.publish(msg)
            if not self.real_robot:
                self.sim.step_sim()
        if self.real_robot:
            input("Please put the robot on the ground and then press any key")
        else:
            self.reset_position()
            self.sim.set_gravity(True)
            time = self.get_time()
            while not self.get_time() - time > 2:
                self.sim.step_sim()

    def pid_params(self, trial, name, client, p, i, d, i_clamp):
        pid_dict = {"p": trial.suggest_uniform(name + "_p", p[0], p[1]),
                    "d": trial.suggest_uniform(name + "_d", i[0], i[1]),
                    "i": trial.suggest_uniform(name + "_i", d[0], d[1]),
                    "i_clamp_min": i_clamp[0],
                    "i_clamp_max": i_clamp[1]}
        if isinstance(client, list):
            for c in client:
                self.set_params(pid_dict, client)
        else:
            self.set_params(pid_dict, client)

    def get_joint_torques(self):
        torques = []
        for name in self.joint_names:
            joint = self.sim.sim.joints[name]
            torques.append(abs(joint.get_torque()))
        return torques

    def get_time(self):
        if self.real_robot:
            return rospy.Time.now().to_sec()
        else:
            return self.sim.get_time()

    def get_joint_position(self, name):
        if self.real_robot:
            index = self.current_joint_msg.name.index(name)
            return self.current_joint_msg.position[index]
        else:
            return self.sim.get_joint_position(name)


class WolfgangOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, direction, sim_type='pybullet', multi_objective=False, stability=False,
                 real_robot=False, repetitions=1):
        super(WolfgangOptimization, self).__init__(namespace, gui, 'wolfgang', direction, sim_type,
                                                   multi_objective=multi_objective, stability=stability,
                                                   real_robot=real_robot, repetitions=repetitions)
        self.reset_height_offset = 0.1

    def suggest_params(self, trial, stabilization):
        load_yaml_to_param(self.namespace, 'bitbots_dynup',
                           '/config/dynup_optimization.yaml',
                           self.rospack)
        self.dynup_params = rospy.get_param(self.namespace + "/dynup")
        self.dynup_params.pop("pid_trunk_roll")
        self.dynup_params.pop("pid_trunk_pitch")

        def add(name, min_value, max_value, step=None, log=False):
            self.dynup_params[name] = trial.suggest_float(name, min_value, max_value, step=step, log=log)

        def fix(name, value):
            self.dynup_params[name] = value
            trial.set_user_attr(name, value)

        step_cartesian = 0.001
        step_time = 1 / 250
        step_angle = 0.01
        if stabilization:
            # activate stabilization
            fix("stabilizing", True)
            # todo bei stabilization die zeiten von den letzten bewegungen nochmal mit optimieren?
            # todo currently dynup only stabilizes after wait in squat time, maybe start before this
            self.pid_params(trial, "trunk_pitch", self.trunk_pitch_client, (-2, 2), (-4, 4), (-0.1, 0.1), (-2, 2))
            # self.pid_params(trial, "trunk_roll", self.trunk_roll_client, (-2, 2), (-4, 4), (0, 0.1), (-2, 2))
        else:
            fix("stabilizing", False)
            # we are not more precise than 1mm or one loop cycle (simulator runs at 240Hz)
            add("leg_min_length", 0.2, 0.25, step=step_cartesian)
            add("arm_side_offset", 0.05, 0.2, step=step_cartesian)
            # add("trunk_x", -0.1, 0.1)
            fix("trunk_x_final", 0)
            add("rise_time", 0, 2, step=step_time)

            # these are basically goal position variables, that the user has to define
            fix("trunk_height", self.trunk_height)
            fix("trunk_pitch", 0)
            fix("foot_distance", 0.2)
            fix("hand_walkready_pitch", -60)

            if self.direction == "front":
                add("trunk_x_front", -0.1, 0.1, step=step_cartesian)
                add("max_leg_angle", 0, 90, step=step_angle)
                add("trunk_overshoot_angle_front", -90, 0, step=step_angle)
                add("time_hands_side", 0, 1, step=step_time)
                add("time_hands_rotate", 0, 1, step=step_time)
                add("time_foot_close", 0, 1, step=step_time)
                add("time_hands_front", 0, 1, step=step_time)
                add("time_foot_ground_front", 0, 1, step=step_time)
                add("time_torso_45", 0, 1, step=step_time)
                add("time_to_squat", 0, 1, step=step_time)
                add("wait_in_squat_front", 0, 2, step=step_time)
            elif self.direction == "back":
                add("trunk_x_back", -0.1, 0.1, step=step_cartesian)
                add("hands_behind_back_x", 0.0, 0.4, step=step_cartesian)
                add("hands_behind_back_z", 0.0, 0.4, step=step_cartesian)
                add("trunk_height_back", 0.0, 0.4, step=step_cartesian)
                add("trunk_forward", 0.0, 0.1, step=step_cartesian)
                add("foot_angle", 0.0, 90, step=step_angle)
                add("trunk_overshoot_angle_back", 0.0, 90, step=step_angle)
                add("time_legs_close", 0, 1, step=step_time)
                add("time_foot_ground_back", 0, 1, step=step_time)
                add("time_full_squat_hands", 0, 1, step=step_time)
                add("time_full_squat_legs", 0, 1, step=step_time)
                add("wait_in_squat_back", 0, 1, step=step_time)
            else:
                print(f"direction {self.direction} not specified")

        self.set_params(self.dynup_params, self.dynup_client)
        return


class NaoOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, direction, sim_type='pybullet', multi_objective=False, stability=False):
        super(NaoOptimization, self).__init__(namespace, gui, 'nao', direction, sim_type,
                                              multi_objective=multi_objective, stability=stability)
        self.reset_height_offset = 0.005

    def suggest_params(self, trial):
        node_param_dict = {}

        def add(name, min_value, max_value):
            node_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            node_param_dict[name] = value
            trial.set_user_attr(name, value)

        add("foot_distance", 0.106, 0.25)
        add("leg_min_length", 0.1, 0.2)
        add("arm_side_offset", 0.05, 0.2)
        add("trunk_x", -0.2, 0.2)
        add("rise_time", 0, 1)

        # these are basically goal position variables, that the user has to define
        fix("trunk_height", 0.4)
        fix("trunk_pitch", 0)

        if self.direction == "front":
            # add("max_leg_angle", 20, 80)
            # add("trunk_overshoot_angle_front", -90, 0)
            # add("time_hands_side", 0, 1)
            # add("time_hands_rotate", 0, 1)
            # add("time_foot_close", 0, 1)
            # add("time_hands_front", 0, 1)
            # add("time_torso_45", 0, 1)
            # add("time_to_squat", 0, 1)
            # add("wait_in_squat_front", 0, 2)
            pass
        elif self.direction == "back":
            pass  # todo
        else:
            print(f"direction {self.direction} not specified")

        self.set_params(node_param_dict, self.dynup_client)
        return


def load_robot_param(namespace, rospack, name):
    rospy.set_param(namespace + '/robot_type_name', name)
    set_param_to_file(namespace + "/robot_description", name + '_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml', rospack)
