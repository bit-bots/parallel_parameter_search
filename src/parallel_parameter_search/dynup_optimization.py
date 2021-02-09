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
from sensor_msgs.msg import Imu

from parallel_parameter_search.utils import fused_from_quat


class AbstractDynupOptimization(AbstractRosOptimization):
    def __init__(self, namespace, gui, robot, direction, externalforce, sim_type, stability=False, foot_link_names=(), multi_objective=False):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot)
        self.direction = direction
        self.externalforce = externalforce
        self.multi_objective = multi_objective
        self.sim_type = sim_type
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names, terrain=stability, field=False)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot)
        else:
            print(f'sim type {sim_type} not known')
        # load dynup params
        load_yaml_to_param(self.namespace, 'bitbots_dynup',
                           '/config/dynup_sim.yaml',
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
        self.dynamixel_controller_pub = rospy.Publisher(self.namespace + "/DynamixelController/command", JointCommand)
        self.number_of_iterations = 10
        self.time_limit = 20
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
        self.dynup_complete = False

        self.head_ground_time = 0
        self.rise_phase_time = 0
        self.total_trial_length = 0

        self.max_head_height = 0
        self.min_pitch = 0
        self.min_fused_pitch = 0
        self.imu_offset_sum = 0
        self.trunk_y_offset_sum = 0
        self.trial_duration = 0
        self.trial_running = False
        self.dynup_params = {}

        self.dynup_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'dynup/', timeout=60)
        self.trunk_pitch_client = dynamic_reconfigure.client.Client(
            self.namespace + '/' + 'dynup/pid_trunk_pitch/', timeout=60)
        self.trunk_roll_client = dynamic_reconfigure.client.Client(
            self.namespace + '/' + 'dynup/pid_trunk_roll/', timeout=60)

        self.dynup_step_done = False

    def result_cb(self, msg):
        if msg.result.successful:
            self.dynup_complete = True
            rospy.logerr("Dynup complete.")
        else:
            rospy.logerr("Dynup was cancelled.")

    def command_cb(self, msg):
        self.dynup_step_done = True

    def objective(self, trial):
        # for testing transforms
        while False:
            self.sim.set_robot_pose_rpy([0, 0, 1], [0.0, 0.0, 0.4])
            self.sim.step_sim()
            pos, rpy = self.sim.get_robot_pose_rpy()
            print(f"x: {round(pos[0], 2)}")
            print(f"y: {round(pos[1], 2)}")
            print(f"z: {round(pos[2], 2)}")
            print(f"roll: {round(rpy[0], 2)}")
            print(f"pitch: {round(rpy[1], 2)}")
            print(f"yaw: {round(rpy[2], 2)}")
            time.sleep(1)

        self.suggest_params(trial)

        self.dynup_params = rospy.get_param(self.namespace + "/dynup")
        # self.sim.randomize_terrain(0.01)
        self.reset()
        success = self.run_attempt()

        # scoring
        # better score for lifting the head higher, in cm
        head_score = 100 - self.max_head_height * 100
        pitch_score = self.min_pitch
        fused_ptich_score = self.min_fused_pitch
        # only divide by the frames we counted
        mean_imu_offset = 0
        if self.non_gimbal_frames > 0:
            mean_imu_offset = math.degrees(self.imu_offset_sum / self.non_gimbal_frames)
        mean_y_offset = self.trunk_y_offset_sum / (self.sim.get_time() - self.start_time)
        trial_failed_loss = self.total_trial_length - (self.sim.get_time() - self.start_time)
        speed_loss = self.sim.get_time() - self.start_time
        success_loss = 100 if not success else self.total_trial_length
        print(f"Head height: {head_score}")
        print(f"imu offset: {mean_imu_offset}")
        print(f"trunk y: {mean_y_offset}")
        print(f"trail fail: {200 * trial_failed_loss}")
        print(f"speed: {speed_loss}")
        print(f"success loss: {success_loss}")

        if self.multi_objective:
            # turn into multi loss problem
            # max head
            # min pitch
            # min imu offset
            # min planned time
            # max survive time
            # in einem zweiten versuch dann mit stabilisierung
            # score ob mans bis in die hocke geschafft hat
            return [head_score, success_loss, mean_imu_offset]
        else:
            #todo vlt lieber die angular velocities nehmen statt imu offsets
            #todo falls er zu sehr auf zeit optimiert und das in der echten welt nicht mehr klappt, dann den zeit wert aus der score funktion nehmen oder kleiner machen
            return head_score + success_loss + mean_imu_offset  # + mean_y_offset + 200 * trial_failed_loss + speed_loss
            #return fused_ptich_score + success_loss + mean_imu_offset



    def run_attempt(self):
        self.trial_running = True
        msg = DynUpActionGoal()
        msg.goal.direction = self.direction
        self.dynup_request_pub.publish(msg)
        self.start_time = self.sim.get_time()
        end_time = self.sim.get_time() + self.time_limit
        # reset params
        self.max_head_height = 0
        self.min_pitch = 90
        self.min_fused_pitch = 90
        self.imu_offset_sum = 0
        self.trunk_y_offset_sum = 0
        self.non_gimbal_frames = 0

        while not rospy.is_shutdown():
            if not self.externalforce == 0 and self.sim_type == 'pybullet':
                force_vector = [random.randrange(self.externalforce),
                                random.randrange(self.externalforce),
                                random.randrange(self.externalforce)]
                self.sim.apply_force(-1, force_vector, [0, 0, 0])
            self.sim.step_sim()

            # calculate loss
            pos, rpy = self.sim.get_robot_pose_rpy()
            # remember the closest we got to be straight, but dont take negative values, this would be overshooting
            self.min_pitch = max(0, min(self.min_pitch, math.degrees(rpy[1])))  # todo check gimbal issues
            #print(math.degrees(rpy[1]))
            pos, quat = self.sim.get_robot_pose()
            fused_roll, fused_pitch, fused_yaw, hemi = fused_from_quat(quat)
            #only take values which are in positive hemi. otherwise we take values where the robot is tilted more than 90°
            if hemi == 1:
                self.min_fused_pitch = max(0, min(self.min_fused_pitch, math.degrees(fused_pitch)))
            #print(math.degrees(fused_roll), math.degrees(fused_pitch), math.degrees(fused_yaw), hemi)

            imu_frame_error = 0
            imu_frame_error_parts = 0
            # only account for pitch in the last phase
            if self.sim.get_time() - self.start_time > self.rise_phase_time:
                imu_frame_error += abs(fused_pitch)
                imu_frame_error_parts += 1
            imu_frame_error += abs(fused_yaw)
            imu_frame_error += abs(fused_roll)
            imu_frame_error_parts += 2
            self.imu_offset_sum += imu_frame_error / imu_frame_error_parts
            self.non_gimbal_frames += 1
            self.trunk_y_offset_sum += abs(pos[1])

            head_position = self.sim.get_link_pose("head")
            self.max_head_height = max(self.max_head_height, head_position[2])

            # early abort if robot falls, but not in first phase where head is always close to ground
            if self.sim.get_time() - self.start_time > self.head_ground_time and head_position[2] < 0.15:
                print("head")
                return False

            # avoid simulator glitches
            if head_position[2] > 1:
                print("too high")
                return False

            # early abort if IK glitch occurs
            if self.sim.get_joint_position("RAnkleRoll") > 0.9 or self.sim.get_joint_position("RAnkleRoll") < -0.9:
                print("Ik bug")
                return False

            if self.dynup_complete:
                # dont waste time waiting for the time limit to arrive
                end_time = self.sim.get_time()
                self.dynup_complete = False
            # wait a bit after finishing to check if the robot falls during this time
            if self.sim.get_time() - self.start_time > self.time_limit or self.sim.get_time() - end_time > 3:
                self.trial_duration = self.sim.get_time() - self.start_time
                self.dynup_complete = False
                self.trial_running = False
                # also check if robot reached correct height
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
                self.head_ground_time = self.dynup_params["time_hands_side"] + \
                                        self.dynup_params["time_hands_rotate"] + \
                                        self.dynup_params["time_foot_close"] + \
                                        self.dynup_params["time_hands_front"] + \
                                        self.dynup_params["time_foot_ground_front"] + \
                                        self.dynup_params["time_torso_45"]
                self.rise_phase_time = self.head_ground_time + \
                                       self.dynup_params["time_to_squat"]
                self.total_trial_length = self.rise_phase_time + \
                                          self.dynup_params["wait_in_squat_front"] + \
                                          self.dynup_params["rise_time"] + 3
            elif self.direction == "back":
                self.head_ground_time = self.dynup_params["time_legs_close"] + \
                                        self.dynup_params["time_foot_ground_back"]
                self.rise_phase_time = self.head_ground_time + \
                                       self.dynup_params["time_full_squat_hands"] + \
                                       self.dynup_params["time_full_squat_legs"]
                self.total_trial_length = self.rise_phase_time + \
                                          self.dynup_params["wait_in_squat_back"] + \
                                          self.dynup_params["rise_time"] + 3
            else:
                print(f"Direction {self.direction} not known")

            # trial continues for 3 sec after dynup completes
        except KeyError:
            rospy.logwarn("Parameter server not available yet, continuing anyway.")
            self.rise_phase_time = 100  # todo: hacky high value that should never be reached. But it works
        # reset simulation
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        time = self.sim.get_time()

        while self.sim.get_time() - time < 2:
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
            self.sim.step_sim()
        self.reset_position()
        self.sim.set_gravity(True)
        time = self.sim.get_time()
        while not self.sim.get_time() - time > 2:
            self.sim.step_sim()


class WolfgangOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, direction, externalforce, sim_type='pybullet', multi_objective=False):
        super(WolfgangOptimization, self).__init__(namespace, gui, 'wolfgang', direction, externalforce, sim_type, multi_objective=multi_objective)
        self.reset_height_offset = 0.1

    def suggest_params(self, trial):
        node_param_dict = {}

        def pid_params(name, client, p, i, d, i_clamp):
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

        # todo maybe use step or log? robot has limited precision in space and time anyway
        def add(name, min_value, max_value):
            node_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            node_param_dict[name] = value
            trial.set_user_attr(name, value)

        add("leg_min_length", 0.2, 0.25)
        add("arm_side_offset", 0.05, 0.2)
        #add("trunk_x", -0.1, 0.1)
        fix("trunk_x_final", 0)
        add("rise_time", 0, 1)

        fix("stabilizing", False)
        #todo bei stabilization die zeiten von den letzten bewegungen nochmal mit optimieren?
        #todo currently dynup only stabilizes after wait in squat time, maybe start before this
        #todo maybe only learn pitch first and roll in a second optimization?
        #todo maybe use fused angles
        #pid_params("trunk_pitch", self.trunk_pitch_client, (0, 2), (0, 4), (0, 0.1), (-1, 1))
        #pid_params("trunk_roll", self.trunk_roll_client, (0, 2), (0, 4), (0, 0.1), (-1, 1))

        # these are basically goal position variables, that the user has to define
        fix("trunk_height", self.trunk_height)
        fix("trunk_pitch", 0)
        fix("foot_distance", 0.2)

        if self.direction == "front":
            add("trunk_x_front", -0.1, 0.1)
            add("max_leg_angle", 0, 90)
            add("trunk_overshoot_angle_front", -90, 0)
            add("time_hands_side", 0, 1)
            add("time_hands_rotate", 0, 1)
            add("time_foot_close", 0, 1)
            add("time_hands_front", 0, 1)
            add("time_torso_45", 0, 1)
            add("time_to_squat", 0, 1)
            add("wait_in_squat_front", 0, 2)
        elif self.direction == "back":
            add("trunk_x_back", -0.1, 0.1)
            add("hands_behind_back_x", 0.0, 0.4)
            add("hands_behind_back_z", 0.0, 0.4)
            add("trunk_height_back", 0.0, 0.4)
            add("trunk_forward", 0.0, 0.1)
            add("foot_angle", 0.0, 90)
            add("trunk_overshoot_angle_back", 0.0, 90)
            add("time_legs_close", 0, 1)
            add("time_foot_ground_back", 0, 1)
            add("time_full_squat_hands", 0, 1)
            add("time_full_squat_legs", 0, 1)
            add("wait_in_squat_back", 0, 1)
        else:
            print(f"direction {self.direction} not specified")

        self.set_params(node_param_dict, self.dynup_client)
        return


class NaoOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, direction, externalforce, sim_type='pybullet', multi_objective=False):
        super(NaoOptimization, self).__init__(namespace, gui, 'nao', direction, externalforce, sim_type, multi_objective=multi_objective)
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
