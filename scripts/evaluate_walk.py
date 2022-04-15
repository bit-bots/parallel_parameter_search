#!/usr/bin/env python3
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization
from dataclasses import make_dataclass
import pandas as pd
from bitbots_msgs.msg import JointCommand
import math
import numpy as np

class EvaluateWalk(AbstractWalkOptimization):

    def __init__(self, namespace, gui, robot, sim_type="webots", foot_link_names=()):
        super().__init__(robot, config_name=f"walking_{robot}_simulator")
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.node, gui, robot, world="optimization_" + robot, ros_active=False)
        else:
            print(f'sim type {sim_type} not known')
        self.robot = robot
        self.time_limit = 10
        self.repetitions = 1

        for parameter_msg in self.walk_parameters:
            if parameter_msg.name == "engine.trunk_pitch":
                self.trunk_pitch = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_height":
                self.trunk_height = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_pitch_p_coef_forward":
                self.trunk_pitch_p_coef_forward = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_pitch_p_coef_turn":
                self.trunk_pitch_p_coef_turn = parameter_msg.value.double_value
        if self.trunk_pitch is None or self.trunk_height is None or self.trunk_pitch_p_coef_turn is None or \
                self.trunk_pitch_p_coef_forward is None:
            print("Parameters not set correctly")
            exit()
        self.reset_height_offset = 0.24

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["Shoulder-L [shoulder]", "Shoulder-R [shoulder]", "UpperArm-L",
                                         "UpperArm-R", "LowerArm-L", "LowerArm-R"]
        joint_command_msg.positions = [math.radians(60.0), math.radians(-60.0), math.radians(10.0),
                                       math.radians(-10.0), math.radians(-135.0), math.radians(135.0)]
        return joint_command_msg

    def evaluate_walk(self):
        Result = make_dataclass("Result",
                                [("v_x", float), ("v_y", float), ("v_yaw", float), ("fall", bool), ("pose_obj", float),
                                 ("x_speed_multiplier", float), ("y_speed_multiplier", float),
                                 ("yaw_speed_multiplier", float)])
        maximal_speeds = []
        results = []
        self.reset()

        def test_speed(start_speed, increase, direction):
            self.reset()
            speed = start_speed
            while True:
                falls = 0
                speed[0] += increase[0]
                speed[1] += increase[1]
                speed[2] += increase[2]
                for i in range(self.repetitions):
                    self.reset_position()
                    fall, pose_obj, orientation_obj, gyro_obj, end_poses = \
                        self.evaluate_direction(*speed, self.time_limit)
                    goal_end_pose = end_poses[0]
                    actual_end_pose = end_poses[1]

                    distance_travelled_in_correct_direction = np.dot(direction, np.array(actual_end_pose))
                    # need to use factor as we have acceleration and deccaleration phases
                    actual_speed = distance_travelled_in_correct_direction / 7.5
                    print(f"speed {actual_speed}")

                    real_speed_multipliers = []
                    #if goal_end_pose[0] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[0] / actual_end_pose[0])
                    #if goal_end_pose[1] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[1] / actual_end_pose[1])
                    #if goal_end_pose[2] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[2] / actual_end_pose[2])
                    #results.append(Result(*speed, fall, pose_obj, *real_speed_multipliers))
                    falls += fall

                if falls == self.repetitions:
                    print(f"Fall at {speed}")
                    print("")
                    maximal_speeds.append(speed)
                    break

        #test_speed([0.1, 0, 0], [0.05, 0, 0], [1,0,0])
        #test_speed([-0.1, 0, 0], [-0.05, 0, 0], [-1,0,0])
        #test_speed([0, 0.05, 0], [0, 0.025, 0], [0,1,0])
        test_speed([0, 0, 1], [0, 0, 0.25], [0,0,1])

        #test_speed([0, 0, 0], [0.05, 0.05, 0])

        print(maximal_speeds)
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_pickle(f"./walk_evaluation_{self.robot}.pkl")


walk_evaluation = EvaluateWalk("worker", True, "mrl_hsl")
walk_evaluation.evaluate_walk()

walk_evaluation.sim.close()
