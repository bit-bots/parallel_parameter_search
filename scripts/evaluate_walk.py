#!/usr/bin/env python3
from bitbots_quintic_walk import PyWalk

from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization
from dataclasses import make_dataclass
import pandas as pd


class EvaluateWalk(AbstractWalkOptimization):

    def __init__(self, namespace, gui, robot, sim_type="webots", foot_link_names=()):
        super(EvaluateWalk, self).__init__(namespace, robot, False, config_name="_evaluation")
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot, world="walk_optim_" + robot, ros_active=False)
        else:
            print(f'sim type {sim_type} not known')
        self.robot = robot
        self.time_limit = 10
        self.repetitions = 1

        self.trunk_height = self.param_yaml_data["walking"]["engine"]["trunk_height"]
        self.trunk_pitch = self.param_yaml_data["walking"]["engine"]["trunk_pitch"]
        self.trunk_pitch_p_coef_forward = self.param_yaml_data["walking"]["engine"]["trunk_pitch_p_coef_forward"]
        self.trunk_pitch_p_coef_turn = self.param_yaml_data["walking"]["engine"]["trunk_pitch_p_coef_turn"]
        self.reset_height_offset = 0.012

    def evaluate_walk(self):
        Result = make_dataclass("Result",
                                [("v_x", float), ("v_y", float), ("v_yaw", float), ("fall", bool), ("pose_obj", float),
                                 ("v_x_multiplier", float), ("v_y_multiplier", float), ("v_yaw_multiplier", float)])
        maximal_speeds = []
        results = []
        self.reset()

        def test_speed(start_speed, increase):
            speed = start_speed
            while True:
                falls = 0
                speed[0] += increase[0]
                speed[1] += increase[1]
                speed[2] += increase[2]
                for i in range(self.repetitions):
                    self.reset_position()
                    fall, didnt_move, pose_obj, orientation_obj, gyro_obj, end_poses = \
                        self.evaluate_direction(*speed, 1, self.time_limit)
                    goal_end_pose = end_poses[0]
                    actual_end_pose = end_poses[1]
                    real_speed_multipliers = []
                    if goal_end_pose[0] == 0:
                        real_speed_multipliers.append(1)
                    else:
                        real_speed_multipliers.append(actual_end_pose[0] / goal_end_pose[0])
                    if goal_end_pose[1] == 0:
                        real_speed_multipliers.append(1)
                    else:
                        real_speed_multipliers.append(actual_end_pose[1] / goal_end_pose[1])
                    if goal_end_pose[2] == 0:
                        real_speed_multipliers.append(1)
                    else:
                        real_speed_multipliers.append(actual_end_pose[2] / goal_end_pose[2])
                    results.append(Result(*speed, fall, pose_obj, *real_speed_multipliers))
                    falls += fall

                if falls == self.repetitions:
                    print(f"Fall at {speed}")
                    maximal_speeds.append(speed)
                    break

        test_speed([0, 0, 0], [0.05, 0, 0])
        test_speed([0, 0, 0], [-0.05, 0, 0])
        test_speed([0, 0, 0], [0, 0.05, 0])
        test_speed([0, 0, 0], [0, 0, 0.05])
        test_speed([0, 0, 0], [0.05, 0.05, 0])

        print(maximal_speeds)
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_pickle(f"./walk_evaluation_{self.robot}.pkl")


walk_evaluation = EvaluateWalk("worker", True, "wolfgang")
walk_evaluation.evaluate_walk()

walk_evaluation.sim.close()
