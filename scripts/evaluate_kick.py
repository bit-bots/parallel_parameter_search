#!/usr/bin/env python3

import math
from itertools import cycle

from PIL import Image, ImageDraw

from bitbots_msgs.msg import JointCommand

from parallel_parameter_search.kick_optimization import WolfgangKickEngineOptimization


class KickRunner(WolfgangKickEngineOptimization):
    def __init__(self):
        super().__init__(namespace='evaluator', gui=False, sim_type='webots')

    def set_goal(self, ball_x, ball_y, angle, speed):
        msg = self.get_kick_goal_msg(ball_x, ball_y, angle, speed)
        self.kick.set_goal(msg, self.sim.get_joint_state_msg())
        self.sim.place_ball(ball_x, ball_y)
        self.sim.step_sim()
        self.sim.step_sim()

    def set_kick_params(self, params: dict):
        self.kick.set_params(params)

    def perform(self, timeout=120):
        start_time = self.sim.get_time()
        kick_finished = False
        self.last_time = self.sim.get_time()
        # wait till kick is finished
        while not kick_finished and self.sim.get_time() - start_time < timeout:
            current_time = self.sim.get_time()
            joint_command = self.kick.step(current_time - self.last_time,
                                           self.sim.get_joint_state_msg())  # type: JointCommand
            print(joint_command)
            if len(joint_command.joint_names) == 0:
                kick_finished = True
            else:
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                self.sim.step_sim()

        # wait till ball stops moving
        vx, vy, vz = self.sim.get_ball_velocity()
        ball_velocity = (vx ** 2 + vy ** 2 + vz ** 2) ** (1 / 3)
        while ball_velocity > 0.1 and self.sim.get_time() - start_time < timeout:
            self.sim.step_sim()
            vx, vy, vz = self.sim.get_ball_velocity()
            ball_velocity = (vx ** 2 + vy ** 2 + vz ** 2) ** (1 / 3)

        return self.sim.get_time() - start_time < timeout

    def get_robot_fell(self):
        pos, rpy = self.sim.get_robot_pose_rpy()
        return (abs(rpy[0]) > math.radians(45) or
                abs(rpy[1]) > math.radians(45) or
                pos[2] < self.reset_trunk_height / 2)

    def get_ball_position(self):
        return self.sim.get_ball_position()


if __name__ == '__main__':
    import argparse
    import json
    import matplotlib.pyplot as plt
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', default=os.getcwd(), nargs='?')
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.directory, 'params.json')):
        sys.exit('No params.json found in directory')

    with open(os.path.join(args.directory, 'params.json')) as f:
        params = json.load(f)

    evaluate_goals = [
        (0.2, 0, math.radians(0), 1),
        (0.2, 0, math.radians(45), 1),
        (0.2, 0, math.radians(-45), 1),
        (0.2, 0.1, math.radians(20), 1),
        (0.2, 0, math.radians(-90), 1),
    ]

    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink')

    kick_runner = KickRunner()
    for param_id, param_set in params.items():
        plt.title(f'Parameter set {param_id}')
        figure, axes = plt.subplots()
        axes.set_aspect(1)
        axes.set_xlim((-3, 3))
        axes.set_ylim((-0.5, 3))
        for i, goal in enumerate(evaluate_goals):
            kick_runner.set_kick_params(param_set)
            kick_runner.reset()
            kick_runner.set_goal(*goal)
            kick_runner.perform()
            robot_fell = kick_runner.get_robot_fell()
            ball_position = kick_runner.get_ball_position()
            ball_position_dx = -ball_position[1]
            ball_position_dy = ball_position[0]
            ball = plt.Circle((ball_position_dx, ball_position_dy), 0.1, color=colors[i])
            axes.add_patch(ball)
            xs = (-goal[1], -goal[1] - math.sin(goal[2]) * 5)
            ys = (goal[0], goal[0] + math.cos(goal[2]) * 5)
            plt.plot(xs, ys, color=colors[i])
        figure.savefig(os.path.join(args.directory, f'evaluation_{param_id}.png'), dpi=150)
        plt.clf()
        plt.cla()
