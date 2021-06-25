import sys
import math
import optuna
import rospkg
import rospy
from geometry_msgs.msg import Quaternion
from bitbots_msgs.msg import KickGoal, JointCommand
from tf.transformations import quaternion_from_euler

from bitbots_dynamic_kick import PyKick
from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractKickOptimization(AbstractRosOptimization):
    def __init__(self, namespace, gui, robot_name, sim_type, multi_objective, foot_link_names=(), kamikaze=False):
        super().__init__(namespace)
        self.multi_objective = multi_objective
        self.kamikaze = kamikaze
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        # load walk params
        load_yaml_to_param(self.namespace, 'bitbots_dynamic_kick',
                           '/config/kick_config.yaml', self.rospack)

        self.last_time = 0
        load_yaml_to_param("/robot_description_kinematics", robot_name + '_moveit_config',
                           '/config/kinematics.yaml', self.rospack)
        # create kick as python class to call it later
        self.kick = PyKick(self.namespace)

        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot_name + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot_name)
        else:
            sys.exit(f'sim type {sim_type} not known')

        # each direction consists of x, y (ball position), yaw (kick direction) and kick speed
        #self.directions = ((0.2, 0.09, 0, 1),  # straight kick, left
        #                   (0.2, -0.09, 0, 1),  # straight kick, right
        #                   (0.2, 0, 0, 1),  # straight kick in the middle
        #                   (0.2, 0.09, math.radians(90), 1),  # side kick, to left side
        #                   (0.2, -0.09, -math.radians(90), 1),  # side kick, to right side
        #                   )
        self.directions = ((0.2, 0, 0),)

        self.kick_speed = 0

        # needs to be specified by subclasses
        self.reset_trunk_height = 0
        self.reset_trunk_pitch = 0
        self.reset_rpy_offset = (0, 0, 0)

    def objective(self, trial):
        """
        Perform all kicks on a parameter set
        Returns the overall cost
        """
        # get parameter to evaluate from optuna
        self.suggest_kick_params(trial)

        if self.multi_objective:
            # fell?, time, velocity, directional error
            cost = [0, 0, 0, 0]
        else:
            cost = 0
        for d, direction in enumerate(self.directions):
            self.reset()
            fell_before_kick, fell, steps, ball_velocity, ball_direction, ball_pos = self.evaluate_goal(*direction, trial)
            direction_error = self.compute_yaw_error(direction)
            if self.multi_objective:
                cost_try = [0 if fell else 1, steps, ball_velocity, direction_error]
                cost = [cost[i] + cost_try[i] for i in range(len(cost))]
                if fell_before_kick:
                    # Do not continue evaluating this, instead give large costs
                    cost = [
                        cost[0] + len(self.directions) - d,  # 1 for each of the following
                        cost[1] + (10 / self.sim.get_timestep()) * (len(self.directions) - d),  # assume 10s for every evaluation
                        cost[2],  # 0 for velocity
                        cost[3] + math.pi / 2 * (len(self.directions) - d),  # pi/2 error for each remaining
                    ]
                    return cost
            else:
                if self.kamikaze:
                    cost_try = -ball_pos[0]
                else:
                    if fell:
                        # just take a rather large cost
                        cost_try = 10 / self.sim.get_timestep()
                    else:
                        # Minimize error and steps, maximize velocity
                        cost_try = direction_error * steps / ball_velocity
                cost += cost_try
                # check if we failed in this direction and terminate this trial early
                if not self.kamikaze and fell:
                    # terminate early and give large cost for each try left
                    return cost + (10 / self.sim.get_timestep()) * (len(self.directions) - d)
        return cost

    def suggest_kick_params(self, trial: optuna.Trial):
        param_dict = {}

        def add(name, min_value, max_value, step):
            param_dict[name] = trial.suggest_float(name, min_value, max_value, step=step)

        def fix(name, value):
            param_dict[name] = value
            trial.set_user_attr(name, value)

        fix('engine_rate', 1 / self.sim.get_timestep())

        add('foot_rise', 0.05, 0.15, 0.01)
        add('foot_distance', 0.15, 0.25, 0.01)
        add('kick_windup_distance', 0.1, 0.6, 0.01)
        add('trunk_height', 0.35, 0.45, 0.01)
        add('trunk_roll', math.radians(-30), math.radians(30), math.radians(0.1))
        add('trunk_pitch', math.radians(-30), math.radians(30), math.radians(0.1))
        add('trunk_yaw', math.radians(-45), math.radians(45), math.radians(0.1))

        add('move_trunk_time', 0.1, 1, step=0.01)
        add('raise_foot_time', 0.1, 1, step=0.01)
        add('move_to_ball_time', 0.1, 0.5, step=0.01)
        add('kick_time', 0.01, 0.2, step=0.01)
        add('move_back_time', 0.01, 0.3, step=0.01)
        add('lower_foot_time', 0.01, 0.2, step=0.01)
        add('move_trunk_back_time', 0.05, 0.4, step=0.01)

        fix('choose_foot_corridor_width', 0.4)

        fix('use_center_of_pressure', False)
        add('stabilizing_point_x', -0.1, 0.1, 0.01)
        add('stabilizing_point_y', -0.1, 0.1, 0.01)

        add('windup_hip', 0, 90, 1)
        add('windup_knee', 0, 120, 1)
        add('windup_ankle', -30, 90, 1)
        add('knee_time', 0, param_dict['kick_time'], 0.01)
        add('ankle_time', 0, param_dict['kick_time'], 0.01)

        self.kick.set_params(param_dict)

        self.kick_speed = trial.suggest_float('kick_speed', 0, 10, step=0.01)
        self.goal_x = trial.suggest_float('goal_x', 0.1, 0.3, step=0.01)
        self.ball_x = trial.suggest_float('ball_x', 0.1, 0.3, step=0.01)

    def evaluate_goal(self, _, y, yaw, trial: optuna.Trial):
        """
        Evaluate a single kick goal, i.e. one execution of a kick
        Returns robot fell down before kick?, robot fell down?, number of timesteps, maximum ball velocity, ball direction
        """
        max_ball_velocity = 0
        max_ball_velocity_vector = (0, 0)
        goal_msg = self.get_kick_goal_msg(self.goal_x, y, yaw)
        self.kick.set_goal(goal_msg, self.sim.get_joint_state_msg())
        self.sim.place_ball(self.ball_x, y)
        self.sim.step_sim()
        self.sim.step_sim()
        print(f'goal: {self.goal_x} {y} {yaw}')
        start_time = self.sim.get_time()
        kick_finished = False
        self.last_time = self.sim.get_time()
        # wait till kick is finished or robot fell down
        while not kick_finished:
            vx, vy, vz = self.sim.get_ball_velocity()
            abs_velocity = (vx ** 2 + vy ** 2 + vz ** 2) ** (1 / 3)
            if abs_velocity > max_ball_velocity:
                max_ball_velocity = abs_velocity
                max_ball_velocity_vector = (vx, vy)
            # test if the robot has fallen down, then return maximal cost
            pos, rpy = self.sim.get_robot_pose_rpy()
            passed_time = self.sim.get_time() - start_time
            # only terminate early if we did not kick yet
            if passed_time <= (trial.params['move_trunk_time'] + trial.params['raise_foot_time'] +
                               trial.params['move_to_ball_time'] + trial.params['kick_time']) and \
                    (abs(rpy[0]) > math.radians(45) or
                     abs(rpy[1]) > math.radians(45) or
                     pos[2] < self.reset_trunk_height / 2):
                # add extra information to trial
                passed_timesteps = max(1, passed_time / self.sim.get_timestep())
                trial.set_user_attr('early_termination_at', (self.goal_x, y, yaw))
                return True, True, passed_timesteps, max_ball_velocity, max_ball_velocity_vector, self.sim.get_ball_position()

            current_time = self.sim.get_time()
            joint_command = self.kick.step(current_time - self.last_time,
                                           self.sim.get_joint_state_msg())  # type: JointCommand
            if len(joint_command.joint_names) == 0:
                kick_finished = True
            else:
                self.sim.set_joints(joint_command)
                joint_torque_command = self.kick.get_torque_command()
                if len(joint_torque_command.joint_names) > 0:
                    self.sim.set_joints(joint_torque_command)
                self.last_time = current_time
                self.sim.step_sim()

        passed_time = self.sim.get_time() - start_time
        passed_timesteps = max(1, passed_time / self.sim.get_timestep())
        # kick is finished, wait until the ball is no longer moving
        vx, vy, vz = self.sim.get_ball_velocity()
        abs_velocity = (vx ** 2 + vy ** 2 + vz ** 2) ** (1 / 3)
        while abs_velocity > 0.05 and self.sim.get_time() - start_time < 20:
            self.sim.step_sim()
            vx, vy, vz = self.sim.get_ball_velocity()
            abs_velocity = (vx ** 2 + vy ** 2 + vz ** 2) ** (1 / 3)

        pos, rpy = self.sim.get_robot_pose_rpy()
        if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.reset_trunk_height / 2:
            # robot fell
            fall = True
        else:
            fall = False

        return False, fall, passed_timesteps, max_ball_velocity, max_ball_velocity_vector, self.sim.get_ball_position()

    def run_kick(self, duration):
        """Execute the whole kick, only used for debug"""
        start_time = self.sim.get_time()
        while not rospy.is_shutdown() and (duration is None or self.sim.get_time() - start_time < duration):
            self.sim.step_sim()
            current_time = self.sim.get_time()
            joint_command = self.kick.step(current_time - self.last_time, self.sim.get_joint_state_msg())
            self.sim.set_joints(joint_command)
            self.last_time = current_time

    def compute_yaw_error(self, kick_goal):
        """
        Compute the error in the yaw after the kick.
        :param kick_goal: The kick goal ([ball_x, ball_y, goal_yaw, speed])
        """
        ball_x, ball_y = self.sim.get_ball_position()
        direction = math.atan2(ball_y - kick_goal[1], ball_x - kick_goal[0])
        direction_error = abs(kick_goal[2] - direction)
        return direction_error

    def reset_position(self):
        """Set the robot on the ground"""
        height = self.reset_trunk_height
        pitch = self.reset_trunk_pitch
        (x, y, z, w) = quaternion_from_euler(self.reset_rpy_offset[0],
                                             self.reset_rpy_offset[1] + pitch,
                                             self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def set_to_walkready(self):
        """Set the robot to walkready position"""
        walkready = {
            "HeadPan": 0.0,
            "HeadTilt": 0.0,
            "LShoulderPitch": 34.341900058232895,
            "LShoulderRoll": -0.10187296337477583,
            "LElbow": 17.4089039999328,
            "RShoulderPitch": -34.52380590822163,
            "RShoulderRoll": 0.10154916169401929,
            "RElbow": -17.44692966568184,
            "LHipYaw": -0.9420455109052479,
            "LHipRoll": 4.156114615225981,
            "LHipPitch": 42.51810038124632,
            "LKnee": 76.46280204571194,
            "LAnklePitch": -31.726953021845897,
            "LAnkleRoll": 4.261305249797921,
            "RHipYaw": 1.3537998702115084,
            "RHipRoll": -5.959789935735837,
            "RHipPitch": -42.127449488675886,
            "RKnee": -75.79323597048764,
            "RAnklePitch": 31.484250845285846,
            "RAnkleRoll": -7.3988993750834995
        }
        msg = JointCommand()
        for name, pos in walkready.items():
            msg.joint_names.append(name)
            msg.positions.append(math.radians(pos))
            msg.velocities.append(-1)
            msg.accelerations.append(-1)
        self.sim.set_joints(msg)

    def reset(self):
        # reset simulation
        # set the robot to walkready first

        # move ball away
        self.sim.place_ball(3, 0)
        self.sim.set_gravity(False)
        self.sim.reset_robot_init()
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        for _ in range(20):
            self.sim.step_sim()
        # move the robot in the air
        self.set_to_walkready()
        for _ in range(20):
            self.sim.step_sim()
        self.sim.set_gravity(True)
        # set it back on the ground
        self.reset_position()
        for _ in range(20):
            self.sim.step_sim()

    def get_kick_goal_msg(self, x, y, yaw):
        msg = KickGoal()
        msg.header.stamp = rospy.Time(self.sim.get_time())
        msg.header.frame_id = "base_footprint"
        msg.kick_direction = Quaternion(*quaternion_from_euler(0, 0, yaw))
        msg.kick_speed = self.kick_speed
        msg.ball_position.x = x
        msg.ball_position.y = y
        return msg


class WolfgangKickEngineOptimization(AbstractKickOptimization):
    def __init__(self, namespace, gui, sim_type='pybullet', multi_objective=False, kamikaze=False):
        super().__init__(namespace, gui, 'wolfgang', sim_type, multi_objective, kamikaze=kamikaze)
        # These are the start values from the kick, they come from the walking
        self.reset_trunk_height = 0.42
        self.reset_trunk_pitch = 0.26
