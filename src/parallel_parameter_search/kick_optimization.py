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
    def __init__(self, namespace, gui, robot_name, sim_type, foot_link_names=()):
        super().__init__(namespace)
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
        self.directions = ((0.2, 0.09, 0, 1),  # straight kick, left
                           (0.2, -0.09, 0, 1),  # straight kick, right
                           (0.2, 0, 0, 1),  # straight kick in the middle
                           (0.2, 0.09, math.radians(90), 1),  # side kick, to left side
                           (0.2, -0.09, -math.radians(90), 1),  # side kick, to right side
                           )

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

        cost = 0
        for d, direction in enumerate(self.directions):
            # move ball away
            self.sim.place_ball(3, 0)
            self.reset()
            early_term, cost_try = self.evaluate_goal(*direction, trial)
            cost += cost_try
            # check if we failed in this direction and terminate this trial early
            if early_term:
                # terminate early and give 1 cost for each try left
                return cost + 1 * (len(self.directions) - d)
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
        add('kick_windup_distance', 0.1, 0.4, 0.01)
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

        self.kick.set_params(param_dict)

    def evaluate_goal(self, x, y, yaw, speed, trial: optuna.Trial):
        """
        Evaluate a single kick goal, i.e. one execution of a kick
        Returns whether early termination occurred (bool), the cost of the try (between 0 and 1)
        """
        goal_msg = self.get_kick_goal_msg(x, y, yaw, speed)
        self.kick.set_goal(goal_msg, self.sim.get_joint_state_msg())
        self.sim.place_ball(x, y)
        self.sim.step_sim()
        self.sim.step_sim()
        print(f'goal: {x} {y} {yaw} {speed}')
        start_time = self.sim.get_time()
        kick_finished = False
        self.last_time = self.sim.get_time()
        # wait till kick is finished or robot fell down
        while not kick_finished:
            # test if the robot has fallen down, then return maximal cost
            pos, rpy = self.sim.get_robot_pose_rpy()
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.reset_trunk_height / 2:
                # add extra information to trial
                trial.set_user_attr('early_termination_at', (x, y, yaw))
                return True, 1

            current_time = self.sim.get_time()
            joint_command = self.kick.step(current_time - self.last_time,
                                           self.sim.get_joint_state_msg())  # type: JointCommand
            if len(joint_command.joint_names) == 0:
                kick_finished = True
            else:
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                self.sim.step_sim()

        passed_time = self.sim.get_time() - start_time
        passed_timesteps = max(1, passed_time / self.sim.get_timestep())
        # kick is finished, wait if robot is falling down
        while self.sim.get_time() < start_time + passed_time + 2:
            self.sim.step_sim()
        # get ball speed and take max

        pos, rpy = self.sim.get_robot_pose_rpy()
        if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.reset_trunk_height / 2:
            # robot fell
            return False, 1

        cost = self.compute_cost(yaw)
        return False, cost / passed_timesteps

    def run_kick(self, duration):
        """Execute the whole kick, only used for debug"""
        start_time = self.sim.get_time()
        while not rospy.is_shutdown() and (duration is None or self.sim.get_time() - start_time < duration):
            self.sim.step_sim()
            current_time = self.sim.get_time()
            joint_command = self.kick.step(current_time - self.last_time, self.sim.get_joint_state_msg())
            self.sim.set_joints(joint_command)
            self.last_time = current_time

    def compute_cost(self, yaw):
        """
        Compute the cost after the kick. yaw is the kick goal direction
        Problem: longer kick time is currently rewarded because the ball is rolling longer...
        Current solution: normalize with timesteps
        """
        ball_x, ball_y = self.sim.get_ball_position()
        distance = math.sqrt(ball_x ** 2 + ball_y ** 2)
        direction = math.atan2(ball_y, ball_x)
        direction_error = abs(yaw - direction)
        # We want high distance and low direction error
        return min(1, direction_error / distance)

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
        walkready = {"LAnklePitch": -30, "LAnkleRoll": 0, "LHipPitch": 30, "LHipRoll": 0,
                     "LHipYaw": 0, "LKnee": 60, "RAnklePitch": 30, "RAnkleRoll": 0,
                     "RHipPitch": -30, "RHipRoll": 0, "RHipYaw": 0, "RKnee": -60,
                     "LShoulderPitch": 0, "LShoulderRoll": 0, "LElbow": 45, "RShoulderPitch": 0,
                     "RShoulderRoll": 0, "RElbow": -45, "HeadPan": 0, "HeadTilt": 0}
        msg = JointCommand()
        for name, pos in walkready.items():
            msg.joint_names.append(name)
            msg.positions.append(math.radians(pos))
        self.sim.set_joints(msg)

    def reset(self):
        # reset simulation
        # set the robot to walkready first

        self.sim.set_gravity(False)
        # move the robot in the air
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.set_to_walkready()
        for _ in range(20):
            self.sim.step_sim()
        self.sim.set_gravity(True)
        # set it back on the ground
        self.reset_position()
        for _ in range(20):
            self.sim.step_sim()

    def get_kick_goal_msg(self, x, y, yaw, speed):
        msg = KickGoal()
        msg.header.stamp = rospy.Time(self.sim.get_time())
        msg.header.frame_id = "base_footprint"
        msg.kick_direction = Quaternion(*quaternion_from_euler(0, 0, yaw))
        msg.kick_speed = speed
        msg.ball_position.x = x
        msg.ball_position.y = y
        return msg


class WolfgangKickEngineOptimization(AbstractKickOptimization):
    def __init__(self, namespace, gui, sim_type='pybullet'):
        super().__init__(namespace, gui, 'wolfgang', sim_type)
        # These are the start values from the kick, they come from the walking
        self.reset_trunk_height = 0.42
        self.reset_trunk_pitch = 0.26
