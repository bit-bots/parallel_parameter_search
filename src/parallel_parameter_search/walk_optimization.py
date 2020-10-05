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
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim


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
        self.time_limit = 20

        # needs to be specified by subclasses
        self.directions = None
        self.reset_height_offset = None

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        self.reset()

        cost = 0
        # standing as first test, is not in loop as it will only be done once
        early_term, cost_try = self.evaluate_direction(0, 0, 0, trial, 1, 0)
        cost += cost_try
        if early_term:
            # terminate early and give 100 cost for each try left
            return 100 * (self.number_of_iterations - 1) * len(self.directions) + 100 * len(self.directions) + cost

        for eval_try in range(1, self.number_of_iterations + 1):
            failed_directions_try = 0
            d = 0
            for direction in self.directions:
                d += 1
                self.reset_position()
                early_term, cost_try = self.evaluate_direction(*direction, trial, eval_try, self.time_limit)
                cost += cost_try
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 100 cost for each try left
                    return 100 * (self.number_of_iterations - eval_try) * len(self.directions) + 100 * (
                            len(self.directions) - d) + cost
            if failed_directions_try == len(self.directions):
                # terminate early if we didn't succeed at all
                # add future costs of missed trials
                return cost + (self.number_of_iterations - eval_try) * len(self.directions) * 100
        return cost

    def suggest_walk_params(self, trial):
        raise NotImplementedError

    def _suggest_walk_params(self, trial, trunk_height, foot_distance, foot_rise):
        engine_param_dict = {}

        def add(name, min_value, max_value):
            engine_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            engine_param_dict[name] = value
            trial.set_user_attr(name, value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 0.5, 3)
        add('foot_distance', foot_distance[0], foot_distance[1])
        add('trunk_height', trunk_height[0], trunk_height[1])
        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_x_offset', -0.03, 0.03)

        add('trunk_x_offset_p_coef_forward', -1, 1)
        add('trunk_x_offset_p_coef_turn', -1, 1)

        # add('first_step_swing_factor', 0.0, 2)
        # add('first_step_trunk_phase', -0.5, 0.5)
        fix('first_step_swing_factor', 1)
        fix('first_step_trunk_phase', -0.5)

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        fix('foot_overshoot_phase', 1)
        fix('foot_overshoot_ratio', 0.0)

        # add('trunk_y_offset', -0.03, 0.03)
        # add('foot_rise', 0.04, 0.08)
        # add('foot_apex_phase', 0.0, 1.0)
        fix('trunk_y_offset', 0)
        fix('foot_rise', foot_rise)
        fix('foot_apex_phase', 0.5)

        add('trunk_pitch', -1.0, 1.0)
        # add('trunk_pitch_p_coef_forward', -5, 5)
        # add('trunk_pitch_p_coef_turn', -5, 5)
        fix('trunk_pitch_p_coef_forward', 0)
        fix('trunk_pitch_p_coef_turn', 0)

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        fix('foot_z_pause', 0)
        fix('foot_put_down_phase', 1)
        fix('trunk_pause', 0)

        node_param_dict = {}
        # walk engine should update at same speed as simulation
        node_param_dict["engine_freq"] = 1 / self.sim.get_timestep()
        # don't use loop closure when optimizing parameter
        node_param_dict["pressure_phase_reset_active"] = False
        node_param_dict["effort_phase_reset_active"] = False
        node_param_dict["phase_rest_active"] = False
        # make sure that steps are not limited
        node_param_dict["imu_active"] = False
        node_param_dict["max_step_x"] = 100.0
        node_param_dict["max_step_y"] = 100.0
        node_param_dict["max_step_xy"] = 100.0
        node_param_dict["max_step_z"] = 100.0
        node_param_dict["max_step_angular"] = 100.0

        if self.walk_as_node:
            self.set_params(engine_param_dict)
            self.set_params(node_param_dict)
        else:
            self.current_params = engine_param_dict
            self.walk.set_engine_dyn_reconf(engine_param_dict)
            self.walk.set_node_dyn_reconf(node_param_dict)

    def evaluate_direction(self, x, y, yaw, trial: optuna.Trial, iteration, time_limit):
        start_time = self.sim.get_time()
        self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
        print(F'cmd: {x * iteration} {y * iteration} {yaw * iteration}')

        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            if passed_time > time_limit:
                # reached time limit, stop robot
                self.set_cmd_vel(0, 0, 0)

            if passed_time > time_limit + 5:
                # robot should have stopped now, evaluate the fitness
                return self.compute_cost(x * iteration, y * iteration, yaw * iteration)

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45):
                # add extra information to trial
                trial.set_user_attr('early_termination_at', (
                    x * iteration, y * iteration, yaw * iteration))
                early_term, cost = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                return True, cost

            try:
                if self.walk_as_node:
                    # give time to other algorithms to compute their responses
                    # use wall time, as ros time is standing still
                    time.sleep(0.01)
                else:
                    current_time = self.sim.get_time()
                    joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                                   self.sim.get_imu_msg(),
                                                   self.sim.get_joint_state_msg())
                    self.sim.set_joints(joint_command)
                    self.last_time = current_time
                self.sim.step_sim()
            except:
                pass

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def run_walking(self, duration):
        start_time = self.sim.get_time()
        while not rospy.is_shutdown() and (duration is None or self.sim.get_time() - start_time < duration):
            self.sim.step_sim()
            current_time = self.sim.get_time()
            joint_command = self.walk.step(current_time - self.last_time, self.current_speed, self.sim.get_imu_msg(),
                                           self.sim.get_joint_state_msg())
            # print(joint_command)
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
                                               self.sim.get_joint_state_msg())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                if not fake:
                    self.sim.step_sim()
                phase = self.walk.get_phase()
                # phase will advance by the simulation time times walk frequency
                next_phase = phase + self.sim.get_timestep() * self.walk.get_freq()
                # do double step to always have torso at same position
                if (phase >= 0.5 and next_phase >= 1.0):  # or (phase < 0.5 and next_phase >= 0.5):
                    # next time the walking step will change
                    break
                if self.sim.get_time() - start_time > 5:
                    # if walking does not perform step correctly due to impossible IK problems, do a timeout
                    break

    def compute_cost(self, x, y, yaw):
        """
        x,y,yaw have to be either 1 for being goo, -1 for being bad or 0 for making no difference
        """
        # factor to increase the weight of the yaw since it is a different unit then x and y
        yaw_factor = 5
        # todo better formular
        pos, rpy = self.sim.get_robot_pose_rpy()

        # 2D pose
        current_pose = [pos[0], pos[1], rpy[2]]
        correct_pose = [x * self.time_limit,
                        y * self.time_limit,
                        (yaw * self.time_limit) % (2 * math.pi)]  # todo we dont take multiple rotations into account
        cost = abs(current_pose[0] - correct_pose[0]) \
               + abs(current_pose[1] - correct_pose[1]) \
               + abs(current_pose[2] - correct_pose[
            2]) * yaw_factor  # todo take closest distance in circle through 0 into account
        # method doesn't work for going forward and turning at the same times
        if yaw != 0:  # and (x != 0 or y != 0):
            # just give 0 cost for surviving
            cost = 0
        # test if robot moved at all for simple case
        early_term = False
        if yaw == 0 and ((x != 0 and abs(current_pose[0]) < 0.5 * abs(correct_pose[0])) or (
                y != 0 and abs(current_pose[1]) < 0.5 * abs(correct_pose[1]))):
            early_term = True
        return early_term, cost

    def reset_position(self):
        height = self.current_params['trunk_height'] + self.reset_height_offset
        pitch = self.current_params['trunk_pitch']
        (x, y, z, w) = tf.transformations.quaternion_from_euler(0, pitch, 0)

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
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.run_walking(duration=2)

    def set_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = yaw
        if self.walk_as_node:
            self.cmd_vel_pub.publish(msg)
        else:
            self.current_speed = msg


class WolfgangWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(WolfgangWalkOptimization, self).__init__(namespace, 'wolfgang', walk_as_node)
        self.reset_height_offset = 0.005
        self.directions = [[0.1, 0, 0],
                           [-0.1, 0, 0],
                           [0, 0.05, 0],
                           [0, -0.05, 0],
                           [0, 0, 0.5],
                           [0, 0, -0.5],
                           [0.1, 0, 0.5],
                           [0, 0.05, -0.5],
                           [0.1, 0.05, 0.5],
                           [-0.1, -0.05, -0.5]
                           ]
        if sim_type == 'pybullet':
            self.sim = PybulletSim(self.namespace, gui)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.38, 0.45), (0.1, 0.3), 0.05)


class DarwinWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(DarwinWalkOptimization, self).__init__(namespace, 'darwin', walk_as_node)
        self.reset_height_offset = 0.09
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path('darwin_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=['MP_ANKLE2_L', 'MP_ANKLE2_R'])
            self.sim.set_joints_dict({"LShoulderRoll": 0.5, "LElbow": -1.15, "RShoulderRoll": -0.5, "RElbow": 1.15})
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.20, 0.24), (0.08, 0.15), 0.02)


class OP3WalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(OP3WalkOptimization, self).__init__(namespace, 'op3', walk_as_node)
        self.reset_height_offset = 0.01
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path('op3_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=['r_ank_roll_link', 'l_ank_roll_link'])
            self.sim.set_joints_dict({"l_sho_roll": 1.20, "r_sho_roll": -1.20})
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.13, 0.24), (0.08, 0.15), 0.02)


class NaoWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(NaoWalkOptimization, self).__init__(namespace, 'nao', walk_as_node)
        self.reset_height_offset = 0.01
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path('nao_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=['l_ankle', 'r_ankle'])
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.27, 0.32), (0.1, 0.17), 0.03)


class ReemcWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(ReemcWalkOptimization, self).__init__(namespace, 'reemc', walk_as_node)
        self.reset_height_offset = 0.12
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path('reemc_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=['leg_left_6_link', 'leg_right_6_link'])
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.6, 0.8), (0.15, 0.30), 0.1)


class TalosWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(TalosWalkOptimization, self).__init__(namespace, 'talos', walk_as_node)
        self.reset_height_offset = 0.12
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path('talos_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=['leg_left_6_link', 'leg_right_6_link'])
            self.sim.set_joints_dict({"arm_left_4_joint": -1.57, "arm_right_4_joint": -1.57})
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.8, 1.2), (0.15, 0.4), 0.1)


def load_robot_param(namespace, rospack, name):
    rospy.set_param(namespace + '/robot_type_name', name)
    set_param_to_file(namespace + "/robot_description", name + '_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml', rospack)
