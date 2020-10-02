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

from parallel_parameter_search.abstract_ros_optimization import set_param_to_file, load_yaml_to_param, \
    AbstractRosOptimization
from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractWalkOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, walk_as_node):
        super().__init__(namespace)
        rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, rospack, robot_name)

        # load walk params
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk',
                           '/config/walking_' + robot_name + '_optimization.yaml',
                           rospack)

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
                               '/config/kinematics.yaml', rospack)
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
        # starting with hardest first, to get faster early termination
        # todo maybe start with half that speed but make more trials
        # todo eine cleverere reinfolge für die verschiedenen fälle, auch abhängig davon welche geschafft oder nicht geschafft wurden
        # todo add scenarios where the speed command changes multiple times while the robot is already walking
        # todo take into account that falling to a specific site is propably only depending on a subset of the parameters

        for eval_try in range(1, self.number_of_iterations + 1):
            failed_directions_try = 0
            d = 0
            for direction in self.directions:
                d += 1
                self.reset_position()
                early_term, cost_try = self.evaluate_direction(*direction, trial, eval_try)
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

    def evaluate_direction(self, x, y, yaw, trial: optuna.Trial, iteration):
        start_time = self.sim.get_time()
        self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
        print(F'cmd: {x * iteration} {y * iteration} {yaw * iteration}')

        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            if passed_time > self.time_limit:
                # reached time limit, stop robot
                self.set_cmd_vel(0, 0, 0)

            if passed_time > self.time_limit + 5:
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
                    #print(current_time - self.last_time)
                    #print(self.current_speed)
                    #print(self.sim.get_imu_msg())
                    #print(self.sim.get_joint_state_msg())
                    joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                                   self.sim.get_imu_msg(),
                                                   self.sim.get_joint_state_msg())
                    #print(joint_command)
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
            self.sim.set_joints(joint_command)
            self.last_time = current_time

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
            2]) * yaw_factor  # todo take closest distance in cricle through 0 into account
        # method doesn't work for going forward and turning at the same times
        # todo better computation of correct end pose, maybe use foot position
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
        # self.sim.reset()
        # let the robot do a few steps in the air to get correct walkready position
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.set_cmd_vel(0.1, 0, 0)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.run_walking(duration=2)
        self.set_cmd_vel(0, 0, 0)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.run_walking(duration=2)
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
    def __init__(self, namespace, gui, walk_as_node):
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
        self.sim = PybulletSim(self.namespace, gui)

    def suggest_walk_params(self, trial):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        add('double_support_ratio', 0.0, 0.45)
        add('freq', 1.5, 3)
        add('foot_distance', 0.1, 0.3)
        add('trunk_height', 0.38, 0.45)
        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_x_offset', -0.03, 0.03)

        add('trunk_x_offset_p_coef_forward', -1, 1)
        add('trunk_x_offset_p_coef_turn', -1, 1)

        # add('first_step_swing_factor', 0.0, 2)
        # add('first_step_trunk_phase', -0.5, 0.5)
        param_dict['first_step_swing_factor'] = 1
        param_dict['first_step_trunk_phase'] = -0.5

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        param_dict['foot_overshoot_phase'] = 1
        param_dict['foot_overshoot_ratio'] = 0.0

        # add('trunk_y_offset', -0.03, 0.03)
        # add('foot_rise', 0.04, 0.08)
        # add('foot_apex_phase', 0.0, 1.0)
        param_dict['trunk_y_offset'] = 0
        param_dict['foot_rise'] = 0.1
        param_dict['foot_apex_phase'] = 0.5
        # todo put this as addition arguments to trial

        # add('trunk_pitch', -1.0, 1.0)
        # add('trunk_pitch_p_coef_forward', -5, 5)
        # add('trunk_pitch_p_coef_turn', -5, 5)
        param_dict['trunk_pitch'] = 0
        param_dict['trunk_pitch_p_coef_forward'] = 0
        param_dict['trunk_pitch_p_coef_turn'] = 0

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        param_dict['foot_z_pause'] = 0
        param_dict['foot_put_down_phase'] = 1
        param_dict['trunk_pause'] = 0

        # todo 'trunk' nochmal ander nennen? body?

        # todo also find PID values, maybe in a second step after finding walking params
        # todo de/activate phase reset while searching params? yes

        if self.walk_as_node:
            self.set_params(param_dict)
        else:
            self.current_params = param_dict
            self.walk.set_engine_dyn_reconf(param_dict)


class DarwinWalkOptimization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node):
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
        self.sim = WebotsSim(self.namespace, gui)

    def suggest_walk_params(self, trial):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 1.5, 3)
        add('foot_distance', 0.08, 0.17)
        add('trunk_height', 0.18, 0.24)
        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_x_offset', -0.03, 0.03)

        add('trunk_x_offset_p_coef_forward', -1, 1)
        add('trunk_x_offset_p_coef_turn', -1, 1)

        #add('first_step_swing_factor', 0.0, 2)
        #add('first_step_trunk_phase', -0.5, 0.5)
        param_dict['first_step_swing_factor'] = 1
        param_dict['first_step_trunk_phase'] = -0.5

        #add('foot_overshoot_phase', 0.0, 1.0)
        #add('foot_overshoot_ratio', 0.0, 1.0)
        param_dict['foot_overshoot_phase'] = 1
        param_dict['foot_overshoot_ratio'] = 0.0

        # add('trunk_y_offset', -0.03, 0.03)
        # add('foot_rise', 0.04, 0.08)
        # add('foot_apex_phase', 0.0, 1.0)
        param_dict['trunk_y_offset'] = 0
        param_dict['foot_rise'] = 0.05
        param_dict['foot_apex_phase'] = 0.5
        # todo put this as addition arguments to trial

        # add('trunk_pitch', -1.0, 1.0)
        # add('trunk_pitch_p_coef_forward', -5, 5)
        # add('trunk_pitch_p_coef_turn', -5, 5)
        param_dict['trunk_pitch'] = 0
        param_dict['trunk_pitch_p_coef_forward'] = 0
        param_dict['trunk_pitch_p_coef_turn'] = 0

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        param_dict['foot_z_pause'] = 0
        param_dict['foot_put_down_phase'] = 1
        param_dict['trunk_pause'] = 0

        if self.walk_as_node:
            self.set_params(param_dict)
        else:
            self.current_params = param_dict
            self.walk.set_engine_dyn_reconf(param_dict)


def load_robot_param(namespace, rospack, name):
    rospy.set_param(namespace + '/robot_type_name', name)
    set_param_to_file(namespace + "/robot_description", name + '_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml', rospack)
