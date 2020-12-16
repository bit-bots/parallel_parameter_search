import math
import time

import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState


class AbstractMoveBaseOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, gui='false', sim_type='pybullet', foot_link_names=()):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot_name + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names, ros_active=True)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot_name)
        else:
            print(f'sim type {sim_type} not known')

        # start robot state publisher for tf
        self.state_publisher = roslaunch.core.Node('robot_state_publisher', 'robot_state_publisher',
                                                   'robot_state_publisher', namespace=self.namespace)
        self.state_publisher.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.state_publisher)
        # we also need tf of base footprint
        self.state_publisher = roslaunch.core.Node('humanoid_base_footprint', 'base_footprint',
                                                   'base_footprint_publisher', namespace=self.namespace)
        self.state_publisher.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.state_publisher)
        # start move base node
        self.map_node = roslaunch.core.Node('map_server', 'map_server', 'map_server', namespace=self.namespace,
                                            args="$(find bitbots_localization)/models/field2019.yaml")
        self.map_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.map_node)
        self.tf_map_node = roslaunch.core.Node('bitbots_move_base', 'tf_map_odom.py', 'tf_odom_to_map',
                                               namespace=self.namespace)
        self.tf_map_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.tf_map_node)
        load_yaml_to_param(self.namespace + '/move_base/global_costmap', 'bitbots_move_base',
                           '/config/costmap_common_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base/local_costmap', 'bitbots_move_base',
                           '/config/costmap_common_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/local_costmap_config.yaml',
                           self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/global_costmap_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/move_base_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/dwa_local_planner_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/global_planner_config.yaml', self.rospack)
        self.move_base_node = roslaunch.core.Node('move_base', 'move_base', 'move_base', namespace=self.namespace)
        self.move_base_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.move_base_node)
        # start walking with special parameters that have no limits
        # todo which walk parameter file should be used. maybe find better solution for this
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk',
                           '/config/walking_' + robot_name + '_robot_no_limits.yaml', self.rospack)
        self.walk_node = roslaunch.core.Node('bitbots_quintic_walk', 'WalkNode', 'walking', namespace=self.namespace)
        self.walk_node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock"),
                                     ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.walk_node)

        # let nodes start
        self.sim.run_simulation(100, 0.01)

        self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/move_base/DWAPlannerROS', timeout=60)

        self.number_of_iterations = 10
        self.time_limit = 10

        # needs to be specified by subclasses
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 0, 0]

        self.goals = [create_goal_msg(1, 0, 0),
                      create_goal_msg(-1, 0, 0),
                      create_goal_msg(0, 1, 0),
                      create_goal_msg(0, 0, math.pi / 2),
                      create_goal_msg(2, 3, math.pi / 2)]

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_params(trial)
        self.reset()

        cost = 0
        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for goal in self.goals:
                d += 1
                early_term, time_try = self.evaluate_direction(goal, trial)
                # scale cost to time_limit
                cost += time_try / self.time_limit
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 1 cost for each try left
                    return 1 * (self.number_of_iterations - iteration) * len(self.directions) + 1 * (
                            len(self.directions) - d) + cost
        return cost

    def suggest_params(self, trial):
        # todo
        raise NotImplementedError

    def is_goal_reached(self):
        # todo
        raise NotImplementedError

    def evaluate_nav_goal(self, goal: PoseStamped, trial: optuna.Trial):
        start_time = self.sim.get_time()
        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            # test timeout
            if passed_time > self.time_limit:
                self.reset()
                trial.set_user_attr('time_limit_reached',
                                    (goal.pose.position.x, goal.pose.position.y, goal.pose.orientation))
                return True, self.time_limit

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
                self.reset()
                # add extra information to trial
                # todo add only yaw instead of quaternion
                # todo include max vel parameter or something similar
                trial.set_user_attr('fallen',
                                    (goal.pose.position.x, goal.pose.position.y, goal.pose.orientation))
                return True, self.time_limit

            # todo add test if the robot did not move for more than x seconds, to find oszillations more easily

            # test if we are successful
            if self.is_goal_reached():
                # only need to reset the position of the robot, since we stopped correctly
                self.reset_position()
                return False, passed_time

            # give time to other algorithms to compute their responses
            # use wall time, as ros time is standing still
            time.sleep(0.01)
            self.sim.step_sim()

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def reset(self):
        # reset simulation
        # todo cancel goals, stop walking
        # todo get in walkready
        self.reset_position()

    def set_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = 0
        msg.angular.z = yaw
        self.cmd_vel_pub.publish(msg)


def create_goal_msg(x, y, yaw):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = 0
    (x, y, z, w) = tf.transformations.quaternion_from_euler(0, 0, yaw)
    msg.pose.orientation.x = x
    msg.pose.orientation.y = y
    msg.pose.orientation.z = z
    msg.pose.orientation.w = w

    return msg


class WolfgangMoveBaseOptimization(AbstractMoveBaseOptimization):
    def __init__(self, namespace, gui, sim_type='pybullet'):
        super(WolfgangMoveBaseOptimization, self).__init__(namespace=namespace, gui=gui, robot_name='wolfgang',
                                                           sim_type=sim_type)
        self.reset_height_offset = 0.005
