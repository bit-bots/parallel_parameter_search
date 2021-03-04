import math
import time

import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf
import transforms3d
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from nav_msgs.msg import Odometry

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState


class AbstractMoveBaseOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, gui='false', sim_type='pybullet', foot_link_names=()):
        #todo add option to just run in visualization??? no fall detection put much faster
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot_name + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names, ros_active=True)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot_name, ros_active=True)
        else:
            print(f'sim type {sim_type} not known')

        # start robot state publisher for tf
        self.state_publisher = roslaunch.core.Node('robot_state_publisher', 'robot_state_publisher',
                                                   'robot_state_publisher', namespace=self.namespace)
        self.state_publisher.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.state_publisher)
        # we also need tf of base footprint
        self.foot_print = roslaunch.core.Node('humanoid_base_footprint', 'base_footprint',
                                              'base_footprint_publisher', namespace=self.namespace)
        self.foot_print.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.foot_print)
        # we need odom fuser to provide odom tf
        self.odom = roslaunch.core.Node('bitbots_odometry', 'odometry_fuser',
                                        'odometry_fuser', namespace=self.namespace)
        self.odom.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static"),
                                ("motion_odometry", "true_odom")]
        self.launch.launch(self.odom)
        # start map server
        self.map_node = roslaunch.core.Node('map_server', 'map_server', 'map_server', namespace=self.namespace,
                                            args="$(find bitbots_localization)/config/fields/webots/map_server.yaml")
        self.map_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.map_node)
        # fake localization
        self.tf_map_node = roslaunch.core.Node('bitbots_move_base', 'tf_map_odom.py', 'tf_odom_to_map',
                                               namespace=self.namespace)
        self.tf_map_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.tf_map_node)
        # start move base node
        load_yaml_to_param(self.namespace + '/move_base/global_costmap', 'bitbots_move_base',
                           '/config/costmap_common_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base/local_costmap', 'bitbots_move_base',
                           '/config/costmap_common_config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/local_costmap_config.yaml',
                           self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/global_costmap_config.yaml',
                           self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/move_base_config.yaml',
                           self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/dwa_local_planner_config.yaml',
                           self.rospack)
        load_yaml_to_param(self.namespace + '/move_base', 'bitbots_move_base', '/config/global_planner_config.yaml',
                           self.rospack)
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

        # Let nodes start, otherwise the dynreconf client will timeout
        while not rospy.is_shutdown():
            try:
                self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/move_base/DWAPlannerROS',
                                                                        timeout=0)
                break
            except:
                # wait some more till the service is up
                self.sim.run_simulation(0.5, 0.1)

        self.move_base_result = None
        self.goal_publisher = rospy.Publisher(self.namespace + "/move_base/goal", MoveBaseActionGoal, tcp_nodelay=True,
                                              queue_size=1)
        self.cancel_publisher = rospy.Publisher(self.namespace + "/move_base/cancel", GoalID, tcp_nodelay=True,
                                                queue_size=1)
        self.move_base_subscriber = rospy.Subscriber(self.namespace + "/move_base/result", MoveBaseActionResult,
                                                     self.result_cb,
                                                     queue_size=1)

        self.number_of_iterations = 1
        #todo maybe reset timelimit based on best time
        self.time_limit = 60

        # needs to be specified by subclasses
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 0, 0]

        self.goals = [create_goal_msg(1, 0, 0),
                      create_goal_msg(0, 1, 0),
                      create_goal_msg(0, 0, math.pi / 2),
                      create_goal_msg(-1, 0, 0),
                      create_goal_msg(2, 3, math.pi / 2)]

    def objective(self, trial: optuna.Trial):
        # get parameter to evaluate from optuna
        self.suggest_params(trial)
        self.reset()

        cost = 0
        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for goal in self.goals:
                d += 1
                early_term, time_try = self.evaluate_nav_goal(goal, trial)
                # scale cost to time_limit
                cost += time_try / self.time_limit
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 1 cost for each try left
                    return 1 * (self.number_of_iterations - iteration) * len(self.goals) + 1 * (
                            len(self.goals) - d) + cost
        return cost

    def suggest_params(self, trial: optuna.Trial):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            param_dict[name] = value
            trial.set_user_attr(name, value)

        add('max_vel_x', 0.1, 1)
        add('min_vel_x', -1, -0.05)
        add('max_vel_y', 0.08, 1)
        fix('min_vel_y', - trial.params["max_vel_y"])
        add('max_vel_trans', 0, trial.params["max_vel_x"] + trial.params["max_vel_y"])
        fix('min_vel_trans', 0)
        add('max_vel_theta', 0.7, 10)
        fix('min_vel_theta', 0)

        add('acc_lim_x', 1, 5)
        add('acc_lim_y', 1, 5)
        add('acc_lim_trans', 1, 5)
        add('acc_lim_theta', 4, 50)

        add('path_distance_bias', 0, 10)
        add('goal_distance_bias', 0, 10)
        add('occdist_scale', 0, 10)
        add('twirling_scale', 0, 10)

        self.set_params(param_dict, self.dynconf_client)

    def evaluate_nav_goal(self, goal: MoveBaseActionGoal, trial: optuna.Trial):
        self.goal_publisher.publish(goal)
        start_time = self.sim.get_time()
        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            # test timeout
            if passed_time > self.time_limit:
                self.reset()
                goal_quat = goal.goal.target_pose.pose.orientation
                goal_rpy = transforms3d.euler.quat2euler((goal_quat.w, goal_quat.x, goal_quat.y, goal_quat.z))
                trial.set_user_attr('time_limit_reached',
                                    (goal.goal.target_pose.pose.position.x, goal.goal.target_pose.pose.position.y,
                                     goal_rpy[2]))
                return True, self.time_limit

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
                self.reset()
                # add extra information to trial
                # todo include max vel parameter or something similar
                goal_quat = goal.goal.target_pose.pose.orientation
                goal_rpy = transforms3d.euler.quat2euler((goal_quat.w, goal_quat.x, goal_quat.y, goal_quat.z))
                trial.set_user_attr('fallen',
                                    (goal.goal.target_pose.pose.position.x, goal.goal.target_pose.pose.position.y,
                                     goal_rpy[2]))
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

    def result_cb(self, msg: MoveBaseActionResult):
        self.move_base_result = msg.status.status

    def is_goal_reached(self):
        # todo maybe check if it actually reached? but normally due to true_odom it should be fine
        if self.move_base_result == 3:
            self.move_base_result = None
            return True
        return False

    def reset_position(self):
        self.sim.set_ball_position(5, 0)
        self.sim.run_simulation(1, 0.001)
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))
        self.sim.run_simulation(1, 0.001)


    def reset(self):
        # cancel goals and wait till robot stopped in walkready
        self.cancel_publisher.publish(GoalID())
        self.sim.run_simulation(1, 0.001)
        # reset simulation
        self.reset_position()

    def set_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = 0
        msg.angular.z = yaw
        self.cmd_vel_pub.publish(msg)


def create_goal_msg(x, y, yaw):
    time = rospy.Time.now()
    msg = MoveBaseActionGoal()
    msg.header.stamp = time
    msg.header.frame_id = "base_foot_print"
    msg.goal.target_pose.header.stamp = time
    msg.goal.target_pose.header.frame_id = "map"
    msg.goal.target_pose.pose.position.x = x
    msg.goal.target_pose.pose.position.y = y
    msg.goal.target_pose.pose.position.z = 0
    (x, y, z, w) = tf.transformations.quaternion_from_euler(0, 0, yaw)
    msg.goal.target_pose.pose.orientation.x = x
    msg.goal.target_pose.pose.orientation.y = y
    msg.goal.target_pose.pose.orientation.z = z
    msg.goal.target_pose.pose.orientation.w = w

    return msg


class WolfgangMoveBaseOptimization(AbstractMoveBaseOptimization):
    def __init__(self, namespace, gui, sim_type='pybullet'):
        super(WolfgangMoveBaseOptimization, self).__init__(namespace=namespace, gui=gui, robot_name='wolfgang',
                                                           sim_type=sim_type)
        self.reset_height_offset = 0.00
        self.trunk_height = 0.4
        self.trunk_pitch = 0.14
