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
from bitbots_msgs.msg import JointCommand
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from bitbots_localization.srv import ResetFilter

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
        self.odom.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.odom)
        # start map server
        self.map_node = roslaunch.core.Node('map_server', 'map_server', 'map_server', namespace=self.namespace,
                                            args="$(find bitbots_localization)/config/fields/webots/map_server.yaml")
        self.map_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.map_node)
        # imu filter
        self.imu_node = roslaunch.core.Node('imu_complementary_filter', 'complementary_filter_node', 'complementary_filter_gain_node',
                                            namespace=self.namespace)
        rospy.set_param(self.imu_node.name + "/do_bias_estimation", True)
        rospy.set_param(self.imu_node.name + "/bias_alpha", 0.05)
        rospy.set_param(self.imu_node.name + "/da_adaptive_gain", False)
        rospy.set_param(self.imu_node.name + "/gain_acc", 0.04)
        self.imu_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.imu_node)
        # localization
        load_yaml_to_param(self.namespace + '/bitbots_localization', 'bitbots_localization', '/config/config.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/bitbots_localization', 'bitbots_localization', '/config/fields/webots/config.yaml', self.rospack)
        rospy.set_param(self.namespace + '/bitbots_localization/fieldname', 'webots')
        self.localization_node = roslaunch.core.Node('bitbots_localization', 'localization', 'bitbots_localization',
                                                     namespace=self.namespace)
        self.localization_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.localization_node)
        # localization handler
        self.localization_handler_node = roslaunch.core.Node('bitbots_localization', 'localization_handler.py',
                                                             'bitbots_localization_handler', namespace=self.namespace)
        self.localization_handler_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.localization_handler_node)
        # map server
        self.map_server_node = roslaunch.core.Node('map_server', 'map_server', 'field_map_server',
                                                   namespace=self.namespace, args='$(find bitbots_localization)/config/fields/webots/map_server.yaml')
        self.map_server_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static"),
                                           ("map", "field/map"), ("map_metadata", "field/map_metadata")]
        self.launch.launch(self.map_server_node)
        # start vision
        load_yaml_to_param(self.namespace + '/bitbots_vision', 'bitbots_vision', '/config/visionparams.yaml', self.rospack)
        load_yaml_to_param(self.namespace + '/bitbots_vision', 'bitbots_vision', '/config/simparams.yaml', self.rospack)
        rospy.set_param(self.namespace + '/bitbots_vision/neural_network_type', 'dummy')
        self.vision_node = roslaunch.core.Node('bitbots_vision', 'vision.py', 'bitbots_vision', namespace=self.namespace)
        self.vision_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.launch.launch(self.vision_node)
        # start transformer
        load_yaml_to_param(self.namespace + '/humanoid_league_transform', 'humanoid_league_transform',
                           '/config/transformer.yaml', self.rospack)
        self.transformer_node = roslaunch.core.Node('humanoid_league_transform', 'transformer.py',
                                                    'humanoid_league_transform', namespace=self.namespace)
        self.transformer_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.transformer_node.launch_prefix = f"taskset -c {self.namespace.split('_')[-1]}"
        self.launch.launch(self.transformer_node)
        # start head behavior
        load_yaml_to_param(self.namespace, 'bitbots_head_behavior', '/config/head_config.yaml', self.rospack)
        rospy.set_param(self.namespace + '/behavior/head/defaults/head_mode', 3)  # field features
        self.head_behavior_node = roslaunch.core.Node('bitbots_head_behavior', 'head_node.py', 'head_behavior',
                                                      namespace=self.namespace)
        self.head_behavior_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static"),
                                              ("head_motor_goals", "DynamixelController/command")]
        self.launch.launch(self.head_behavior_node)
        # bio ik service
        self.bio_ik_service_node = roslaunch.core.Node('bio_ik_service', 'bio_ik_service', 'bio_ik_service',
                                                       namespace=self.namespace)
        self.bio_ik_service_node.remap_args = [("/clock", "clock"), ("/tf", "tf"), ("/tf_static", "tf_static")]
        self.bio_ik_service_node.output = 'screen'
        self.launch.launch(self.bio_ik_service_node)
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
                                     ("/tf", "tf"), ("/tf_static", "tf_static"),
                                     ("walk_engine_odometry", "motion_odometry")]
        self.launch.launch(self.walk_node)

        # Let nodes start, otherwise the dynreconf client will timeout
        while not rospy.is_shutdown():
            try:
                self.move_base_dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/move_base/DWAPlannerROS',
                                                                                  timeout=0)
                break
            except:
                # wait some more till the service is up
                self.sim.run_simulation(0.5, 0.1)
        while not rospy.is_shutdown():
            try:
                self.localization_dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/bitbots_localization',
                                                                                     timeout=0)
                break
            except:
                # wait some more till the service is up
                self.sim.run_simulation(0.5, 0.1)
        print('Waiting for localization reset service... ', end='')
        rospy.wait_for_service(self.namespace + '/reset_localization')
        self.localization_reset_service = rospy.ServiceProxy(self.namespace + '/reset_localization', ResetFilter)
        print('done')

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

        self.start_point = (-2, 3, math.radians(-90))  # typical start position at the field edge
        self.goals = [create_goal_msg(-2, 2, math.radians(-90)),  # one meter forward
                      create_goal_msg(-1, 3, math.radians(-90)),  # one meter to the left
                      create_goal_msg(-2, 3, math.radians(180)),  # turn 90 degrees counterclockwise
                      create_goal_msg(0, 0, 0),  # go to the center point
                      create_goal_msg(-3, -2, 0),
                      create_goal_msg(2, 2, 0)]

    def objective(self, trial: optuna.Trial):
        # get parameter to evaluate from optuna
        self.suggest_params(trial)

        cost = 0
        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for goal in self.goals:
                self.reset()
                d += 1
                early_term, time_try = self.evaluate_nav_goal(goal, trial)
                pos, rot = self.sim.get_robot_pose_rpy()
                x = pos[0]
                y = pos[1]
                yaw = rot[2]
                # velocity^(-1) (seconds per meter)
                distance = math.sqrt((x - self.start_point[0])**2 + (y - self.start_point[1])**2)
                if distance > 0.5:
                    time_cost = (time_try / distance)**2
                else:
                    time_cost = 0
                print('time cost', time_cost)
                # weighted mean squared error, yaw is split in continuous sin and cos components
                goal_pose = goal.goal.target_pose.pose
                goal_rpy = transforms3d.euler.quat2euler([goal_pose.orientation.w, goal_pose.orientation.x,
                                                          goal_pose.orientation.y, goal_pose.orientation.z])
                yaw_error = (math.sin(goal_rpy[2]) - math.sin(yaw))**2 + (math.cos(goal_rpy[2]) - math.cos(yaw))**2
                # normalize pose error
                pose_cost = ((goal_pose.position.x - x)**2 + (goal_pose.position.y - y)**2 + yaw_error)
                print('pose cost', pose_cost)
                cost += 0.01 * time_cost + 50 * pose_cost

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

        #add('path_distance_bias', 0, 10)
        #add('goal_distance_bias', 0, 10)
        #add('occdist_scale', 0, 10)
        #add('twirling_scale', 0, 10)

        add('xy_goal_tolerance', 0, 0.5)
        add('yaw_goal_tolerance', 0, 0.5)

        print('move base suggesting', param_dict)
        self.set_params(param_dict, self.move_base_dynconf_client)

        param_dict = {}

        add('drift_distance_to_direction', 0.5, 6)
        add('drift_distance_to_distance', 0, 2)
        add('drift_roation_to_distance', 0, 2)  # sic
        add('drift_rotation_to_rotation', 0, 10)
        fix('max_rotation', trial.params['max_vel_theta'])
        fix('max_translation', trial.params['max_vel_x'])
        add('line_element_confidence', 0.0001, 0.05)
        add('diffusion_multiplicator', 0, 0.1)
        add('diffusion_t_std_dev', 0.2, 4)

        print('localization suggesting', param_dict)
        self.set_params(param_dict, self.localization_dynconf_client)

    def evaluate_nav_goal(self, goal: MoveBaseActionGoal, trial: optuna.Trial):
        self.goal_publisher.publish(goal)
        start_time = self.sim.get_time()
        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            # test timeout
            if passed_time > self.time_limit:
                goal_quat = goal.goal.target_pose.pose.orientation
                goal_rpy = transforms3d.euler.quat2euler((goal_quat.w, goal_quat.x, goal_quat.y, goal_quat.z))
                trial.set_user_attr('time_limit_reached',
                                    (goal.goal.target_pose.pose.position.x, goal.goal.target_pose.pose.position.y,
                                     goal_rpy[2]))
                return False, self.time_limit

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
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
            msg.velocities.append(-1)
            msg.accelerations.append(-1)
        self.sim.set_joints(msg)

    def reset_position(self):
        self.sim.set_ball_position(5, 0)
        self.sim.run_simulation(1, 0.001)
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2] - math.radians(90))

        self.sim.reset_robot_pose((-2, 3, height), (x, y, z, w))
        self.set_to_walkready()
        self.localization_reset_service(0, None, None)
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
