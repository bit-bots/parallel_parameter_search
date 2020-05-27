import math
import time
import sys

import optuna

import rospy
import roslaunch
import rospkg
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import dynamic_reconfigure.client

from wolfgang_pybullet_sim.simulation import Simulation
from wolfgang_pybullet_sim.ros_interface import ROSInterface

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization, load_yaml_to_param, \
    set_param_to_file


class PybulletOptimization(AbstractRosOptimization):

    def __init__(self, namespace, gui):
        super(PybulletOptimization, self).__init__(namespace)
        self.gui = gui
        # todo maybe use simulation + interface class directly and just provide a paramter to the interface class for the namespace
        if self.gui:
            self.sim_node = roslaunch.core.Node('wolfgang_pybullet_sim', 'simulation_with_gui.py', 'sim',
                                                namespace=self.namespace)
        else:
            self.sim_node = roslaunch.core.Node('wolfgang_pybullet_sim', 'simulation_headless.py', 'sim',
                                                namespace=self.namespace)
        self.launch.launch(self.sim_node)

        # wait till the simulation was started, visible by clock being available
        clock_topic = '/' + self.namespace + '/clock'
        while not rospy.is_shutdown():
            topic_names = [x[0] for x in rospy.get_published_topics(self.namespace)]
            if clock_topic in topic_names:
                break
            pass

        self.reset_pub = rospy.Publisher(self.namespace + '/reset', Bool, queue_size=1)

    def reset(self):
        self.reset_pub.publish(Bool())


class WalkPybulletOptimization(PybulletOptimization):
    def __init__(self, namespace, gui):
        super(WalkPybulletOptimization, self).__init__(namespace, gui)
        rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_wolfgang_param(self.namespace, rospack)

        # load walk params
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk', '/config/walking_wolfgang_robot.yaml', rospack)

        self.walk_node = roslaunch.core.Node('bitbots_quintic_walk', 'WalkNode', 'walking',
                                             namespace=self.namespace)
        self.walk_node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock")]
        self.launch.launch(self.walk_node)
        self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/engine', timeout=60)

        self.cmd_vel_pub = rospy.Publisher(self.namespace + '/cmd_vel', Twist, queue_size=10)

        self.robot_pose = None
        self.true_odom_sub = rospy.Subscriber(self.namespace + '/true_odom', Odometry, self.odom_cb, queue_size=1,
                                              tcp_nodelay=True)

        self.number_of_tries = 2
        self.time_limit = 20

        # wait for the simulation interface to provide data
        while not self.robot_pose and not rospy.is_shutdown():
            self.launch.spin_once()

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)

        fitness = 0
        evaluation_time = 0
        directions = [[0.1, 0, 0],
                      [0, 0.1, 0]]

        for eval_try in range(0, self.number_of_tries):
            for direction in directions:
                self.reset()
                fitness_try, time_try = self.evaluate_direction(*direction)
                # see if we want to terminate early due to unpromising trial
                if fitness_try == -1:
                    return -1
                fitness += fitness_try
                evaluation_time += evaluation_time

        # return mean fitness for one try
        return fitness / (self.number_of_tries * len(directions))

    def suggest_walk_params(self, trial):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        add('freq', 0.2, 3)
        add('double_support_ratio', 0.0, 0.5)
        add('foot_rise', 0.07, 0.1)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_height', 0.33, 0.48)
        add('trunk_pitch', -0.2, 0.4)
        add('trunk_phase', -0.5, 0.5)
        add('foot_overshoot_ratio', 0.0, 0.5)
        add('foot_overshoot_phase', 0.7, 1.0)
        add('foot_apex_phase', 0.3, 0.7)
        add('trunk_x_offset', -0.05, 0.05)

        self.dynconf_client.update_configuration(param_dict)

    def evaluate_direction(self, x, y, yaw):
        fitness_try = 0
        start_time = rospy.get_time()
        # todo roboter muss noch sicher in start position gestellt werden
        self.send_cmd_vel(x, y, yaw)

        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = rospy.get_time() - start_time
            if passed_time > self.time_limit:
                # reached time limit, evaluate the fitness
                fitness_try = self.measure_fitness(x, y, yaw)

            # todo
            """    
            # test if robot moved at least a bit        
            if not self.robot_has_moved and current_time > 20 :
                # has not moved 
                rospy.loginfo("robot didn't move")
                return True
            """
            # todo does not work
            # test if the robot has fallen down
            if self.robot_pose.position.z < 0.3:
                return -1, rospy.get_time() - start_time

            try:
                rospy.sleep(0.01)
            except:
                pass

        self.send_cmd_vel(0.0, 0, 0)
        return fitness_try, rospy.get_time() - start_time

    def measure_fitness(self, x, y, yaw):
        """
        x,y,yaw have to be either 1 for being goo, -1 for being bad or 0 for making no difference
        """
        # factor to increase the weight of the yaw since it is a different unit then x and y
        yaw_factor = 5

        return math.sqrt(max(x * self.robot_pose.position.x ** 2 +
                             y * self.robot_pose.position.y ** 2 +
                             yaw * self.robot_pose.orientation.z ** 2 * yaw_factor, 0.0))

    def odom_cb(self, msg: Odometry):
        self.robot_pose = msg.pose.pose

    def send_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = yaw
        self.cmd_vel_pub.publish(msg)


def load_wolfgang_param(namespace, rospack):
    rospy.set_param(namespace + '/robot_type_name', 'Wolfgang')
    set_param_to_file(namespace + "/robot_description", 'wolfgang_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", 'wolfgang_moveit_config',
                      '/config/wolfgang.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", 'wolfgang_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", 'wolfgang_moveit_config',
                       '/config/joint_limits.yaml', rospack)


if __name__ == '__main__':
    study = optuna.create_study(study_name='test')
    optimization = WalkPybulletOptimization('test', True)
    print("start optimization")
    study.optimize(optimization.objective, n_trials=5)
