import math
import time
import sys
import threading
import numpy as np

import optuna

import rospy
import roslaunch
import rospkg
import tf
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
        self.sim = Simulation(gui)
        self.sim_interface = ROSInterface(self.sim, namespace=self.namespace + '/', node=False)
        self.current_params = None

    def set_params(self, param_dict):
        self.current_params = param_dict
        # need to let run clock while setting parameters, otherwise service system behind it will block
        # let simulation run in a thread until dyn reconf setting is finished
        stop_clock = False

        def clock_thread():
            while not stop_clock or rospy.is_shutdown():
                self.sim_interface.step()

        dyn_thread = threading.Thread(target=self.dynconf_client.update_configuration, args=[param_dict])
        clock_thread = threading.Thread(target=clock_thread)
        clock_thread.start()
        dyn_thread.start()
        dyn_thread.join()
        stop_clock = True
        clock_thread.join()


class WalkPybulletOptimization(PybulletOptimization):
    def __init__(self, namespace, gui):
        super(WalkPybulletOptimization, self).__init__(namespace, gui)
        rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_wolfgang_param(self.namespace, rospack)

        # load walk params
        # todo load a another config file that only provides the parameters necessary for optimization, i.e. no speed limits
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk', '/config/walking_wolfgang_optimization.yaml',
                           rospack)

        self.walk_node = roslaunch.core.Node('bitbots_quintic_walk', 'WalkNode', 'walking',
                                             namespace=self.namespace)
        self.walk_node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock")]
        self.launch.launch(self.walk_node)
        self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/engine', timeout=60)

        self.cmd_vel_pub = rospy.Publisher(self.namespace + '/cmd_vel', Twist, queue_size=10)

        self.robot_pose = None
        self.number_of_tries = 2
        self.time_limit = 20

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)

        cost = 0
        # starting with hardest first, to get faster early termination
        directions = [[0.1, 0, 0],
                      [0.1, 0.1, 1],
                      [-0.1, -0.1, -1],
                      [-0.1, 0, 0],
                      [0, 0.1, 0],
                      [0, -0.1, 0],
                      [0, 0, 1],
                      [0, 0, -1]]

        for eval_try in range(0, self.number_of_tries):
            for direction in directions:
                self.reset()
                cost_try = self.evaluate_direction(*direction, trial, eval_try)
                # check if we failed in this direction and terminate this trial early
                if cost_try is None:
                    return np.inf
                cost += cost_try

        # return mean cost per try
        return cost / (self.number_of_tries * len(directions))

    def suggest_walk_params(self, trial):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 0.2, 3)
        add('foot_apex_phase', 0.3, 0.7)
        add('foot_distance', 0.15, 0.25)
        add('foot_rise', 0.07, 0.1)
        add('foot_overshoot_phase', 0.7, 1.0)
        add('foot_overshoot_ratio', 0.0, 0.5)
        add('trunk_height', 0.33, 0.48)
        add('trunk_phase', -0.5, 0.5)
        add('trunk_pitch', -0.2, 0.4)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_x_offset', -0.05, 0.05)
        add('trunk_y_offset', -0.05, 0.05)
        add('first_step_swing_factor', 0.5, 2)
        add('first_step_trunk_phase', -0.5, 0.5)

        self.set_params(param_dict)

    def evaluate_direction(self, x, y, yaw, trial: optuna.Trial, step):
        start_time = rospy.get_time()
        self.send_cmd_vel(x, y, yaw)

        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = rospy.get_time() - start_time
            if passed_time > self.time_limit:
                # reached time limit, evaluate the fitness
                return self.measure_fitness(x, y, yaw)

            # todo
            """    
            # test if robot moved at least a bit        
            if not self.robot_has_moved and current_time > 20 :
                # has not moved 
                rospy.loginfo("robot didn't move")
                return True
            """
            # test if the robot has fallen down
            if self.sim.get_robot_pose()[0][2] < 0.3:
                # add extra information to trial
                #trial.set_user_attr('early_termination_at', (x, y, yaw))
                # prune this trial, aka early termination
                #trial.report(100, step)
                #raise optuna.exceptions.TrialPruned()
                # todo think about if pruning is really the best idea, or if a low fitness value should be returned
                return None

            try:
                self.sim_interface.step()
                # give time to other algorithms to compute their responses
                # use wall time, as ros time is standing still
                time.sleep(0.0001)
            except:
                pass

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def measure_fitness(self, x, y, yaw):
        """
        x,y,yaw have to be either 1 for being goo, -1 for being bad or 0 for making no difference
        """
        # factor to increase the weight of the yaw since it is a different unit then x and y
        yaw_factor = 5
        # todo better formular
        pos, rpy = self.sim.get_robot_pose_rpy()
        # 2D pose
        current_pose = [pos[0], pos[1], rpy[2]]
        # test if robot moved at all for simple case
        if x != 0 and y == 0 and yaw == 0 and current_pose[0] < 0.25:
            print("Did not move enough")
            return None
        correct_pose = [x * self.time_limit,
                        y * self.time_limit,
                        yaw * self.time_limit / (2 * math.pi)]  # todo we dont take multipe rotations into account
        cost = abs(current_pose[0] - correct_pose[0]) \
               + abs(current_pose[1] - correct_pose[1]) \
               + abs(current_pose[2] - correct_pose[2]) * yaw_factor
        # method doesn't work for going forward and turning at the same time
        # todo better computation of correct end pose
        if (x != 0 or y != 0) and yaw != 0:
            # just give 0 cost for surviving
            cost = 0
        print(F"cost: {cost}")
        return cost

    def reset(self):
        # reset simulation
        self.sim.reset()
        # let the robot do a few steps in the air to get correct walkready position
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.send_cmd_vel(0.1, 0, 0)
        self.sim_interface.run_simulation(duration=2, sleep=0.001)
        self.send_cmd_vel(0, 0, 0)
        self.sim_interface.run_simulation(duration=2, sleep=0.001)
        self.sim.set_gravity(True)
        height = self.current_params['trunk_height'] + 0.005
        pitch = self.current_params['trunk_pitch']
        (x, y, z, w) = tf.transformations.quaternion_from_euler(0, pitch, 0)

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))
        self.sim_interface.run_simulation(duration=1, sleep=0.001)

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
    study.optimize(optimization.objective, n_trials=20)
    # todo other possibilities to use it: kick, stand up, vision, path planning
    # todo set damping in joints
    # todo make a version that works on actual robot, using HCM and maybe buttons
