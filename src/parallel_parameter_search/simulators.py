import os
import subprocess
import sys
import time

import rospkg
import rospy
from wolfgang_pybullet_sim.simulation import Simulation
from wolfgang_pybullet_sim.ros_interface import ROSInterface
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param

from bitbots_msgs.msg import JointCommand, FootPressure

from wolfgang_webots_sim.utils import fix_webots_folder

from wolfgang_webots_sim.webots_controller import WebotsController


class AbstractSim:

    def __init__(self):
        pass

    def step_sim(self):
        raise NotImplementedError

    def run_simulation(self, duration, sleep):
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and (duration is None or rospy.get_time() - start_time < duration):
            self.step_sim()
            time.sleep(sleep)

    def set_gravity(self, on):
        raise NotImplementedError

    def reset_robot_pose(self, pos, quat):
        raise NotImplementedError

    def get_robot_pose_rpy(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_time(self):
        raise NotImplementedError

    def set_joints(self, joint_command_msg):
        raise NotImplementedError

    def set_joints_dict(self, dict):
        msg = JointCommand()
        for key in dict.keys():
            msg.joint_names.append(key)
            msg.positions.append(dict[key])
        self.set_joints(msg)

    def randomize_terrain(self, max_height):
        raise NotImplementedError

    def get_pressure_left(self):
        raise NotImplementedError

    def get_pressure_right(self):
        raise NotImplementedError

    def place_ball(self, x, y):
        raise NotImplementedError


class PybulletSim(AbstractSim):

    def __init__(self, namespace, gui, urdf_path=None, foot_link_names=[], terrain=False):
        super(AbstractSim, self).__init__()
        self.namespace = namespace
        # load simuation params
        rospack = rospkg.RosPack()
        # print(self.namespace)
        load_yaml_to_param("/" + self.namespace, 'wolfgang_pybullet_sim', '/config/config.yaml', rospack)
        self.gui = gui
        self.sim: Simulation = Simulation(gui, urdf_path=urdf_path, foot_link_names=foot_link_names, terrain=terrain)
        self.sim_interface: ROSInterface = ROSInterface(self.sim, namespace="/" + self.namespace + '/', node=False)

    def step_sim(self):
        self.sim.step()

    def set_gravity(self, on):
        self.sim.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.sim.reset_robot_pose(pos, quat)

    def get_robot_pose_rpy(self):
        return self.sim.get_robot_pose_rpy()

    def reset(self):
        self.sim.reset()

    def get_time(self):
        return self.sim.time

    def get_imu_msg(self):
        return self.sim_interface.get_imu_msg()

    def get_joint_state_msg(self):
        return self.sim_interface.get_joint_state_msg()

    def set_joints(self, joint_command_msg):
        self.sim_interface.joint_goal_cb(joint_command_msg)

    def get_timestep(self):
        return self.sim.timestep

    def randomize_terrain(self, max_height):
        self.sim.terrain.randomize(max_height)

    def get_pressure_left(self):
        return self.sim_interface.get_pressure_filtered_left()

    def get_pressure_right(self):
        return self.sim_interface.get_pressure_filtered_right()


class WebotsSim(AbstractSim):

    def __init__(self, namespace, gui, robot="wolfgang"):
        # start webots
        super().__init__()
        rospack = rospkg.RosPack()
        path = rospack.get_path("wolfgang_webots_sim")

        arguments = ["webots",
                     "--batch",
                     # TODO argument for world
                     path + "/worlds/kick_optimization.wbt"]
        if not gui:
            arguments.append("--minimize")
        sim_proc = subprocess.Popen(arguments)

        os.environ["WEBOTS_PID"] = str(sim_proc.pid)
        fix_webots_folder(sim_proc.pid)

        if gui:
            mode = 'normal'
        else:
            mode = 'fast'
        self.robot_controller = WebotsController(namespace, False, mode, robot)

    def step_sim(self):
        self.robot_controller.step()

    def set_gravity(self, on):
        self.robot_controller.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.robot_controller.reset_robot_pose(pos, quat)

    def set_robot_pose_rpy(self, pos, rpy):
        self.robot_controller.set_robot_pose_rpy(pos, rpy)

    def get_robot_pose_rpy(self):
        return self.robot_controller.get_robot_pose_rpy()

    def reset(self):
        self.robot_controller.reset()

    def get_time(self):
        return self.robot_controller.time

    def get_imu_msg(self):
        return self.robot_controller.get_imu_msg()

    def get_joint_state_msg(self):
        return self.robot_controller.get_joint_state_msg()

    def set_joints(self, joint_command_msg):
        self.robot_controller.command_cb(joint_command_msg)

    def get_timestep(self):
        # webots time step is in ms, so we need to convert
        return self.robot_controller.timestep / 1000

    def get_pressure_left(self):
        rospy.logwarn_once("pressure method not implemented")
        return FootPressure()

    def get_pressure_right(self):
        rospy.logwarn_once("pressure method not implemented")
        return FootPressure()

    def place_ball(self, x, y):
        self.robot_controller.set_ball_pose([x, y, 0.08])

    def get_ball_position(self):
        pos = self.robot_controller.get_ball_pose()
        return pos[0], pos[1]

    def get_ball_velocity(self):
        # only return linear velocity
        return self.robot_controller.get_ball_velocity()[:3]
