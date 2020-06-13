import os
import subprocess
import sys
import time

import rospy
from wolfgang_pybullet_sim.simulation import Simulation
from wolfgang_pybullet_sim.ros_interface import ROSInterface


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


class PybulletSim(AbstractSim):

    def __init__(self, namespace, gui):
        super(AbstractSim, self).__init__()
        self.namespace = namespace
        self.gui = gui
        self.sim: PybulletSim = Simulation(gui)
        self.sim_interface: ROSInterface = ROSInterface(self.sim, namespace=self.namespace + '/', node=False)

    def step_sim(self):
        self.sim_interface.step()

    def set_gravity(self, on):
        self.sim.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.sim.reset_robot_pose(pos, quat)

    def get_robot_pose_rpy(self):
        return self.sim.get_robot_pose_rpy()

    def reset(self):
        self.sim.reset()


class WebotsSim(AbstractSim):

    def __init__(self, namespace, gui):
        # start webots
        super().__init__()
        arguments = ["webots",
                     "--batch",
                     sys.path[0][:-23] + "/worlds/RunningRobotEnv_extern.wbt"]
        # todo load different world that is empty
        if not gui:
            arguments.append("--minimize")
        sim_proc = subprocess.Popen(arguments)

        os.environ["WEBOTS_PID"] = str(sim_proc.pid)
        self.robot_controller = darwin_ros.DarwinController(sim_proc.pid, namespace)

    def step_sim(self):
        self.robot_controller.step_sim()

    def set_gravity(self, on):
        self.robot_controller.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.robot_controller.reset_robot_pose(pos, quat)

    def get_robot_pose_rpy(self):
        self.robot_controller.get_robot_rpy()

    def reset(self):
        self.robot_controller.reset()

# todo gazebo
