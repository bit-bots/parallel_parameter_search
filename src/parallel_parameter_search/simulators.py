import os
import subprocess
import sys
import time
from abc import ABC

import rospkg
import rospy
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
from wolfgang_pybullet_sim.simulation import Simulation
from wolfgang_pybullet_sim.ros_interface import ROSInterface
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param

from bitbots_msgs.msg import JointCommand, FootPressure

try:
    from wolfgang_webots_sim.webots_robot_supervisor_controller import RobotSupervisorController
except:
    rospy.logerr("Could not load webots sim. If you want to use it, source the setenvs.sh")


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

    def set_self_collision(self, active):
        raise NotImplementedError

    def reset_robot_pose(self, pos, quat, reset_joints=False):
        raise NotImplementedError

    def get_robot_pose(self):
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

    def set_ball_position(self, x, y):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

class PybulletSim(AbstractSim):

    def __init__(self, namespace, gui, urdf_path=None, foot_link_names=[], terrain=False, field=False, robot="wolfgang"):
        super(AbstractSim, self).__init__()
        self.namespace = namespace
        # load simuation params
        rospack = rospkg.RosPack()
        # print(self.namespace)
        load_yaml_to_param("/" + self.namespace, 'wolfgang_pybullet_sim', '/config/config.yaml', rospack)
        self.gui = gui
        self.sim: Simulation = Simulation(gui, urdf_path=urdf_path, foot_link_names=foot_link_names, terrain=terrain,
                                          field=field, robot=robot)
        self.sim_interface: ROSInterface = ROSInterface(self.sim, namespace="/" + self.namespace + '/', node=False)

    def step_sim(self):
        self.sim_interface.step()

    def set_gravity(self, on):
        self.sim.set_gravity(on)

    def reset_simulation(self):
        self.sim.reset_simulation()

    def reset_robot_pose(self, pos, quat, reset_joints=False):
        self.sim.reset_robot_pose(pos, quat, reset_joints=reset_joints)

    def set_robot_pose(self, pos, quat):
        self.sim.set_robot_pose(pos, quat)

    def get_robot_pose(self):
        return self.sim.get_robot_pose()

    def get_robot_pose_rpy(self):
        return self.sim.get_robot_pose_rpy()

    def get_robot_velocity(self):
        return self.sim.get_robot_velocity()

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

    def get_link_pose(self, link_name):
        return self.sim.get_link_pose(link_name)

    def randomize_terrain(self, max_height):
        self.sim.terrain.randomize(max_height)

    def apply_force(self, link_id, force, position):
        self.sim.apply_force(link_id, force, position)

    def get_pressure_left(self):
        return self.sim_interface.get_pressure_filtered_left()

    def get_pressure_right(self):
        return self.sim_interface.get_pressure_filtered_right()

    def get_joint_position(self, name):
        return self.sim.get_joint_position(name)

    def set_ball_position(self, x, y):
        pass

    def get_joint_names(self):
        return self.sim.get_joint_names()

    def set_self_collision(self, active):
        rospy.logwarn_once("self collision in pybullet has to be set during loading of URDF")
        return

    def close(self):
        return

class WebotsSim(AbstractSim, ABC):

    def __init__(self, namespace, gui, robot="wolfgang", ros_active=False, world="robot_supervisor"):
        # start webots
        super().__init__()
        rospack = rospkg.RosPack()
        self.ros_active = ros_active
        if ros_active:
            self.true_odom_publisher = rospy.Publisher(namespace + "/true_odom", Odometry, queue_size=1)
        path = rospack.get_path("wolfgang_webots_sim")

        arguments = ["webots",
                     "--batch",
                     path + "/worlds/" + world + ".wbt"]
        if not gui:
            arguments.append("--minimize")
            arguments.append("--no-rendering")
        self.sim_proc = subprocess.Popen(arguments, stdout=subprocess.PIPE)

        os.environ["WEBOTS_PID"] = str(self.sim_proc.pid)

        if gui:
            mode = 'normal'
        else:
            mode = 'fast'

        self.robot_controller = RobotSupervisorController(ros_active, mode, robot, base_ns=namespace + '/',
                                                          model_states_active=False, camera_active=False)

    def step_sim(self):
        self.robot_controller.step()
        if self.ros_active:
            self.publish_true_odom()

    def publish_true_odom(self):
        position, orientation = self.robot_controller.get_robot_pose_quat()
        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position = Point(*position)
        odom_msg.pose.pose.orientation = Quaternion(*orientation)
        self.true_odom_publisher.publish(odom_msg)

    def set_gravity(self, on):
        self.robot_controller.set_gravity(on)

    def set_self_collision(self, active):
        self.robot_controller.set_self_collision(active)

    def reset_robot_pose(self, pos, quat, reset_joints=False):
        self.robot_controller.reset_robot_pose(pos, quat)

    def set_robot_pose(self, pos, quat):
        self.robot_controller.set_robot_pose_quat(pos, quat)

    def set_robot_pose_rpy(self, pos, rpy):
        self.robot_controller.set_robot_pose_rpy(pos, rpy)

    def get_robot_pose(self):
        return self.robot_controller.get_robot_pose_quat()

    def get_robot_pose_rpy(self):
        return self.robot_controller.get_robot_pose_rpy()

    def reset(self):
        self.robot_controller.reset()

    def reset_robot_init(self):
        self.robot_controller.reset_robot_init()

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

    def get_link_pose(self, link_name):
        return self.robot_controller.get_link_pose(link_name)

    def set_ball_position(self, x, y):
        self.robot_controller.set_ball_pose([x, y, 0])

    def get_joint_names(self):
        return self.robot_controller.get_joint_state_msg().name

    def get_robot_velocity(self):
        msg = self.robot_controller.get_imu_msg().angular_velocity
        return None, (msg.x, msg.y, msg.z)

    def get_joint_position(self, name):
        msg = self.robot_controller.get_joint_state_msg()
        for i in range(0, len(msg.name)):
            if name == msg.name[i]:
                return msg.position[i]
        sys.exit(f"joint {name} not found")

    def close(self):
        print("hi")
        self.sim_proc.terminate()
        self.sim_proc.wait()
        self.sim_proc.kill()
