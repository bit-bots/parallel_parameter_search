import os
import resource
import random

import rclpy
from ament_index_python import get_package_share_directory
from bitbots_msgs.msg import FootPressure
from bitbots_moveit_bindings.libbitbots_moveit_bindings import initRos
from bitbots_quintic_walk_py.py_walk import PyWalk
from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import Imu, JointState


import os
import psutil
twist_msg = Twist()
twist_msg.linear.x = 0.01
imu_msg = Imu()
imu_msg.orientation.w = 1.0
joint_state_msg = JointState()
pressure_msg = FootPressure()

rclpy.init()
initRos()
namespace = "anon_" + str(os.getpid()) + "_" + str(random.randint(0, 10000000)) + "_"
node = Node(namespace + "optimizer", allow_undeclared_parameters=True)

robot_name = "wolfgang"
moveit_parameters = load_moveit_parameter(robot_name)
walk_parameters = get_parameters_from_ros_yaml("walking",
                                               f"{get_package_share_directory('bitbots_quintic_walk')}"
                                               f"/config/walking_{robot_name}_simulator.yaml",
                                               use_wildcard=True)
walk = PyWalk(namespace, walk_parameters + moveit_parameters)
i=0
last_memory = 0
while rclpy.ok():
    #for i in range(100):
    i += 1

    walk.test_memory_leak_methods(serialize_message(twist_msg))

    # these are okay
    # deserialize_message(serialize_message(twist_msg), Twist)
    # walk.reset()
    # walk.get_phase()

    # these lead to memory leaks
    # walk.step(0.01, twist_msg, imu_msg, joint_state_msg, pressure_msg, pressure_msg)
    # walk.get_left_foot_pose()
    # walk.test_memory_leak_to()
    # walk.test_memory_leak_from(twist_msg)

    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_percent()
    for child in current_process.children(recursive=True):
        mem += child.memory_percent()
    #print(mem)
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss
    if last_memory != memory:
        print(i)
        i = 0
    if i % 100 == 0:
        print(str(memory) + "    " + str(mem))
    last_memory = memory


    #serialize_message(msg)