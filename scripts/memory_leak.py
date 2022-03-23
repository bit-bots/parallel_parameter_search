import resource

import rclpy
from ament_index_python import get_package_share_directory
from bitbots_msgs.msg import FootPressure
from bitbots_quintic_walk_py.py_walk import PyWalk
from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml
from geometry_msgs.msg import Twist
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu, JointState

twist_msg = Twist()
imu_msg = Imu()
joint_state_msg = JointState()
pressure_msg = FootPressure()

robot_name = "wolfgang"
moveit_parameters = load_moveit_parameter(robot_name)
walk_parameters = get_parameters_from_ros_yaml("walking",
                                               f"{get_package_share_directory('bitbots_quintic_walk')}"
                                               f"/config/walking_{robot_name}_optimization.yaml",
                                               use_wildcard=True)
walk = PyWalk("torsten", walk_parameters + moveit_parameters)

while rclpy.ok():
    for i in range(1000):
        walk.step(0.01, twist_msg, imu_msg, joint_state_msg, pressure_msg, pressure_msg)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


    #serialize_message(msg)