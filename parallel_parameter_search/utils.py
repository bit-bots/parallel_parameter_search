import math

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory


def set_param_to_file(node, param, package, file):
    path = get_package_share_directory(package)
    with open(path + file, 'r') as file:
        file_content = file.read()
    node.set_parameters([rclpy.parameter.Parameter(param, rclpy.Parameter.Type.DOUBLE, file_content)])


def load_yaml_to_param(node, namespace, package, file):
    path = get_package_share_directory(package)
    with open(path + file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in data.items():
        # sometimes first level contains another yaml dict of values
        if isinstance(value, dict):
            for key_2, value_2 in value.items():
                node.set_parameters([rclpy.parameter.Parameter(namespace + '/' + key + '/' + key_2, rclpy.Parameter.Type.DOUBLE, value_2)])
        else:
            node.set_parameters([rclpy.parameter.Parameter(namespace + '/' + key, rclpy.Parameter.Type.DOUBLE, value)])
    return data

def load_robot_param(node, namespace, name):
    node.set_parameters([rclpy.parameter.Parameter(namespace + '/robot_type_name', rclpy.Parameter.Type.DOUBLE, name)])
    set_param_to_file(node, namespace + "/robot_description", name + '_description', '/urdf/robot.urdf')
    set_param_to_file(node, namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf')
    load_yaml_to_param(node, namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml')
    load_yaml_to_param(node, namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml')


def fused_from_quat(q):
    # Fused yaw of Quaternion
    fused_yaw = 2.0 * math.atan2(q[2], q[3])  # Output of atan2 is [-tau/2,tau/2], so this expression is in [-tau,tau]
    if fused_yaw > math.tau / 2:
        fused_yaw -= math.tau  # fused_yaw is now in[-2* pi, pi]
    if fused_yaw <= -math.tau / 2:
        fused_yaw += math.tau  # fused_yaw is now in (-pi, pi]

    # Calculate the fused pitch and roll
    stheta = 2.0 * (q[1] * q[3] - q[0] * q[2])
    sphi = 2.0 * (q[1] * q[2] + q[0] * q[3])
    if stheta >= 1.0:  # Coerce stheta to[-1, 1]
        stheta = 1.0
    elif stheta <= -1.0:
        stheta = -1.0
    if sphi >= 1.0:  # Coerce sphi to[-1, 1]
        sphi = 1.0
    elif sphi <= -1.0:
        sphi = -1.0
    fused_pitch = math.asin(stheta)
    fused_roll = math.asin(sphi)

    # compute hemi parameter
    hemi = (0.5 - (q[0] * q[0] + q[1] * q[1]) >= 0.0)
    return fused_roll, fused_pitch, fused_yaw, hemi
