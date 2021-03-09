import math

import yaml
import rospy


def set_param_to_file(param, package, file, rospack):
    path = rospack.get_path(package)
    with open(path + file, 'r') as file:
        file_content = file.read()
    rospy.set_param(param, file_content)


def load_yaml_to_param(namespace, package, file, rospack):
    path = rospack.get_path(package)
    with open(path + file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in data.items():
        # sometimes first level contains another yaml dict of values
        if isinstance(value, dict):
            for key_2, value_2 in value.items():
                rospy.set_param(namespace + '/' + key + '/' + key_2, value_2)
        else:
            rospy.set_param(namespace + '/' + key, value)


def fused_from_quat(q):
    # Fused yaw of Quaternion
    fused_yaw = 2.0 * math.atan2(q[2], q[3])  # Output of atan2 is [-pi,pi], so this expression is in [-2*pi,2*pi]
    if fused_yaw > math.pi:
        fused_yaw -= 2 * math.pi  # fused_yaw is now in[-2 * pi, pi]
    if fused_yaw <= -math.pi:
        fused_yaw += 2 * math.pi  # fused_yaw is now in (-pi, pi]

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
