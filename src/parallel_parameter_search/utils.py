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
