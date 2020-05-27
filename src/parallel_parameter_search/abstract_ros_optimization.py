import rosgraph
import rospy
import rosnode
import roslaunch
import yaml
import sys


class AbstractRosOptimization:

    def __init__(self, namespace):
        # make all nodes use simulation time via /clock topic
        rospy.set_param('/use_sim_time', True)

        # unfortunately we can not start the node in our namespace. therefore it is anonymous, so names don't collide
        # rospy.names._set_caller_id('/bbb')
        # print(rosgraph.names.get_ros_namespace(argv='__ns:=/bbb'))
        # rospy.init_node('optimizer', anonymous=True)#, argv='__ns:=/aaa')

        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        node_names = rosnode.get_node_names()
        current_highest_number = -1
        for node_name in node_names:
            # see if the start of a node name after / is the one of the namespace
            if node_name[1:len(namespace) + 1] == namespace:
                # get the number behind it
                number = node_name[len(namespace) + 2:].split('/')[0]
                current_highest_number = max(current_highest_number, int(number))
        self.namespace = namespace + '_' + str(current_highest_number + 1)
        print(F"Will use namespace {self.namespace}")
        # launch a dummy node to show other workers that this namespace is taken
        self.dummy_node = roslaunch.core.Node('parallel_parameter_search', 'dummy_node.py', name='dummy_node',
                                              namespace=self.namespace)
        self.launch.launch(self.dummy_node)

        # we can not init node for specific name space, but we can remap the clock topic
        rospy.init_node('optimizer', anonymous=True, argv=['clock:=/' + self.namespace + '/clock'])

        while False:
            try:
                self.launch.spin_once()
            except KeyboardInterrupt:
                exit(0)

    def objective(self, trial):
        """
        The actual optimization target for optuna.
        """
        raise NotImplementedError()

    def close(self):
        """
        Destroys environment after finishing the optimization.
        """
        # todo maybe close all nodes in namespace?
        raise NotImplementedError()


class AbstractGazeboOptimization(AbstractRosOptimization):

    def __init__(self, namespace, gui):
        super(AbstractGazeboOptimization, self).__init__(namespace)


def set_param_to_file(param, package, file, rospack):
    path = rospack.get_path(package)
    with open(path + file, 'r') as file:
        file_content = file.read()
    rospy.set_param(param, file_content)


def load_yaml_to_param(namespace, package, file, rospack):
    path = rospack.get_path(package)
    with open(path + file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    for key in data.keys():
        # sometimes first level contains another yaml dict of values
        if isinstance(data[key], dict):
            for key_2 in data[key]:
                rospy.set_param(namespace + '/' + key + '/' + key_2, data[key][key_2])
        else:
            rospy.set_param(namespace + '/' + key, data[key])
