from time import sleep

import rclpy
from rclpy.node import Node
import rosnode
import roslaunch
import threading

from rclpy.time import Time
from rosgraph_msgs.msg import Clock


class AbstractRosOptimization(Node):

    def __init__(self, namespace):
        super().__init__("optimizer")
        # make all nodes use simulation time via /clock topic
        self.set_parameters([rclpy.parameter.Parameter('/use_sim_time', rclpy.Parameter.Type.DOUBLE, True)])
        self.current_params = None

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
        self.set_parameters([rclpy.parameter.Parameter("/" + self.namespace + '/use_sim_time', rclpy.Parameter.Type.DOUBLE, True)])
        # launch a dummy node to show other workers that this namespace is taken
        self.dummy_node = roslaunch.core.Node('parallel_parameter_search', 'dummy_node.py', name='dummy_node',
                                              namespace=self.namespace)
        self.launch.launch(self.dummy_node)

        self.dynconf_client = None
        self.sim = None

    def set_params(self, param_dict, client, node_to_spin=None):
        self.current_params = param_dict
        # need to let run clock while setting parameters, otherwise service system behind it will block
        # let simulation run in a thread until dyn reconf setting is finished
        stop_clock = False

        def clock_thread():
            pub = self.create_publisher(Clock, "/clock", 1)
            msg = Clock()
            while not stop_clock or not rclpy.ok():
                self.sim.step_sim()
                # this magic sleep is necessary because of reasons
                sleep(0.01)
                sim_time = self.sim.get_time()
                msg.clock = Time(seconds=int(sim_time), nanoseconds=sim_time % 1 * 1e9)
                pub.publish(msg)
                if node_to_spin:
                    node_to_spin.spin_ros()

        dyn_thread = threading.Thread(target=client.update_configuration, args=[param_dict])
        clock_thread = threading.Thread(target=clock_thread)
        clock_thread.start()
        dyn_thread.start()
        dyn_thread.join()
        stop_clock = True
        clock_thread.join()

    def objective(self, trial):
        """
        The actual optimization target for optuna.
        """
        raise NotImplementedError()
