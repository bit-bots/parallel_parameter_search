import rospy
import rosnode
import roslaunch
import threading


class AbstractRosOptimization:

    def __init__(self, namespace):
        # make all nodes use simulation time via /clock topic
        rospy.set_param('/use_sim_time', True)
        self.current_params = None

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

        self.dynconf_client = None
        self.sim = None

        while False:
            try:
                self.launch.spin_once()
            except KeyboardInterrupt:
                exit(0)

    def set_params(self, param_dict):
        self.current_params = param_dict
        # need to let run clock while setting parameters, otherwise service system behind it will block
        # let simulation run in a thread until dyn reconf setting is finished
        stop_clock = False

        def clock_thread():
            while not stop_clock or rospy.is_shutdown():
                self.sim.step_sim()

        dyn_thread = threading.Thread(target=self.dynconf_client.update_configuration, args=[param_dict])
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
