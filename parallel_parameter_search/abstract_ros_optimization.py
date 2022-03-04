import rclpy
from bitbots_moveit_bindings.libbitbots_moveit_bindings import initRos
from rclpy.node import Node


class AbstractRosOptimization:

    def __init__(self, robot_name):
        self.robot_name = robot_name
        # need to init ROS for python and c++ code
        rclpy.init()
        initRos()
        self.node: Node = Node("optimizer", allow_undeclared_parameters=True)
        # make all nodes use simulation time via /clock topic
        # actually not since we use direct python interfaces and the simulation runs in the same thread
        # self.node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.current_params = None
        self.sim = None

    def objective(self, trial):
        """
        The actual optimization target for optuna.
        """
        raise NotImplementedError()
