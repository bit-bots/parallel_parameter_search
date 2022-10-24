from parallel_parameter_search.simulators import WebotsSim
from rclpy.node import Node
import random
import os
import rclpy
import time

rclpy.init()
namespace = "anon_" + str(os.getpid()) + "_" + str(random.randint(0, 10000000)) + "_"
node = Node(namespace + "optimizer", allow_undeclared_parameters=True)
robot_name = "wolfgang"
sim = WebotsSim(node, True, robot_name, world="optimization_" + robot_name, ros_active=False)
while True:
    time.sleep(1)
sim.close()