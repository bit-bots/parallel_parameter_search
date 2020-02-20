#!/usr/bin/env python2.7
import dynamic_reconfigure.client
import rospy
import tf
import sys
import roslaunch
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from std_srvs.srv import Empty
from parallel_parameter_search.srv import RequestParameters, SubmitFitness
from gazebo_msgs.srv import GetModelState, SetModelState, SetPhysicsProperties
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates
import gazebo_msgs
from std_msgs.msg import Float64MultiArray 
from bitbots_msgs.msg import JointCommand
from sensor_msgs.msg import Imu
import dynamic_reconfigure.client
import math
from dynamic_reconfigure.msg import Config, DoubleParameter

class WalkWorker:
    """
    This worker class handles the testing of one Gazebo instance. 
    It requests parameters from the master does evaluation runs with them and returns fitness values for these sets.
    It will stop when the master returns no further parameter sets.
    """
    def __init__(self, number):
        rospy.init_node("worker", log_level=rospy.INFO)
        self.number = int(number)
        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()
        self.stop_try = False
        self.robot_has_moved = False

        # read in config        
        self.model_name = rospy.get_param("/worker/evaluation/model_name")
        self.link_name = rospy.get_param("/worker/evaluation/link_name")
        self.time_limit = rospy.get_param("/worker/evaluation/time_limit")
        self.number_of_tries = rospy.get_param("/worker/evaluation/number_of_tries")
        self.use_dyn_reconf = rospy.get_param("/worker/evaluation/use_dyn_reconf")
        self.dyn_reconf_name = rospy.get_param("/worker/evaluation/dyn_reconf_name")
        self.start_position = rospy.get_param("/worker/evaluation/start_position")
        self.start_orientation = rospy.get_param("/worker/evaluation/start_orientation")

        self.sim_time_step = rospy.get_param("/worker/simulation/sim_time_step")
        self.sim_max_update_rate = rospy.get_param("/worker/simulation/sim_max_update_rate")
        self.ode_sor_pgs_precon_iters = rospy.get_param("/worker/simulation/ode_sor_pgs_precon_iters")
        self.ode_sor_pgs_iters = rospy.get_param("/worker/simulation/ode_sor_pgs_iters")
        self.ode_sor_pgs_w = rospy.get_param("/worker/simulation/ode_sor_pgs_w")
        self.ode_sor_pgs_rms_error_tol = rospy.get_param("/worker/simulation/ode_sor_pgs_rms_error_tol")
        self.ode_contact_surface_layer = rospy.get_param("/worker/simulation/ode_contact_surface_layer")
        self.ode_contact_max_correcting_vel = rospy.get_param("/worker/simulation/ode_contact_max_correcting_vel")
        self.ode_cfm = rospy.get_param("/worker/simulation/ode_cfm")
        self.ode_erp = rospy.get_param("/worker/simulation/ode_erp")

        #####
        self.joint_goal_publisher = rospy.Publisher('DynamixelController/command', JointCommand, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.cmd_vel_msg = Twist()
        self.cmd_vel_msg.linear.x = 0.05
        self.cmd_vel_zero = Twist()
        self.imu_fall_counter = 0
        #####


        #self.model_state_sub = rospy.Subscriber("gazebo/model_states", ModelStates, self.model_state_callback)
        self.link_state_sub = rospy.Subscriber("gazebo/link_states", LinkStates, self.link_state_callback)
        rospy.wait_for_service('gazebo/reset_simulation', timeout=60)
        rospy.wait_for_service('gazebo/get_model_state', timeout=10)
        rospy.wait_for_service('/request_parameters', timeout=10)
        rospy.wait_for_service('/submit_fitness', timeout=10)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty, persistent=True)
        self.set_robot_pose = rospy.ServiceProxy('gazebo/set_model_state', SetModelState, persistent=True)
        self.get_robot_pose = rospy.ServiceProxy('gazebo/get_model_state', GetModelState, persistent=True)
        self.get_parameters = rospy.ServiceProxy('/request_parameters', RequestParameters)
        self.submit_fitness = rospy.ServiceProxy('/submit_fitness', SubmitFitness)
        self.set_gravity_serv = rospy.ServiceProxy('gazebo/set_physics_properties', SetPhysicsProperties, persistent=True)
        if self.use_dyn_reconf:
            self.dynconf_client = dynamic_reconfigure.client.Client(self.dyn_reconf_name, timeout=60)
        else:
            self.params_pub = rospy.Publisher("engine_params", Config)
            # start new node which should be evaluated
            #node = roslaunch.core.Node('bitbots_quintic_walk', 'QuinticWalkingNode')
            #node.namespace = rospy.get_namespace()
            #node.name = "walking"
            #node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock")]
            #self.process = self.launch.launch(node)

        # magic sleep to make sure everything is up and running, especially walking
        rospy.sleep(1)
        while not rospy.is_shutdown():
            self.do_run()

        # kill robot moving process
        #process.stop()

    def do_run(self):
        """
        Main method which evaluates a set of parameters and sends back the fitness.
        """
        rospy.loginfo("Starting new run")
        resp = None

        # get new set of parameters
        try:
            resp = self.get_parameters()
        except rospy.service.ServiceException:
            rospy.logwarn("Couldn't reach master. Propably there are no more values")
            rospy.signal_shutdown("Master did shutdown")
        set_number = resp.set_number
        # only do something if there is still work to do
        if resp.set_available:
            # set the parameters for this node
            self.set_params(resp.parameter_names, resp.parameters)    

            fitness = 0
            evaluation_time = 0
            times_stopped = 0
            for eval_try in range(0, self.number_of_tries):
                rospy.loginfo("try " + str(eval_try +1) + " of " + str(self.number_of_tries) + " for set " + str(set_number))
                
                # we test multiple skills and compute a resulting fitness
                # TODO maybe return 3 independent fitness values for the 3 walking directions

                # forward and sideward
                self.reinitilize_simulation()
                self.send_cmd_vel(-0.03,-0.03,0)                                                
                start_time = rospy.get_time()
                interupted = self.evaluate(start_time)
                evaluation_time += rospy.get_time() - start_time
                if interupted:
                    # we've fallen down. Fitness is zero
                    fitness = 0 
                    times_stopped += 1
                    # we dont want to do anything else
                    break
                self.send_cmd_vel(0.0,0,0)                            
                fitness += self.measure_fitness(1, 1, -1) 

                # sideward walking
                self.reinitilize_simulation()
                self.send_cmd_vel(0,0.05,0)                                                
                start_time = rospy.get_time()
                interupted = self.evaluate(start_time)
                evaluation_time += rospy.get_time() - start_time
                if interupted:
                    # we've fallen down. Fitness is zero
                    fitness = 0 
                    times_stopped += 1
                    # we dont want to do anything else
                    break
                self.send_cmd_vel(0.0,0,0)                            
                fitness += self.measure_fitness(1,-1,1) 

                # turning
                self.reinitilize_simulation()
                self.send_cmd_vel(0,0,0.05)                                                
                start_time = rospy.get_time()
                interupted = self.evaluate(start_time)
                evaluation_time += rospy.get_time() - start_time
                if interupted:
                    # we've fallen down. Fitness is zero
                    fitness = 0 
                    times_stopped += 1
                    # we dont want to do anything else
                    break
                self.send_cmd_vel(0.0,0,0)                            
                fitness += self.measure_fitness(-1,-1,1) 
                
                # forward walking
                self.reinitilize_simulation()
                self.send_cmd_vel(0.05,0,0)                                                
                start_time = rospy.get_time()
                interupted = self.evaluate(start_time)
                evaluation_time += rospy.get_time() - start_time
                if interupted:
                    # we've fallen down. Fitness is zero
                    fitness = 0 
                    times_stopped += 1
                    # we dont want to do anything else
                    break
                self.send_cmd_vel(0.0,0,0)                            
                fitness += self.measure_fitness(1,-1,-1) 

                # backward walking
                self.reinitilize_simulation()
                self.send_cmd_vel(-0.05,0,0)
                start_time = rospy.get_time()
                interupted = self.evaluate(start_time)
                evaluation_time += rospy.get_time() - start_time
                if interupted:
                    # we've fallen down. Fitness is zero
                    fitness = 0 
                    times_stopped += 1
                    # we dont want to do anything else
                    break
                self.send_cmd_vel(0.0,0,0)                            
                fitness += self.measure_fitness(1,-1,-1)


            # return fitness
            self.submit_fitness(self.number, set_number, fitness / self.number_of_tries, evaluation_time, times_stopped)

        else:
            rospy.loginfo("No more parameters to test")
            rospy.signal_shutdown("No more parameters to test")
            #TODO kill all other nodes of this worker (in this namespace)

    def evaluate(self, start_time):
        self.stop_try = False               
        # wait till time for test is up or stopping condition has been reached                
        while True:
            current_time = rospy.get_time() - start_time                
            if current_time > self.time_limit:
                # reached time limit
                return False                    
            """
            if not self.robot_has_moved and current_time > 20 :
                # has not moved 
                rospy.loginfo("robot didn't move")
                break
            """
            if self.stop_try:
                # fallen down
                rospy.loginfo("Stopping condition reached")
                self.stop_try = False
                return True

            try:
                rospy.sleep(0.01)
            except:
                pass

    def set_params(self, parameter_names, parameters):
        """
        Sets the parameters (depending on the config) either via dynamic reconfigure or to the parameter server.
        """
        # sanity check
        if len(parameter_names) != len(parameters):
            rospy.logerr("Number of paramter names does not match number of parameters! There is something wrong in the master.")
            exit()
        
        # set parameters either by dyn reconf or on normal parameter server
        if self.use_dyn_reconf:
            param_dict = {}
            for i in range(0, len(parameter_names)):
                param_dict[parameter_names[i]] = parameters[i]
            self.dynconf_client.update_configuration(param_dict)
        else:
            msg = Config()
            for i in range(0, len(parameter_names)):
                parameter = DoubleParameter()
                parameter.name = parameter_names[i]
                parameter.value = parameters[i]
                msg.doubles.append(parameter)
                #rospy.set_param(parameter_names[i], parameters[i])
            #TODO restart software
            rospy.logerr("send")
            self.params_pub.publish(msg)

        # here you can use parameters to alter the start pose
        ####
        i = 0
        for name in parameter_names:
            if name == "trunk_height":
                self.start_position[2] = parameters[i] + 0.01
            elif name == "trunk_pitch":
                self.start_orientation[1] = parameters[i]
            i += 1
        ####

    def reinitilize_simulation(self):
        """
        This reintilizes the simulation before each evaluation try.
        """
        # first put the robot into the air without gravitation so that it can safely go into a start position
        rospy.loginfo("setting model in the air and start walking")
        self.set_gravity(False)
        self.set_model_pose(0, 0, 2, 0, 0, 0)

        self.set_joints_to_start_position()
        
        # reset simulation
        #rospy.loginfo("resetting sim double")
        #self.reset_simulation()                
        # do double reset because physic engine is strange
        #self.reset_simulation()
        rospy.loginfo("resetting model pose")
        self.reset_model_pose()
        self.set_gravity(True)
        rospy.sleep(2) # to let the robot be stable before starting


    def set_joints_to_start_position(self):
        """
        Set the joints to the position that they should have at the start of the evaluation step.
        """
        ####
        self.cmd_vel_publisher.publish(self.cmd_vel_msg)
        rospy.sleep(1)
        rospy.loginfo("stop walking")
        self.cmd_vel_publisher.publish(self.cmd_vel_zero)
        rospy.sleep(2)
        ####

    def start_try(self):
         # give command to start running
        rospy.loginfo("start walking")  
        self.cmd_vel_publisher.publish(self.cmd_vel_msg)

    def measure_fitness(self, x, y, yaw):
        """
        x,y,yaw have to be either 1 for being goo, -1 for being bad or 0 for making no difference
        """
        # get position of robot
        resp = self.get_robot_pose(self.model_name, "world")
        # factor to increase the weight of the yaw since it is a different unit then x and y
        yaw_factor = 5

        return math.sqrt(max(x * resp.pose.position.x**2 + 
                            y * resp.pose.position.y **2 + 
                            yaw * resp.pose.orientation.z**2 * yaw_factor, 0.0))


    def model_state_callback(self, msg):
        #last entry
        index_robot = len(msg.name) - 1
        position = msg.pose[index_robot].position
        self.robot_has_moved = position.x < 0.5 and position.y < 0.5        
            
    def link_state_callback(self, msg):
        """
        Stop try if head is below threshold, since robot has fallen down
        """
        #for entry in msg.name:
        #    rospy.logwarn(entry)
        if self.link_name in msg.name:
            index = msg.name.index(self.link_name)
            if msg.pose[index].position.z < 0.4:
                self.stop_try = True
        else:
            rospy.logwarn_throttle(2, "head is not in link information")

    def reset_model_pose(self):
        position = self.start_position
        orientation = self.start_orientation
        self.set_model_pose(position[0], position[1], position[2], orientation[0], orientation[1], orientation[2])    

    def set_model_pose(self, x, y, z, roll, pitch, yaw):
        req = gazebo_msgs.srv.SetModelStateRequest()
        msg = ModelState()
        msg.model_name = self.model_name
        pose_msg = Pose()
        position_msg = Point()
        position_msg.x = x
        position_msg.y = y        
        position_msg.z = z       
        pose_msg.position = position_msg
        or_msg = Quaternion()
        (x,y,z,w) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        or_msg.x = x
        or_msg.y = y
        or_msg.z = z
        or_msg.w = w
        pose_msg.orientation = or_msg
        msg.pose = pose_msg
        msg.twist = Twist()
        msg.reference_frame = "world"
        req.model_state = msg
        self.set_robot_pose(req)

    def set_gravity(self, on):
        req = gazebo_msgs.srv.SetPhysicsPropertiesRequest()    
        req.time_step = self.sim_time_step
        req.max_update_rate = self.sim_max_update_rate 
        #req.ode_config auto_disable_bodies
        req.ode_config.sor_pgs_precon_iters = self.ode_sor_pgs_precon_iters 
        req.ode_config.sor_pgs_iters = self.ode_sor_pgs_iters 
        req.ode_config.sor_pgs_w = self.ode_sor_pgs_w 
        req.ode_config.sor_pgs_rms_error_tol = self.ode_sor_pgs_rms_error_tol 
        req.ode_config.contact_surface_layer = self.ode_contact_surface_layer 
        req.ode_config.contact_max_correcting_vel = self.ode_contact_max_correcting_vel
        req.ode_config.cfm = self.ode_cfm
        req.ode_config.erp =  self.ode_erp
        #uint32 max_contacts

        if on:
            req.gravity.z = -9.81
        else: 
            req.gravity.z = 0
        self.set_gravity_serv(req)

    def send_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = yaw
        self.cmd_vel_publisher.publish(msg)

if __name__ == "__main__":
    WalkWorker(sys.argv[1])
