#!/usr/bin/env python3
import csv
from random import uniform

import os
import sys
import rospy
import time
from parallel_parameter_search.srv import RequestParameters, SubmitFitness, RequestParametersResponse


class TrainMaster:
    """
    The master distributes parameter sets to the workers and records the resulting fitness values.
    The method of generating those parameter sets can be defined, e.g. random or grid search.
    """
    def __init__(self):
        rospy.init_node('train_master', anonymous=False, log_level=rospy.INFO)        

        # read config
        self.folder_path = rospy.get_param("/master/file_path")
        self.search_method = rospy.get_param("/master/search_method")
        self.parameters = rospy.get_param("/master/parameters")
        self.parameter_name_order = list(self.parameters.keys())

        self.finished = False
        # 2D array. first dimension is set number, second is array of parameter values
        self.parameter_sets = []
        # array of tuples with set_number and corresponding fitness value
        self.fitness_values = []
        self.sets_beeing_evaluated = set()
        self.current_set_index = 0
        self.fitness_values_returned = 0
        self.best_fitness_value = 0

        path = os.getenv("HOME") + self.folder_path
        name = "master"
        ending = ".csv"
        # find number that we have to use in order to not overwrite any data
        i = 0
        while True:
            if not os.path.isfile(path + name + str(i) + ending):
                self.output_file_path = path + name + str(i) + ending
                break
            i += 1
        self.output_file = open(self.output_file_path, "w")
        self.writer = csv.writer(self.output_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.writer.writerow(["fitness","evaluation_time","times_stopped","worker_number"] + self.parameter_name_order)
        self.output_file.flush()

        if self.search_method == "random":
            self.generate_set = self.next_set_random
            self.number_of_sets_to_test = rospy.get_param("/master/number_of_sets_to_test")
        elif self.search_method == "config":
         # just sent empy message so nothing is changed
            self.generate_set = self.empty_set
            self.number_of_sets_to_test = rospy.get_param("/master/number_of_sets_to_test")
        elif self.search_method == "grid":       
            self.generate_grid()     
            self.generate_set = self.next_set_grid
            self.number_of_sets_to_test = 1
            # compute number of possible permutations for grid
            for arr in self.parameter_grid:
                self.number_of_sets_to_test = self.number_of_sets_to_test * len(arr)
            # init counters for all dimensions of the parameter grid
            self.grid_iteration_counters = [0] * len(self.parameter_grid)
            rospy.logwarn("Starting grid search with " + str(self.number_of_sets_to_test) + " permuations.")
        elif self.search_method == "fixed":
            self.generate_set = self.fixed_set
            self.number_of_sets_to_test = rospy.get_param("/master/number_of_sets_to_test")
        else:
            rospy.logerr("Search method not valid. Chose random or grid.")
            rospy.signal_shutdown("fail")

        # Initialize Services
        self.req_param_service = rospy.Service("/request_parameters", RequestParameters,
                                               self.requestParameterCall)
        self.sub_fitness_service = rospy.Service("/submit_fitness", SubmitFitness,
                                                 self.submitFitnessCall)

        while not rospy.is_shutdown() and not self.finished:
            # use python time, since use_sim_time is true but /clock is not published for master
            time.sleep(1)
        self.output_file.close()
        print("### Closing master, your fitness csv file has been written to " + self.output_file_path + " ###")
        rospy.signal_shutdown("finished")

    def requestParameterCall(self, req):
        rospy.logdebug("Call from worker %s", req._connection_header['callerid'])
        resp = RequestParametersResponse()
        if self.current_set_index < self.number_of_sets_to_test:
            rospy.logdebug("Providing parameter set number %d", self.current_set_index)
            resp.set_available = True
            self.parameter_sets.append(self.generate_set())
            resp.parameters = self.parameter_sets[self.current_set_index]
            resp.set_number = self.current_set_index
            resp.parameter_names = self.parameter_name_order
            if not self.search_method == "config":
                resp.parameter_names = self.parameter_name_order
            else:
                resp.parameter_names = []
            self.sets_beeing_evaluated.add(self.current_set_index)
            self.current_set_index += 1
        else:
            rospy.loginfo("No more parameter sets to evaluate")
            resp.set_available = False
        return resp

    def submitFitnessCall(self, req):
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K") #clear line
        rospy.logwarn("Fitness submission by worker %d for set %d: stopped %d value %f", req.worker_number, req.set_number, req.times_stopped, req.fitness)
        # sanity check
        if req.set_number not in self.sets_beeing_evaluated:
            rospy.logerr("A fitness value for a wrong parameter set has been returned. Will not record this parameter.")
            return False
        self.fitness_values.append((req.set_number, req.fitness))
        self.fitness_values_returned += 1
        if req.fitness > self.best_fitness_value:
            self.best_fitness_value = req.fitness
        rospy.logwarn("%d of %d fitness values were returned. Current best is %f", self.fitness_values_returned,
                      self.number_of_sets_to_test, self.best_fitness_value)
        # write this value to the csv file
        self.writer.writerow([req.fitness] + [req.evaluation_time] + [req.times_stopped] + [req.worker_number] + self.parameter_sets[req.set_number])
        self.output_file.flush()
        self.sets_beeing_evaluated.remove(req.set_number)
        if self.fitness_values_returned == self.number_of_sets_to_test:
            # we are finished
            self.finished = True
        return True

    def empty_set(self):
        return []

    def fixed_set(self):
        di = {'trunk_height': 0.39981340504843704, 'foot_rise': 0.10539253465724091, 'freq': 0.60083165829638818, 'trunk_phase': 0.0011266795322085962, 'double_support_ratio': 0.39771674949947405, 'trunk_pitch': 0.24365538745932985, 'foot_distance': 0.16000194813586835, 'trunk_swing': 0.50133271709104776}




        values = []
        for i in range(0,len(self.parameter_name_order)):
            parameter = self.parameter_name_order[i]
            values.append(di[parameter])
        return values

    def next_set_random(self):
        """Generate a parameter set based on random numbers."""
        values = []
        # generate random numbers in usable areas
        for i in range(0,len(self.parameter_name_order)):
            parameter = self.parameter_name_order[i]
            min_val = self.parameters[parameter]["min"]
            max_val = self.parameters[parameter]["max"]
            values.append(uniform(min_val, max_val)) 
        return values

    def next_set_grid(self):
        """Generate parameter sets based on a grid of possible values for each parameter."""
        if self.parameter_grid is None:
            rospy.logerr("You have to give a parameter grid when calling ")
        values = []
        # get the next permutation of the grid
        i = 0
        for dimension in self.parameter_grid:
            values.append(dimension[self.grid_iteration_counters[i]])
            i += 1
        # set counters to next permutation
        i = 0
        for counter in self.grid_iteration_counters:
            # check if we went through all values of this parameter
            if counter + 1 == len(self.parameter_grid[i]):
                # reset counter and go to next dimension
                counter = 0
            else:
                # increase counter and ignore remaining dimensions
                counter += 1
                break
            i += 1
        return values

    def generate_grid(self):
        # parameter grid is a 2D matrix where rows are possible values for paramters
        self.parameter_grid =[]

        for i in range(0,len(self.parameter_name_order)):
            parameter = self.parameter_name_order[i]
            min_val = self.parameters[parameter]["min"]
            max_val = self.parameters[parameter]["max"]
            step_size = self.parameters[parameter]["step_size"]
            # generate possible values between min and max 
            grid_row = []
            val = min_val            
            while val <= max_val:
                grid_row.append(val)
                val += step_size
            self.parameter_grid.append(grid_row)        

if __name__ == "__main__":    
    TrainMaster()