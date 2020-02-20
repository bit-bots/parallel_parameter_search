#!/usr/bin/env python3
import argparse

import rospy
import os
import rosservice
import subprocess

import time


def main():
    parser = argparse.ArgumentParser(description='Start workers')
    parser.add_argument('number_of_workers', type=int,
                        help='The number of workers to start')
    parser.add_argument('start_number', type=int, nargs='?', default=0,
                        help='The number on which the names of the workers are starting to count')
    parser.add_argument('master_uri', nargs='?', default="http://localhost:11311",
                        help='The uri of the ROS master')
    parser.add_argument('gazebo_starting_port', type=int, nargs='?', default="11345",
                        help='The base port were gazebo will be started. the port will be counted upwards for every '
                             'worker')
    args, unknown = parser.parse_known_args()
    number_of_workers = args.number_of_workers
    start_number = args.start_number
    gazebo_starting_port = args.gazebo_starting_port
    master_uri = args.master_uri

    rospy.init_node("worker_launcher")
    # check if master node is there
    try:
        service_list = rosservice.get_service_list()
    except rosservice.ROSServiceIOException:
        rospy.logerr("ROS core is not running!")
        exit(1)

    if "/request_parameters" in service_list:
        rospy.loginfo("Train master node is running. I can start my workers")
    else:
        rospy.logerr("Train master node is not running. Please make sure the Master URI is correct and the train "
                     "master node is running. Then start this script again.")
        exit(1)

    # start worker nodes
    """rospack = rospkg.RosPack()
    path = rospack.get_path("parallel_parameter_search")
    path = path + "/launch/worker.launch"
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)"""
    i = start_number
    while i < start_number + number_of_workers:
        rospy.logwarn("Starting worker with number %d", i)
        g_uri = "127.0.0.1:" + str(gazebo_starting_port + i)
        subprocess.Popen(
            "export ROS_MASTER_URI=" + master_uri + " && roslaunch parallel_parameter_search worker.launch number:=" + str(i) +
            " gazebo_uri:=" + g_uri, shell=True)

        # launch = roslaunch.parent.ROSLaunchParent(uuid, [path])
        # launch.start()
        i += 1
        # make a little break between launching, this makes output nicer and solves race conditions with ros controllers
        time.sleep(30)

    rospy.loginfo("Workers %d to %d were started successfully", start_number, start_number + number_of_workers -1)

    # wait till ros is shutdown
    while not rospy.is_shutdown():
        # use python time, since use_sim_time is true but /clock is not published
        time.sleep(1)
    print("ROS is shutdown, lets clean up")
    # call killing of gzserver just to be sure
    os.system("killall gzserver")
    #os.system("killall python")
    #os.system("killall /usr/bin/python")


if __name__ == "__main__":
    main()
