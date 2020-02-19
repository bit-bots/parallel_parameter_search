#!/usr/bin/env python3

import rospy
import os

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify the number of the worker that you want to watch as argument.")
        exit(1)
    if len(sys.argv) > 2:
        ip = sys.argv[2]
    else:
        ip = "127.0.0.1"
    port = 11345 + int(sys.argv[1])
    print("You are now watching worker number " + sys.argv[1] + " on ip " + ip)
    os.system("GAZEBO_MASTER_URI="+ ip + ":" + str(port) + " rosrun gazebo_ros gzclient")
    print("\nThanks for watching :)")