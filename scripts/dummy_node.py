#!/usr/bin/env python3

import rospy

rospy.init_node('dummy_node', anonymous=False)

while not rospy.is_shutdown():
    rospy.spin()