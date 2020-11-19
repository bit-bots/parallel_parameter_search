#!/usr/bin/env python3

import rospy

# this is just a small node which is used to reserve namespaces
rospy.init_node('dummy_node', anonymous=False)

while not rospy.is_shutdown():
    rospy.spin()