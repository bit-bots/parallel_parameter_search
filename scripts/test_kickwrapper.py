#!/usr/bin/env python
import rospkg
import rospy
from bitbots_msgs.msg import KickGoal, JointCommand
from geometry_msgs.msg import Quaternion
from parallel_parameter_search.simulators import WebotsSim
from bitbots_dynamic_kick import PyKick
from tf.transformations import quaternion_from_euler

from parallel_parameter_search.utils import load_robot_param, load_yaml_to_param

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    rospack = rospkg.RosPack()
    load_robot_param('', rospack, 'wolfgang')

    # load walk params
    load_yaml_to_param('', 'bitbots_dynamic_kick',
                       '/config/kick_config.yaml', rospack)

    load_yaml_to_param("/robot_description_kinematics", 'wolfgang_moveit_config',
                       '/config/kinematics.yaml', rospack)
    sim = WebotsSim('', True)
    kick = PyKick()
    params = {
        'foot_rise': 0.08,
        'foot_distance': 0.18,
        'kick_windup_distance': 0.2,
        'trunk_height': 0.4,
        'trunk_roll': 0,
        'trunk_pitch': 0,
        'trunk_yaw': 0,
        'move_trunk_time': 0.5,
        'raise_foot_time': 0.3,
        'move_to_ball_time': 0.1,
        'kick_time': 0.1,
        'move_back_time': 0.2,
        'lower_foot_time': 0.1,
        'move_trunk_back_time': 0.2,
        'choose_foot_corridor_width': 0.4,
        'use_center_of_pressure': False,
        'stabilizing_point_x': 0.0,
        'stabilizing_point_y': 0.015,
    }
    kick.set_params(params)
    kick_finished = False
    msg = KickGoal()
    msg.header.stamp = rospy.Time(sim.get_time())
    msg.header.frame_id = "base_footprint"
    msg.kick_direction = Quaternion(*quaternion_from_euler(0, 0, 0))
    msg.kick_speed = 1
    msg.ball_position.x = 0.2
    msg.ball_position.y = 0
    kick.set_goal(msg, sim.get_joint_state_msg())
    print(f'goal: {msg}')
    last_time = sim.get_time()
    while not kick_finished:
        # test if the robot has fallen down, then return maximal cost
        current_time = sim.get_time()
        joint_command = kick.step(current_time - last_time,
                                  sim.get_joint_state_msg())  # type: JointCommand
        if len(joint_command.joint_names) == 0:
            kick_finished = True
        else:
            sim.set_joints(joint_command)
            last_time = current_time
            sim.step_sim()
