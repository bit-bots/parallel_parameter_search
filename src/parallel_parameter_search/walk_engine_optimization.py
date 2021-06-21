import math

import rospy
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractWalkEngine(AbstractWalkOptimization):
    def __init__(self, namespace, gui, robot, walk_as_node, sim_type='pybullet', foot_link_names=(), start_speeds=None,
                 repetitions=1, multi_objective=False):
        super(AbstractWalkEngine, self).__init__(namespace, robot, walk_as_node)
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot, world="walk_optim_" + robot, ros_active=False)
        else:
            print(f'sim type {sim_type} not known')

        if not start_speeds:
            print("please set start speeds")
            exit(1)
        self.directions = [[start_speeds[0], 0, 0],
                           [0, start_speeds[1], 0],
                           [0, 0, start_speeds[2]],
                           [-start_speeds[0], -start_speeds[1], 0],
                           [-start_speeds[0], 0, -start_speeds[2]],
                           [start_speeds[0], start_speeds[1], start_speeds[2]]
                           ]
        self.repetitions = repetitions
        self.multi_objective = multi_objective

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        self.reset()

        d = 0
        it = 0
        pose_obj_rep_sum = 0
        orientation_obj_rep_sum = 0
        gyro_obj_rep_sum = 0
        # standing as first test, is not in loop as it will only be done once
        fall_sum, didnt_move_sum, pose_obj, orientation_obj, gyro_obj, end_poses = self.evaluate_direction(0, 0, 0, 1, 1)
        pose_obj_rep_sum += pose_obj
        orientation_obj_rep_sum += orientation_obj
        gyro_obj_rep_sum += gyro_obj

        if fall_sum:
            # terminate early and give 1 cost for each try left
            trial.set_user_attr('early_termination_at', (0, 0, 0))
            fall_sum = 1
        else:
            # iterate over the directions to increase the speed
            for iteration in range(1, self.number_of_iterations + 1):
                it += 1
                d = 0
                do_break = False
                for direction in self.directions:
                    d += 1
                    fall_rep_sum = 0
                    didnt_move_rep_sum = 0
                    # do multiple repetitions of the same values since behavior is not always exactly deterministic
                    for i in range(self.repetitions):
                        self.reset_position()
                        fall, didnt_move, pose_obj, orientation_obj, gyro_obj, end_poses = \
                            self.evaluate_direction(*direction, iteration, self.time_limit)
                        if fall:
                            fall_rep_sum += 1
                        if didnt_move:
                            didnt_move_rep_sum += 1
                        pose_obj_rep_sum += pose_obj
                        orientation_obj_rep_sum += orientation_obj
                        gyro_obj_rep_sum += gyro_obj
                        print(f"pose_obj {pose_obj}")
                        print(f"orientation_obj {orientation_obj}")
                        print(f"gyro obj {gyro_obj}")

                    # use the mean as costs for this try
                    fall_sum += fall_rep_sum

                    # check if we always failed in this direction and terminate this trial early
                    if fall_rep_sum == self.repetitions or didnt_move_rep_sum == self.repetitions:
                        if fall_rep_sum == self.repetitions:
                            trial.set_user_attr('termination_reason', 'fall')
                        else:
                            trial.set_user_attr('termination_reason', 'pose')
                        # terminate early and give 1 cost for each try left
                        # add extra information to trial
                        trial.set_user_attr('early_termination_at',
                                            (direction[0] * iteration, direction[1] * iteration,
                                             direction[2] * iteration))
                        do_break = True
                        break
                if do_break:
                    break
        if it == 0:
            performed_evaluations = 1
        else:
            performed_evaluations = (it - 1) * len(self.directions) + d
        print(f"performed evals {performed_evaluations}")
        last_pose_obj = pose_obj
        pose_obj = pose_obj_rep_sum / performed_evaluations
        orientation_obj = orientation_obj_rep_sum / performed_evaluations
        gyro_obj = gyro_obj_rep_sum / performed_evaluations

        stability_obj = (orientation_obj + gyro_obj) / 2
        print(f"fall_sum {fall_sum}")
        print(f"orientation_obj {orientation_obj}")
        print(f"gyro obj {gyro_obj}")
        print(f"stability_obj {stability_obj}")
        print(f"pose_obj {pose_obj}")

        # add costs based on the the iterations left
        directions_left = (self.number_of_iterations - it) * len(self.directions) + (len(self.directions) - d)
        if it == 0:
            # special case of falling while standing
            directions_left = self.number_of_iterations * len(self.directions)

        trial.set_user_attr("directions_left", directions_left)
        trial.set_user_attr("fall_sum", fall_sum)
        trial.set_user_attr("orientation_obj", orientation_obj)
        trial.set_user_attr("gyro_obj", gyro_obj)
        trial.set_user_attr("stability_obj", stability_obj)
        trial.set_user_attr("pose_obj", pose_obj)
        trial.set_user_attr("last_pose_obj", last_pose_obj)

        if self.multi_objective:
            return [directions_left, pose_obj]
        else:
            # multiple directions left with 10 to make sure the sum of the other objectives is never higher
            return directions_left + 0.5 * pose_obj

    def _suggest_walk_params(self, trial, trunk_height, foot_distance, foot_rise, trunk_x, z_movement):
        engine_param_dict = {}

        def add(name, min_value, max_value):
            engine_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            engine_param_dict[name] = value
            trial.set_user_attr(name, value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 0.5, 5)

        add('foot_distance', foot_distance[0], foot_distance[1])
        add('trunk_height', trunk_height[0], trunk_height[1])

        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_z_movement', 0, z_movement)

        add('trunk_x_offset', -trunk_x, trunk_x)
        add('trunk_y_offset', -trunk_x, trunk_x)

        add('trunk_pitch_p_coef_forward', -5, 5)
        add('trunk_pitch_p_coef_turn', -5, 5)

        add('trunk_pitch', -0.5, 0.5)
        # fix('trunk_pitch', 0.0)
        add('foot_rise', foot_rise[0], foot_rise[1])
        # fix('foot_rise', foot_rise)

        add('first_step_swing_factor', 0.0, 2)
        # fix('first_step_swing_factor', 1)
        fix('first_step_trunk_phase', -0.5)

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        fix('foot_overshoot_phase', 1)
        fix('foot_overshoot_ratio', 0.0)

        fix('foot_apex_phase', 0.5)

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        fix('foot_z_pause', 0)
        fix('foot_put_down_phase', 1)
        fix('trunk_pause', 0)

        node_param_dict = {}
        # walk engine should update at same speed as simulation
        node_param_dict["engine_freq"] = 1 / self.sim.get_timestep()
        # don't use loop closure when optimizing parameter
        node_param_dict["pressure_phase_reset_active"] = False
        node_param_dict["effort_phase_reset_active"] = False
        node_param_dict["phase_rest_active"] = False
        # make sure that steps are not limited
        node_param_dict["imu_active"] = False
        node_param_dict["max_step_x"] = 100.0
        node_param_dict["max_step_y"] = 100.0
        node_param_dict["max_step_xy"] = 100.0
        node_param_dict["max_step_z"] = 100.0
        node_param_dict["max_step_angular"] = 100.0
        node_param_dict["x_speed_multiplier"] = 1.0
        node_param_dict["y_speed_multiplier"] = 1.0
        node_param_dict["yaw_speed_multiplier"] = 1.0

        if self.walk_as_node:
            self.set_params(engine_param_dict)
            self.set_params(node_param_dict)
        else:
            self.current_params = engine_param_dict
            self.walk.set_engine_dyn_reconf(engine_param_dict)
            self.walk.set_node_dyn_reconf(node_param_dict)

        # necessary for correct reset
        self.trunk_height = self.current_params["trunk_height"]
        self.trunk_pitch = self.current_params["trunk_pitch"]
        self.trunk_pitch_p_coef_forward = self.current_params.get("trunk_pitch_p_coef_forward", 0)
        self.trunk_pitch_p_coef_turn = self.current_params.get("trunk_pitch_p_coef_turn", 0)


class WolfgangWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet', repetitions=1, multi_objective=False):
        super(WolfgangWalkEngine, self).__init__(namespace, gui, 'wolfgang', walk_as_node, sim_type,
                                                 start_speeds=[0.2, 0.05, 0.25], repetitions=repetitions,
                                                 multi_objective=multi_objective)
        self.reset_height_offset = 0.012

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.3, 0.45), foot_distance=(0.15, 0.25), foot_rise=(0.05, 0.2),
                                  trunk_x=0.1, z_movement=0.1)


class OP2WalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(OP2WalkEngine, self).__init__(namespace, gui, 'robotis_op2', walk_as_node, sim_type,
                                               foot_link_names=['MP_ANKLE2_L', 'MP_ANKLE2_R'],
                                               start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                                               multi_objective=multi_objective)
        self.reset_height_offset = 0.09

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.15, 0.25), (0.08, 0.16), (0.01, 0.15), 0.03, 0.05)


class OP3WalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(OP3WalkEngine, self).__init__(namespace, gui, 'op3', walk_as_node, sim_type,
                                            foot_link_names=['r_ank_roll_link', 'l_ank_roll_link'],
                                            start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                                            multi_objective=multi_objective)
        self.reset_height_offset = 0.01

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.15, 0.25), (0.08, 0.16), (0.01, 0.15), 0.03, 0.05)


class NaoWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(NaoWalkEngine, self).__init__(namespace, gui, 'nao', walk_as_node, sim_type,
                                            foot_link_names=['l_ankle', 'r_ankle'], start_speeds=[0.05, 0.025, 0.25],
                                            repetitions=repetitions, multi_objective=multi_objective)
        self.reset_height_offset = 0.01

        if sim_type == 'pybullet':
            self.sim.set_joints_dict(
                {"LShoulderPitch": 1.57, "RShoulderPitch": 1.57, 'LShoulderRoll': 0.3, 'RShoulderRoll': -0.3})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.25, 0.35), (0.1, 0.2), (0.01, 0.15), 0.03, 0.05)
