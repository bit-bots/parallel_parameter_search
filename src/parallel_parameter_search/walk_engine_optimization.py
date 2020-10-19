from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractWalkEngine(AbstractWalkOptimization):
    def __init__(self, namespace, gui, robot, walk_as_node, sim_type='pybullet', foot_link_names=()):
        super(AbstractWalkEngine, self).__init__(namespace, robot, walk_as_node)
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui)
        else:
            print(f'sim type {sim_type} not known')

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        self.reset()

        cost = 0
        # standing as first test, is not in loop as it will only be done once
        early_term, cost_try = self.evaluate_direction(0, 0, 0, trial, 1, 0)
        cost += cost_try
        if early_term:
            # terminate early and give 1 cost for each try left
            return 1 * (self.number_of_iterations - 1) * len(self.directions) + 1 * len(self.directions) + cost

        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for direction in self.directions:
                d += 1
                self.reset_position()
                early_term, cost_try = self.evaluate_direction(*direction, trial, iteration, self.time_limit)
                cost += cost_try
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 1 cost for each try left
                    return 1 * (self.number_of_iterations - iteration) * len(self.directions) + 1 * (
                            len(self.directions) - d) + cost
        return cost

    def _suggest_walk_params(self, trial, trunk_height, foot_distance, foot_rise, trunk_x, z_movement):
        engine_param_dict = {}

        def add(name, min_value, max_value):
            engine_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            engine_param_dict[name] = value
            trial.set_user_attr(name, value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 0.5, 3)
        add('foot_distance', foot_distance[0], foot_distance[1])
        add('trunk_height', trunk_height[0], trunk_height[1])
        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_z_movement', 0, z_movement)

        add('trunk_x_offset', -trunk_x, trunk_x)
        add('trunk_y_offset', -0.03, 0.03)

        add('trunk_x_offset_p_coef_forward', -1, 1)
        add('trunk_x_offset_p_coef_turn', -1, 1)

        add('trunk_pitch', -0.5, 0.5)

        fix('foot_rise', foot_rise)


        # add('first_step_swing_factor', 0.0, 2)
        # add('first_step_trunk_phase', -0.5, 0.5)
        fix('first_step_swing_factor', 1)
        fix('first_step_trunk_phase', -0.5)

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        fix('foot_overshoot_phase', 1)
        fix('foot_overshoot_ratio', 0.0)

        # add('foot_rise', 0.04, 0.08)
        # add('foot_apex_phase', 0.0, 1.0)
        fix('foot_apex_phase', 0.5)

        # add('trunk_pitch_p_coef_forward', -5, 5)
        # add('trunk_pitch_p_coef_turn', -5, 5)
        fix('trunk_pitch_p_coef_forward', 0)
        fix('trunk_pitch_p_coef_turn', 0)

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
        self.trunk_pitch_p_coef_forward = self.current_params["trunk_pitch_p_coef_forward"]
        self.trunk_pitch_p_coef_turn = self.current_params["trunk_pitch_p_coef_turn"]


class WolfgangWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(WolfgangWalkEngine, self).__init__(namespace, gui, 'wolfgang', walk_as_node, sim_type)
        self.reset_height_offset = 0.005
        self.directions = [[0.1, 0, 0],
                           [-0.1, 0, 0],
                           [0, 0.05, 0],
                           [0, -0.05, 0],
                           [0, 0, 0.5],
                           [0, 0, -0.5],
                           [0.1, 0, 0.5],
                           [0, 0.05, -0.5],
                           [0.1, 0.05, 0.5],
                           [-0.1, -0.05, -0.5]
                           ]

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.38, 0.42), (0.15, 0.25), 0.1, 0.03, 0.03)


class DarwinWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(DarwinWalkEngine, self).__init__(namespace, gui, 'darwin', walk_as_node, sim_type,
                                               foot_link_names=['MP_ANKLE2_L', 'MP_ANKLE2_R'])
        self.reset_height_offset = 0.09
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.20, 0.24), (0.08, 0.15), 0.02, 0.02, 0.02)


class OP3WalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(OP3WalkEngine, self).__init__(namespace, gui, 'op3', walk_as_node, sim_type,
                                            foot_link_names=['r_ank_roll_link', 'l_ank_roll_link'])
        self.reset_height_offset = 0.01
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            self.sim.set_joints_dict({"l_sho_roll": 1.20, "r_sho_roll": -1.20})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.13, 0.24), (0.08, 0.15), 0.02, 0.02, 0.02)


class NaoWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(NaoWalkEngine, self).__init__(namespace, gui, 'nao', walk_as_node, sim_type,
                                            foot_link_names=['l_ankle', 'r_ankle'])
        self.reset_height_offset = 0.01
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            self.sim.set_joints_dict(
                {"LShoulderPitch": 1.57, "RShoulderPitch": 1.57, 'LShoulderRoll': 0.3, 'RShoulderRoll': -0.3})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.27, 0.32), (0.1, 0.17), 0.03, 0.02, 0.02)


class ReemcWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(ReemcWalkEngine, self).__init__(namespace, gui, 'reemc', walk_as_node, sim_type,
                                              foot_link_names=['leg_left_6_link', 'leg_right_6_link'])
        self.reset_height_offset = -0.1
        self.reset_rpy_offset = (-0.1, 0.15, -0.5)
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.5, 0.7), (0.15, 0.30), 0.1, 0.1, 0.05)


class TalosWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(TalosWalkEngine, self).__init__(namespace, gui, 'talos', walk_as_node, sim_type,
                                              foot_link_names=['leg_left_6_link', 'leg_right_6_link'])
        self.reset_height_offset = -0.13
        self.reset_rpy_offset = (0, 0.15, 0)
        self.directions = [[0.05, 0, 0],
                           [-0.05, 0, 0],
                           [0, 0.025, 0],
                           [0, -0.025, 0],
                           [0, 0, 0.25],
                           [0, 0, -0.25],
                           [0.05, 0.25, 0],
                           [0.05, -0.25, 0],
                           [0.05, 0, -0.25],
                           [-0.05, 0, 0.25],
                           ]
        if sim_type == 'pybullet':
            self.sim.set_joints_dict({"arm_left_4_joint": -1.57, "arm_right_4_joint": -1.57})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.6, 0.8), (0.15, 0.4), 0.1, 0.15, 0.08)
