import dynamic_reconfigure.client
import rosparam
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim


class AbstractWalkStabilization(AbstractWalkOptimization):
    def __init__(self, namespace, gui, walk_as_node, robot, sim_type='pybullet', foot_link_names=(),
                 start_terrain_height=0.01):
        super(AbstractWalkStabilization, self).__init__(namespace, robot, walk_as_node)
        self.start_terrain_height = start_terrain_height

        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names, terrain=True)
        elif sim_type == 'webots':
            print("Webots currently does not support stabilization optimization")
        else:
            print(f'sim type {sim_type} not known')

        # needed to reset robot pose correctly
        self.trunk_height = rosparam.get_param(self.namespace + "/walking/engine/trunk_height")
        self.trunk_pitch = rosparam.get_param(self.namespace + "/walking/engine/trunk_pitch")

        # self.walk.spin_ros()
        self.foot_x_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/pid_foot_pos_x/',
                                                               timeout=60)
        self.foot_y_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/pid_foot_pos_y/',
                                                               timeout=60)
        self.hip_pitch_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/pid_hip_pitch/',
                                                                  timeout=60)
        self.hip_roll_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/pid_hip_roll/',
                                                                 timeout=60)


    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        # reset simulation with initial terrain height
        self.sim.randomize_terrain(self.start_terrain_height)
        self.reset()
        # reset walk controller
        self.walk.reset()

        cost = 0
        # standing as first test, is not in loop as it will only be done once
        early_term, cost_try = self.evaluate_direction(0, 0, 0, trial, 1, 0)
        cost += cost_try
        if early_term:
            # terminate early and give 100 cost for each try left
            return 100 * (self.number_of_iterations - 1) * len(self.directions) + 100 * len(self.directions) + cost

        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for direction in self.directions:
                d += 1
                self.reset_position()
                early_term, cost_try = self.evaluate_direction(*direction, trial, iteration, self.time_limit)
                cost += cost_try
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 100 cost for each try left
                    return 100 * (self.number_of_iterations - iteration) * len(self.directions) + 100 * (
                            len(self.directions) - d) + cost
            # increase difficulty of terrain
            self.randomize_terrain(self.start_terrain_height * iteration)
        return cost

    def _suggest_walk_params(self, trial, pressure_reset, effort_reset, phase_rest, stability_stop, foot_pid,
                             hip_pid):
        # optimal engine parameters are already loaded from yaml
        node_param_dict = {}

        def add(name, min_value, max_value):
            node_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            node_param_dict[name] = value
            trial.set_user_attr(name, value)

        # walk engine should update at same speed as simulation
        fix("engine_freq", 1 / self.sim.get_timestep())
        # make sure that steps are not limited
        fix("max_step_x", 100.0)
        fix("max_step_y", 100.0)
        fix("max_step_xy", 100.0)
        fix("max_step_z", 100.0)
        fix("max_step_angular", 100.0)
        fix("debug_active", True)
        # activate features depending on what is optimized
        fix("pressure_phase_reset_active", pressure_reset)
        if pressure_reset:
            add("ground_min_pressure", 0, 5)
        fix("effort_phase_reset_active", effort_reset)
        fix("phase_rest_active", phase_rest)

        fix("imu_active", stability_stop)
        if stability_stop:
            add("pause_duration", 0.0, 2.0)
            add("imu_pitch_threshold", 0.0, 1.0)
            add("imu_roll_threshold", 0.0, 1.0)
            add("imu_pitch_vel_threshold", 0.0, 10.0)
            add("imu_roll_vel_threshold", 0.0, 10.0)

        def pid_params(name, client, p, i, d, i_clamp):
            pid_dict = {"p": trial.suggest_uniform(name + "_p", p[0], p[1]),
                        "d": trial.suggest_uniform(name + "_d", i[0], i[1]),
                        "i": trial.suggest_uniform(name + "_i", d[0], d[1]),
                        "i_clamp_min": i_clamp[0],
                        "i_clamp_max": i_clamp[1]}
            self.set_params(pid_dict, client, self.walk)

        if foot_pid:
            pid_params("foot_x", self.foot_x_client, (-10, 10), (-1, 1), (-0.1, 0.1), (-10, 10))
            pid_params("foot_y", self.foot_y_client, (-10, 10), (-1, 1), (-0.1, 0.1), (-10, 10))

        if hip_pid:
            pid_params("hip_pitch", self.hip_pitch_client, (-10, 10), (-1, 1), (-0.1, 0.1), (-10, 10))
            pid_params("hip_roll", self.hip_roll_client, (-10, 10), (-1, 1), (-0.1, 0.1), (-10, 10))

        # ankle

        # (cop)

        # trunk fused

        # trunk rpy

        # trunk gyro


        if self.walk_as_node:
            self.set_params(node_param_dict)
        else:
            self.current_params = node_param_dict
            self.walk.set_node_dyn_reconf(node_param_dict)


class WolfgangWalkStabilization(AbstractWalkStabilization):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet'):
        super(WolfgangWalkStabilization, self).__init__(namespace, gui, walk_as_node, "wolfgang", sim_type)
        self.reset_height_offset = 0.005
        self.start_terrain_height = 0.01
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
        self._suggest_walk_params(trial, False, False, False, False, True)
