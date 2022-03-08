#!/usr/bin/env python3

import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, MOTPESampler, RandomSampler, NSGAIISampler
from optuna.integration import WeightsAndBiasesCallback
import numpy as np

from parallel_parameter_search.walk_engine_optimization import OP2WalkEngine, WolfgangWalkEngine, OP3WalkEngine, \
    NaoWalkEngine

from parallel_parameter_search.walk_stabilization import WolfgangWalkStabilization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, op2, op3, nao} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--type', help='Optimization type that should be used {engine, stabilization} ', default=None,
                    type=str, required=True)
parser.add_argument('--startup', help='Startup trials', default=-1,
                    type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {MOTPE, TPE, CMAES, NSGA2, Random}', default=10000,
                    type=str, required=True)
parser.add_argument('--repetitions', help='How often each trial is repeated while beeing evaluated', default=1,
                    type=int, required=False)
parser.add_argument('--suggest', help='Suggest a working solution', action='store_true')

args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

multi_objective = args.sampler in ['MOTPE', 'Random']

if args.type == "engine":
    if args.robot == "op2":
        objective = OP2WalkEngine(gui=args.gui, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "wolfgang":
        objective = WolfgangWalkEngine(gui=args.gui, sim_type=args.sim,
                                       repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "op3":
        objective = OP3WalkEngine(gui=args.gui, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    elif args.robot == "nao":
        objective = NaoWalkEngine(gui=args.gui, sim_type=args.sim,
                                  repetitions=args.repetitions, multi_objective=multi_objective)
    else:
        print(f"robot type \"{args.robot}\" not known.")
elif args.type == "stabilization":
    if args.robot == "wolfgang":
        objective = WolfgangWalkStabilization(gui=args.gui, sim_type=args.sim,
                                              repetitions=args.repetitions, multi_objective=multi_objective)
        # add one trial without stabilitation at the beginning to provide a baseline
        study.enqueue_trial(
            {"pitch.p": 0.0, "pitch.i": 0.0, "pitch.d": 0.0, "pitch.i_clamp_min": 0.0, "pitch.i_clamp_max": 0.0,
             "roll.p": 0.0, "roll.i": 0.0, "roll.d": 0.0, "roll.i_clamp_min": 0.0, "roll.i_clamp_max": 0.0,
             "pause_duration": 0.0, "imu_pitch_threshold": 0.0, "imu_roll_threshold": 0.0,
             "imu_pitch_vel_threshold": 0.0, "imu_roll_vel_threshold": 0.0})
    else:
        print(f"robot type \"{args.robot}\" not known.")
else:
    print(f"Optimization type {args.type} not known.")

num_variables = len(objective.directions)

if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False, constant_liar=True)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
elif args.sampler == "MOTPE":
    if n_startup_trials == -1:
        n_startup_trials = num_variables * 11 - 1
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=False, constant_liar=True)
elif args.sampler == "NSGA2":
    sampler = NSGAIISampler(seed=seed)
elif args.sampler == "Random":
    sampler = RandomSampler(seed=seed)
else:
    print("sampler not correctly specified. Should be one of {TPE, CMAES, MOTPE, NSGA2, Random}")
    exit(1)

if multi_objective:
    study = optuna.create_study(study_name=args.name, storage=args.storage,
                                directions=["maximize"] * num_variables,# + ["minimize"] * num_variables,
                                sampler=sampler, load_if_exists=True)
else:
    study = optuna.create_study(study_name=args.name, storage=args.storage, direction="maximize",
                                sampler=sampler, load_if_exists=True)

study.set_user_attr("sampler", args.sampler)
study.set_user_attr("robot", args.robot)
study.set_user_attr("type", args.type)
study.set_user_attr("repetitions", args.repetitions)

wandb_kwargs = {
    "project": f"optuna-walk-{args.type}",
    "tags": [args.sampler, args.robot, args.sim],
    "resume": "never",
    "group": args.name,  # use group so that we can run multiple studies in parallel
}

wandbc = WeightsAndBiasesCallback(
    metric_name=["objective.forward", "objective.backward", "objective.left", "objective.turn"],
                 #"objective.error_forward", "objective.error_backward", "objective.error_left", "objective.error_turn"],
    wandb_kwargs=wandb_kwargs)

if args.suggest:
    if args.type == "engine":
        if len(study.get_trials()) == 0:
            # old params
            print("#############\nUSING GIVEN PARAMETERS\n#############")
            for i in range(1):
                study.enqueue_trial(
                    {"engine.double_support_ratio": 0.187041787093062,
                     "engine.first_step_swing_factor": 0.988265815486162,
                     "engine.foot_distance": 0.191986968311401, "engine.foot_rise": 0.0805917174531535,
                     "engine.freq": 2.81068228309542, "engine.trunk_height": 0.364281403417376,
                     "engine.trunk_phase": -0.19951206583248, "engine.trunk_pitch": 0.338845862625267,
                     "engine.trunk_pitch_p_coef_forward": -1.36707568402799,
                     "engine.trunk_pitch_p_coef_turn": -0.621298812652778, "engine.trunk_swing": 0.342345300382608,
                     "engine.trunk_x_offset": -0.0178414805249525, "engine.trunk_y_offset": 0.000997552190718013,
                     "engine.trunk_z_movement": 0.0318583647276103, "engine.early_termination_at": [0.0, 0.0, 35.0],
                     "engine.first_step_trunk_phase": -0.5, "engine.foot_apex_phase": 0.5,
                     "engine.foot_overshoot_phase": 1.0, "engine.foot_overshoot_ratio": 0.0,
                     "engine.foot_put_down_phase": 1.0, "engine.foot_z_pause": 0.0, "engine.trunk_pause": 0.0})
    else:
        print("no suggestion specified for this type")
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=True, callbacks=[wandbc])

# close simulator window
objective.sim.close()
