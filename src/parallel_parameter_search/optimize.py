import argparse
import importlib

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

from src.parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Dataabase SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--objective', help='Name of the objective class. Has to be of type AbstractRosOptimization.',
                    default=None, type=str, required=True)
# todo kwargs for optuna
args = parser.parse_args()

importlib.import_module(args.objective)
if not issubclass(args.objective, AbstractRosOptimization):
    print('Objective class is not a subclass of AbstractRosOptimization.')
    exit(1)

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = 10

sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
#pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=10)

study = optuna.create_study(study_name=args.name, storage=args.storage, direction='maximize',
                            sampler=sampler, load_if_exists=True)

objective = args.objective()
study.optimize(objective.objective, n_trials=100, show_progress_bar=True)

objective.close()
