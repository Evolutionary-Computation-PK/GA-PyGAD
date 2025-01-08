import os

import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import MedianPruner
from functools import partial
import hydra
from hydra.utils import instantiate

from source.config.logging_config import setup_logging
from source.genetic_algorithm.binary.strategies import CrossoverStrategyEnum as BinaryCrossoverStrategyEnum
from source.genetic_algorithm.binary.strategies import MutationStrategyEnum as BinaryMutationStrategyEnum
from source.genetic_algorithm.real.strategies import CrossoverStrategyEnum as RealCrossoverStrategyEnum
from source.genetic_algorithm.real.strategies import MutationStrategyEnum as RealMutationStrategyEnum
from source.genetic_algorithm.selection_strategies import SelectionStrategyEnum

logger = setup_logging()


def objective(trial: optuna.trial.Trial, function: callable, num_variables: int, start_interval: float,
              end_interval: float, search_minimum: bool, precision: int, param_ranges: dict,
              num_evolutions: int = 10) -> float:
    """
    Objective function for Optuna study.

    :param param_ranges: Dictionary with the ranges of the parameters to optimize.
    :param trial: Optuna trial object.
    :param function: Fitness function to optimize.
    :param num_variables: Number of variables in the optimization problem.
    :param start_interval: Start of the interval for variable values.
    :param end_interval: End of the interval for variable values.
    :param search_minimum: Boolean indicating if the search is for a minimum.
    :param precision: Precision of the variable values.
    :param num_evolutions: Number of evolutions to run and average the results.
    :return: Combined score of the optimization.
    """
    trial.set_user_attr('num_variables', num_variables)
    trial.set_user_attr('precision', precision)

    population_size = trial.suggest_int('population_size', *param_ranges["population_size"])
    num_generations = trial.suggest_int('num_generations', *param_ranges["num_generations"])

    selection_strategy_type = trial.suggest_categorical('selection_strategy', param_ranges['selection_strategy'])
    # `num_parents_mating` i PyGAD ?
    selection_size = trial.suggest_int('selection_size', *param_ranges["selection_size"]) if selection_strategy_type != 'TOURNAMENT' else None
    tournament_size = trial.suggest_int('tournament_size', *param_ranges[
        "tournament_size"]) if selection_strategy_type == 'TOURNAMENT' else None
    selection_strategy = SelectionStrategyEnum[selection_strategy_type].value(
        selection_size if selection_size else tournament_size)

    elite_strategy = trial.suggest_categorical('elite_strategy', param_ranges['elite_strategy'])
    elite_size = trial.suggest_int('elite_size', *param_ranges['elite_size']) if elite_strategy else 0

    crossover_strategy_type = trial.suggest_categorical('crossover_strategy', param_ranges['crossover_strategy'])
    crossover_probability = trial.suggest_float('crossover_probability', *param_ranges['crossover_probability'])

    # crossover_size = population_size - elite_size
    crossover_strategy = BinaryCrossoverStrategyEnum[crossover_strategy_type]

    mutation_strategy_type = trial.suggest_categorical('mutation_strategy', param_ranges['mutation_strategy'])
    mutation_probability = trial.suggest_float('mutation_probability', *param_ranges['mutation_probability'])
    mutation_strategy = BinaryMutationStrategyEnum[mutation_strategy_type]

    # inversion = None
    # is_inversion = trial.suggest_categorical('inversion', param_ranges['inversion'])
    # if is_inversion:
    #     inversion_probability = trial.suggest_float('inversion_probability', *param_ranges['inversion_probability'])
    #     inversion = Inversion(inversion_probability)

    logger.debug(f"Evolution initialized with the following parameters:\n"
                 f"num_variables: {num_variables}\n"
                 f"precision: {precision}\n"
                 f"population_size: {population_size}\n"
                 f"number_of_generations: {num_generations}\n"
                 f"selection_strategy: {selection_strategy_type}\n"
                 f"selection_size: {selection_size}\n"
                 f"tournament_size: {tournament_size}\n"
                 f"elite_strategy: {elite_strategy}\n"
                 f"elite_size: {elite_size}\n"
                 f"crossover_strategy: {crossover_strategy_type}\n"
                 f"crossover_probability: {crossover_probability}\n"
                 # f"intersection_number: {intersection_number}\n"
                 f"mutation_strategy: {mutation_strategy_type}\n"
                 f"mutation_probability: {mutation_probability}\n")
                 # f"inversion: {is_inversion}\n")
                 # f"inversion_probability: {inversion_probability if is_inversion else "None"}")

    computation_times = []
    best_individuals_values = []
    fitness_values = []

    # Number of evolution to run and average the results
    #TODO: Implement evolution

    # return combined_score
    # return float(avg_fitness_value)


def enqueue_running_and_failed_trials(study: optuna.study.Study) -> None:
    """
    Enqueue running and failed trials in the Optuna study.

    :param study: Optuna study object.
    """
    trials = study.get_trials(deepcopy=False)
    # Set to store unique sample parameters
    enqueued_params = set()

    for trial in trials:
        if trial.state in [optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL]:
            trial_params = tuple(trial.params.items())  # Replace the parameter dictionary with a tuple

            # Check that the parameters have not already been added to the queue
            if trial_params not in enqueued_params:
                logger.info(f"Enqueuing trial with parameters: {trial.params}")
                study.enqueue_trial(trial.params)
                enqueued_params.add(trial_params)


def run_study(database_url: str, function: callable, function_name: str, num_variable: int,
              start_interval: float, end_interval: float, param_ranges: dict, evolutions_number: int = 10,
              optuna_n_trials: int = 30) -> None:
    """
    Run an Optuna study for a given function.

    :param database_url: URL of the database to store the study results.
    :param function: Fitness function to optimize.
    :param function_name: Name of the fitness function.
    :param num_variable: Number of variables in the optimization problem.
    :param start_interval: Start of the interval for variable values.
    :param end_interval: End of the interval for variable values.
    :param param_ranges: Dictionary with the ranges of the parameters to optimize.
    :param evolutions_number: Number of evolutions to run and average the results.
    :param optuna_n_trials: Number of trials to run in the Optuna study.
    """
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(
        study_name=f'{function_name}__var_{num_variable}__maximize__with_time_v2',
        direction='maximize',
        storage=database_url,
        load_if_exists=True,
        pruner=pruner
    )

    logger.info(f"Optimizing {function_name} with {num_variable} variables")
    # enqueue_running_and_failed_trials(study)

    partial_objective = partial(objective, function=function, num_variables=num_variable,
                                start_interval=start_interval, end_interval=end_interval,
                                search_minimum=True, param_ranges=param_ranges,
                                num_evolutions=evolutions_number)
    study.optimize(partial_objective, n_trials=optuna_n_trials)

    logger.info(
        f"Best parameters for {function_name}, num variables {num_variable}: {study.best_params}")


@hydra.main(version_base='1.3', config_path="source/config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.DB_DIR, exist_ok=True)
    database_url = f"{cfg.DB_PREFIX}{cfg.DB_DIR}/{cfg.DB_FILE}"

    config = cfg.functions
    function_name = config.function_name
    function_instance = instantiate(config.function)
    if config.evaluate == "evaluate":
        evaluate_function = function_instance.evaluate
    else:
        evaluate_function = function_instance

    run_study(database_url, evaluate_function, function_name, config.num_variables,
              config.start_interval, config.end_interval, config.param_ranges,
              evolutions_number=config.evolutions_number, optuna_n_trials=config.optuna_n_trials)


if __name__ == '__main__':
    main()
