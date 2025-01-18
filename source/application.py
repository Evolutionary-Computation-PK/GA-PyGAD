# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import time

import numpy as np
import pandas as pd
import pygad
import numpy
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F132014
import matplotlib.pyplot as plt

from source.genetic_algorithm.binary.strategies import CrossoverStrategyEnum as BinaryCrossoverStrategyEnum
from source.genetic_algorithm.binary.strategies import MutationStrategyEnum as BinaryMutationStrategyEnum
from source.genetic_algorithm.real.strategies import CrossoverStrategyEnum as RealCrossoverStrategyEnum
from source.genetic_algorithm.real.strategies import MutationStrategyEnum as RealMutationStrategyEnum
from source.genetic_algorithm.selection_strategies import SelectionStrategyEnum
from source.utils.binary_utils import BinaryUtils

import os
import multiprocessing

# Liczba logicznych rdzeni
logical_cores = os.cpu_count()
print(f"Liczba logicznych rdzeni CPU: {logical_cores}")

# Konfiguracja algorytmu genetycznego

num_genes_Rosenbrock = 2
num_genes_Happycat = 10
func_Rosenbrock = bf.Rosenbrock(n_dimensions=num_genes_Rosenbrock)
func_Happycat = F132014(ndim=num_genes_Happycat).evaluate

Rosenbrock = {
    # binary or real
    "chosen_ga_type": "binary",

    "num_dim": num_genes_Rosenbrock,
    "function": func_Rosenbrock,
    "start_interval": -2.048,
    "end_interval": 2.048,
    "precision_binary": 3,
    # "num_parents_mating": 50,

    # To Modify
    "num_generations": 100,
    "sol_per_pop": 80,

    "parent_selection_type": SelectionStrategyEnum.ROULETTE.value,
    "keep_elitism": 1,
    "K_tournament": 3,

    "crossover_type_real": RealCrossoverStrategyEnum.ARITHMETIC,
    "crossover_type_binary": BinaryCrossoverStrategyEnum.UNIFORM.value,
    "crossover_probability": 0.8,

    "mutation_type_real": RealMutationStrategyEnum.GAUSSIAN,
    "mutation_type_binary": BinaryMutationStrategyEnum.SWAP.value,
    "mutation_probability": 0.2
}

Happycat = {
    # binary or real
    "chosen_ga_type": "binary",

    "num_dim": num_genes_Happycat,
    "function": func_Happycat,
    "start_interval": -100,
    "end_interval": 100,
    "precision_binary": 3,
    # "num_parents_mating": 50,

    # To Modify
    "num_generations": 300,
    "sol_per_pop": 350,

    "parent_selection_type": SelectionStrategyEnum.TOURNAMENT.value,
    "keep_elitism": 10,
    "K_tournament": 3,

    "crossover_type_real": RealCrossoverStrategyEnum.ALFA_BETA_BLEND,
    "crossover_type_binary": BinaryCrossoverStrategyEnum.UNIFORM.value,
    "crossover_probability": 0.8,

    "mutation_type_real": RealMutationStrategyEnum.GAUSSIAN,
    "mutation_type_binary": BinaryMutationStrategyEnum.RANDOM.value,
    "mutation_probability": 0.1
}

chosen_func_config = Happycat
chosen_func_config["num_parents_mating"] = int(chosen_func_config["sol_per_pop"] / 2)
fitness_batch_size = 10

if chosen_func_config["chosen_ga_type"] == "binary":
    chosen_func_config["gene_type"] = int
    start_interval = chosen_func_config["start_interval"]
    end_interval = chosen_func_config["end_interval"]
    chosen_func_config["init_range_low"] = 0
    chosen_func_config["init_range_high"] = 2
    chosen_func_config["gene_space"] = [0, 1]
    number_of_bits_for_gene = BinaryUtils.get_binary_length(start_interval, end_interval,
                                                            chosen_func_config["precision_binary"])
    chosen_func_config["num_genes"] = number_of_bits_for_gene * chosen_func_config["num_dim"]


    def fitness_function_binary(ga_instance, solution, solution_idx):
        decoded_individual = BinaryUtils.decode_individual(solution, start_interval, end_interval,
                                                           chosen_func_config["num_dim"])
        return 1. / (chosen_func_config["function"](decoded_individual) + 1e-10)


    chosen_func_config["fitness_func"] = fitness_function_binary

    chosen_func_config["crossover_type"] = chosen_func_config["crossover_type_binary"]
    chosen_func_config["mutation_type"] = chosen_func_config["mutation_type_binary"]

elif chosen_func_config["chosen_ga_type"] == "real":
    chosen_func_config["gene_type"] = float
    chosen_func_config["num_genes"] = chosen_func_config["num_dim"]
    chosen_func_config["init_range_low"] = chosen_func_config["start_interval"]
    chosen_func_config["init_range_high"] = chosen_func_config["end_interval"]
    chosen_func_config["gene_space"] = None


    def fitness_function_real(ga_instance, solution, solution_idx):
        return 1. / (chosen_func_config["function"](solution) + 1e-10)


    chosen_func_config["fitness_func"] = fitness_function_real

    chosen_func_config["crossover_type"] = chosen_func_config["crossover_type_real"]
    chosen_func_config["mutation_type"] = chosen_func_config["mutation_type_real"]
else:
    raise ValueError("Invalid GA type")

# Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

average_fitness_per_generation = []
stddev_fitness_per_generation = []
best_fitness_per_generation = []

def callback_generation(ga_instance):
    generation_fitness = ga_instance.last_generation_fitness
    inverted_fitness = np.array([1. / x for x in generation_fitness])

    min_fitness = np.min(inverted_fitness)
    max_fitness = np.max(inverted_fitness)
    average_fitness = np.mean(inverted_fitness)
    stddev_fitness = np.std(inverted_fitness)

    average_fitness_per_generation.append(average_fitness)
    stddev_fitness_per_generation.append(stddev_fitness)
    best_fitness_per_generation.append(np.min(inverted_fitness))

    print(f"Generation {ga_instance.generations_completed}: "
          f"Min Fitness = {min_fitness:.6f}, "
          f"Max Fitness = {max_fitness:.6f}, "
          f"Average Fitness = {average_fitness:.6f}, "
          f"Std Dev Fitness = {stddev_fitness:.6f}")


# Właściwy algorytm genetyczny
if __name__ == "__main__":
    number_of_trials = 10
    current_best_solution_fitness = 0
    all_best_solutions_fitness = []
    all_solutions_fitness_from_best_run = []
    best_solution_parameters = None
    best_solution_index = 0
    all_times = []

    for trial in range(number_of_trials):
        ga_instance = pygad.GA(num_generations=chosen_func_config["num_generations"],
                               sol_per_pop=chosen_func_config["sol_per_pop"],
                               num_parents_mating=chosen_func_config["num_parents_mating"],
                               num_genes=chosen_func_config["num_genes"],
                               fitness_func=chosen_func_config["fitness_func"],
                               init_range_low=chosen_func_config["init_range_low"],
                               init_range_high=chosen_func_config["init_range_high"],
                               gene_type=chosen_func_config["gene_type"],
                               gene_space=chosen_func_config["gene_space"],

                               # mutation_num_genes=mutation_num_genes,
                               parent_selection_type=chosen_func_config["parent_selection_type"],
                               keep_elitism=chosen_func_config["keep_elitism"],
                               K_tournament=chosen_func_config["K_tournament"],
                               crossover_type=chosen_func_config["crossover_type"],
                               crossover_probability=chosen_func_config["crossover_probability"],
                               mutation_type=chosen_func_config["mutation_type"],
                               mutation_probability=chosen_func_config["mutation_probability"],
                               save_best_solutions=True,
                               # save_solutions=True,
                               logger=logger,
                               on_generation= callback_generation,
                               parallel_processing=['thread', 10])

        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        exec_time = end_time - start_time
        all_times.append(exec_time)
        ga_instance.logger.info("Execution Time = {execution_time}".format(execution_time=exec_time))

        best = ga_instance.best_solution()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))

        # sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
        ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
        all_best_solutions_fitness.append(ga_instance.best_solutions_fitness)
        # ga_instance.plot_fitness()

        if current_best_solution_fitness < solution_fitness:
            current_best_solution_fitness = solution_fitness
            #     all_solutions_fitness_from_best_run = ga_instance.solutions_fitness
            best_solution_parameters = solution
            best_solution_index = trial

        generations = range(len(average_fitness_per_generation))

        plt.figure(figsize=(8, 6))
        plt.plot(generations, average_fitness_per_generation, marker='o', color='blue')
        plt.title("Wykres średniej wartości funkcji fitness w każdej epoce")
        plt.xlabel("Epoka")
        plt.ylabel("Średnia wartość funkcji fitness")
        plt.grid()
        plt.savefig("graph/average_fitness_plot"+str(trial)+".png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(generations, stddev_fitness_per_generation, marker='s', color='orange')
        plt.title("Wykres odchylenia standardowego w poszczególnych epokach")
        plt.xlabel("Epoka")
        plt.ylabel("Odchylenie standardowe")
        plt.grid()
        plt.savefig("graph/stddev_fitness_plot"+str(trial)+".png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(generations, best_fitness_per_generation, marker='^', color='green')
        plt.title("Najlepsza wartość funkcji fitness w każdej epoce")
        plt.xlabel("Epoka")
        plt.ylabel("Najlepsza wartość funkcji fitness")
        plt.grid()
        plt.savefig("graph/best_fitness_plot"+str(trial)+".png")
        plt.close()
        average_fitness_per_generation = []
        stddev_fitness_per_generation = []
        best_fitness_per_generation = []

    logger.info("Best solution parameters in best trial = {best_solution_parameters}".format(
        best_solution_parameters=best_solution_parameters))
    decoded_solution = BinaryUtils.decode_individual(best_solution_parameters,
                                                     chosen_func_config["start_interval"],
                                                     chosen_func_config["end_interval"],
                                                     chosen_func_config["num_dim"])
    logger.info("Best solution decoded parameters in best trial = {best_solution_parameters}".format(
        best_solution_parameters=decoded_solution))

    logger.info("Best solution fitness in best trial = {best_solution_fitness}".format(
        best_solution_fitness=1. / current_best_solution_fitness))
    logger.info("Min fitness = {min_fitness}".format(min_fitness=numpy.min(all_best_solutions_fitness)))
    logger.info("Max fitness = {max_fitness}".format(max_fitness=numpy.max(all_best_solutions_fitness)))
    logger.info("Average fitness = {average_fitness}".format(average_fitness=numpy.average(all_best_solutions_fitness)))
    logger.info("Average time = {average_time}".format(average_time=numpy.average(all_times)))
    logger.info("Best index ={best_solution_index}".format(best_solution_index=best_solution_index))

    # all_solutions_fitness_from_best_run = np.array(all_solutions_fitness_from_best_run)
    # all_solutions_fitness_from_best_run = 1. / all_solutions_fitness_from_best_run
    # all_solutions_fitness_from_best_run = np.split(all_solutions_fitness_from_best_run,
    #                                                chosen_func_config["num_generations"] + 1)
    # df = pd.DataFrame(all_solutions_fitness_from_best_run)
    # df.to_csv('all_solutions_fitness_from_best_run.csv', index=False, header=False)
