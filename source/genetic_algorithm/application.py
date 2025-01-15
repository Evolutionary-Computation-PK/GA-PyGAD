# na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import pygad
import numpy
import benchmark_functions as bf
from opfunu.cec_based.cec2014 import F132014

from source.genetic_algorithm.binary.strategies import CrossoverStrategyEnum as BinaryCrossoverStrategyEnum
from source.genetic_algorithm.binary.strategies import MutationStrategyEnum as BinaryMutationStrategyEnum
from source.genetic_algorithm.real.strategies import CrossoverStrategyEnum as RealCrossoverStrategyEnum
from source.genetic_algorithm.real.strategies import MutationStrategyEnum as RealMutationStrategyEnum
from source.genetic_algorithm.selection_strategies import SelectionStrategyEnum
from source.utils.binary_utils import BinaryUtils

# Konfiguracja algorytmu genetycznego

num_genes_Rosenbrock = 2
num_genes_Happycat = 10
func_Rosenbrock = bf.Rosenbrock(n_dimensions=num_genes_Rosenbrock)
func_Happycat = F132014(ndim=num_genes_Happycat).evaluate

Rosenbrock = {
    # binary or real
    "chosen_ga_type": "real",

    "num_dim": num_genes_Rosenbrock,
    "function": func_Rosenbrock,
    "start_interval": -2.048,
    "end_interval": 2.048,
    "precision_binary": 4,

    "num_generations": 100,
    "sol_per_pop": 80,
    "num_parents_mating": 50,

    "parent_selection_type": SelectionStrategyEnum.TOURNAMENT.value,
    "keep_elitism": 1,
    "K_tournament": 3,

    "crossover_type_real": RealCrossoverStrategyEnum.ARITHMETIC,
    "crossover_type_binary": BinaryCrossoverStrategyEnum.ONE_POINT,
    "crossover_probability": 0.8,

    "mutation_type_real": RealMutationStrategyEnum.GAUSSIAN,
    "mutation_type_binary": BinaryMutationStrategyEnum.RANDOM,
    "mutation_probability": 0.2
}

Happycat = {
    # binary or real
    "chosen_ga_type": "real",

    "num_dim": num_genes_Happycat,
    "function": func_Happycat,
    "start_interval": -100,
    "end_interval": 100,
    "precision_binary": 4,

    "num_generations": 100,
    "sol_per_pop": 80,
    "num_parents_mating": 50,

    "parent_selection_type": SelectionStrategyEnum.TOURNAMENT.value,
    "keep_elitism": 1,
    "K_tournament": 3,

    "crossover_type_binary": BinaryCrossoverStrategyEnum.ONE_POINT,
    "crossover_type_real": RealCrossoverStrategyEnum.ARITHMETIC,
    "crossover_probability": 0.8,

    "mutation_type_real": RealMutationStrategyEnum.GAUSSIAN,
    "mutation_type_binary": BinaryMutationStrategyEnum.RANDOM,
    "mutation_probability": 0.2
}

chosen_func_config = Happycat
fitness_batch_size = 10

if chosen_func_config["chosen_ga_type"] == "binary":
    chosen_func_config["gene_type"] = int
    start_interval = chosen_func_config["start_interval"]
    end_interval = chosen_func_config["end_interval"]
    chosen_func_config["init_range_low"] = 0
    chosen_func_config["init_range_high"] = 2
    number_of_bits_for_gene = BinaryUtils.get_binary_length(start_interval, end_interval,
                                                            chosen_func_config["precision_binary"])
    chosen_func_config["num_genes"] = number_of_bits_for_gene * chosen_func_config["num_dim"]


    def fitness_function_binary(ga_instance, solution, solution_idx):
        decoded_individual = BinaryUtils.decode_individual(solution, start_interval, end_interval,
                                                           chosen_func_config["num_dim"])
        return 1. / chosen_func_config["function"](decoded_individual)


    chosen_func_config["fitness_func"] = fitness_function_binary

    chosen_func_config["crossover_type"] = chosen_func_config["crossover_type_binary"]
    chosen_func_config["mutation_type"] = chosen_func_config["mutation_type_binary"]

elif chosen_func_config["chosen_ga_type"] == "real":
    chosen_func_config["gene_type"] = float
    chosen_func_config["num_genes"] = chosen_func_config["num_dim"]
    chosen_func_config["init_range_low"] = chosen_func_config["start_interval"]
    chosen_func_config["init_range_high"] = chosen_func_config["end_interval"]


    def fitness_function_real(ga_instance, solution, solution_idx):
        return 1. / chosen_func_config["function"](solution)


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


def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    ga_instance.logger.info("Best    = {fitness}".format(fitness=1. / solution_fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

    tmp = [1. / x for x in ga_instance.last_generation_fitness]  # ponownie odwrotność by zrobić sobie dobre statystyki

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")


# Właściwy algorytm genetyczny
if __name__ == "__main__":
    ga_instance = pygad.GA(num_generations=chosen_func_config["num_generations"],
                           sol_per_pop=chosen_func_config["sol_per_pop"],
                           num_parents_mating=chosen_func_config["num_parents_mating"],
                           num_genes=chosen_func_config["num_genes"],
                           fitness_func=chosen_func_config["fitness_func"],
                           init_range_low=chosen_func_config["init_range_low"],
                           init_range_high=chosen_func_config["init_range_high"],
                           gene_type=chosen_func_config["gene_type"],

                           # mutation_num_genes=mutation_num_genes,
                           parent_selection_type=chosen_func_config["parent_selection_type"],
                           keep_elitism=chosen_func_config["keep_elitism"],
                           K_tournament=chosen_func_config["K_tournament"],
                           crossover_type=chosen_func_config["crossover_type"],
                           crossover_probability=chosen_func_config["crossover_probability"],
                           mutation_type=chosen_func_config["mutation_type"],
                           mutation_probability=chosen_func_config["mutation_probability"],
                           save_best_solutions=True,
                           save_solutions=True,
                           logger=logger,
                           on_generation=on_generation,
                           parallel_processing=['thread', 4])

    ga_instance.run()

    best = ga_instance.best_solution()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))

    # sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
    ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
    ga_instance.plot_fitness()
