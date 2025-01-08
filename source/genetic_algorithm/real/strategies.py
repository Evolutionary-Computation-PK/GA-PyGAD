from enum import Enum

from source.genetic_algorithm.real.crossover import Crossover
from source.genetic_algorithm.real.mutation import Mutation


class CrossoverStrategyEnum(Enum):
    ARITHMETIC = Crossover.arithmetic_crossover
    MEAN = Crossover.mean_crossover
    LINEAR = Crossover.linear_crossover
    ALFA_BLEND = Crossover.alfa_blend_crossover
    ALFA_BETA_BLEND = Crossover.alfa_beta_blend_crossover


class MutationStrategyEnum(Enum):
    GAUSSIAN = Mutation.gaussian_mutation
    UNIFORM = Mutation.uniform_mutation
