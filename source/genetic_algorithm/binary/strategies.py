from enum import Enum


class CrossoverStrategyEnum(Enum):
    ONE_POINT = "single_point"
    TWO_POINT = "two_points"
    UNIFORM = "uniform"


class MutationStrategyEnum(Enum):
    RANDOM = "random"
    SWAP = "swap"
