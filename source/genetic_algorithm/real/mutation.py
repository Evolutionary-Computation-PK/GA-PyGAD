import numpy as np
import random
import pygad


class Mutation:
    @staticmethod
    def gaussian_mutation(offspring : np.ndarray, ga_instance : pygad.GA) -> np.ndarray:
        """
        Mutate offspring with gaussian mutation

        :param offspring: an array offspring to mutate.
        :param ga_instance: instance of PyGAD.GA class
        :return: an array of the mutated offspring.
        """
        if ga_instance.mutation_probability is None:
            return offspring

        mutation_probability = ga_instance.mutation_probability
        a = ga_instance.init_range_low
        b = ga_instance.init_range_high

        gene_idx = np.random.choice(range(offspring.shape[1]))
        for chromosome_index in range(offspring.shape[0]):
            if random.random() <= mutation_probability:
                n = np.random.normal(0, 1)
                offspring[chromosome_index, gene_idx] += n

                if ga_instance.gene_space is not None:
                    if isinstance(ga_instance.gene_space, list):
                        gene_space = ga_instance.gene_space[gene_idx]
                    else:
                        gene_space = ga_instance.gene_space

                    if isinstance(gene_space, dict) and "low" in gene_space and "high" in gene_space:
                        a = gene_space["low"]
                        b = gene_space["high"]

                offspring[chromosome_index, gene_idx] = np.clip(offspring[chromosome_index, gene_idx], a, b)

        return offspring


    @staticmethod
    def uniform_mutation(offspring, ga_instance):
        """
        Mutate offspring with uniform mutation
        :param offspring: array of the offspring to be mutated.
        :param ga_instance: instance of PyGAD.GA class
        :return: array of the mutated offspring.
        """
        if ga_instance.mutation_probability is None:
            return offspring

        mutation_probability = ga_instance.mutation_probability
        a = ga_instance.init_range_low
        b = ga_instance.init_range_high

        gene_idx = np.random.choice(range(offspring.shape[1]))
        for chromosome_index in range(offspring.shape[0]):
            if random.random() <= mutation_probability:
                offspring[chromosome_index, gene_idx] = random.uniform(a, b)

        return offspring