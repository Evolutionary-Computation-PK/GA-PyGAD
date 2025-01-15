from warnings import catch_warnings

import numpy
import random

import numpy as np


class Crossover:

    # TODO: Przykładowa implementacja do zmiany (możemy korzystać z atrybutów ga_instance)
    @staticmethod
    def example_crossover(parents, offspring_size, ga_instance):

        """
        Applies single-point crossover between pairs of parents.
        This function selects a random point at which crossover occurs between the parents, generating offspring.

        Parameters:
            parents (array-like): The parents to mate for producing the offspring.
            offspring_size (int): The number of offspring to produce.

        Returns:
            array-like: An array containing the produced offspring.
        """

        if ga_instance.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the K points at which crossover takes place between each two parents. The point does not have to be always at the center of the solutions.
        # This saves time by calling the numpy.random.randint() function only once.
        crossover_points = numpy.random.randint(low=0,
                                                high=parents.shape[1],
                                                size=offspring_size[0])

        for k in range(offspring_size[0]):
            # Check if the crossover_probability parameter is used.
            if not (ga_instance.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = list(set(numpy.where(probs <= ga_instance.crossover_probability)[0]))

                # If no parent satisfied the probability, no crossover is applied and a parent is selected as is.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(indices, 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k + 1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_points[k]] = parents[parent1_idx, 0:crossover_points[k]]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_points[k]:] = parents[parent2_idx, crossover_points[k]:]

            if ga_instance.allow_duplicate_genes == False:
                if ga_instance.gene_space is None:
                    offspring[k], _, _ = ga_instance.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                                    min_val=ga_instance.random_mutation_min_val,
                                                                                    max_val=ga_instance.random_mutation_max_val,
                                                                                    mutation_by_replacement=ga_instance.mutation_by_replacement,
                                                                                    gene_type=ga_instance.gene_type,
                                                                                    num_trials=10)
                else:
                    offspring[k], _, _ = ga_instance.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                                    gene_type=ga_instance.gene_type,
                                                                                    num_trials=10)

        return offspring

    @staticmethod
    def arithmetic_crossover(parents, offspring_size, ga_instance):
        offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        k = offspring_size[0]

        while k > 0:

            try:
                individual1_indx, individual2_indx = Crossover.generate_individuals(parents, offspring, ga_instance, k)
            except Exception:
                continue

            new_chromosome1, new_chromosome2 = zip(*[
                Crossover.generate_new_genes_arithmetic(parents[individual1_indx][j],
                                                        parents[individual2_indx][j],
                                                        ga_instance.init_range_low,
                                                        ga_instance.init_range_high)
                for j in range(parents.shape[1])
            ])

            offspring[k - 1] = new_chromosome1
            offspring[k - 2] = new_chromosome2
            k = k - 2

        return offspring

    @staticmethod
    def generate_new_genes_arithmetic(gene1, gene2, start_interval, end_interval):
        isValid = False
        gene_individual1 = 0
        gene_individual2 = 0

        while not isValid:
            alpha = numpy.random.rand()
            gene_individual1 = alpha * gene1 + (1 - alpha) * gene2
            gene_individual2 = alpha * gene2 + (1 - alpha) * gene1
            isValid = Crossover.validate_gene_in_interval(gene_individual1, start_interval,
                                                          end_interval) and Crossover.validate_gene_in_interval(
                gene_individual2, start_interval, end_interval)

        return gene_individual1, gene_individual2

    @staticmethod
    def validate_gene_in_interval(gene, start_interval, end_interval):
        return start_interval <= gene <= end_interval

    @staticmethod
    def generate_individuals(parents, offspring, ga_instance, k):
        if not (ga_instance.crossover_probability is None):
            probs = numpy.random.random(size=parents.shape[0])
            indices = list(set(numpy.where(probs <= ga_instance.crossover_probability)[0]))

            # If no parent satisfied the probability, no crossover is applied and a parent is selected as is.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                raise Exception
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(indices, 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
        return parent1_idx, parent2_idx

    @staticmethod
    def mean_crossover(parents, offspring_size, ga_instance):
        offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        k = offspring_size[0]

        while k > 0:

            try:
                individual1_indx, individual2_indx = Crossover.generate_individuals(parents, offspring, ga_instance, k)
            except Exception:
                continue

            new_individual = (parents[individual1_indx] + parents[individual2_indx]) / 2

            offspring[k - 1] = new_individual
            k = k - 1

        return offspring

    @staticmethod
    def linear_crossover(parents, offspring_size, ga_instance):
        offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        k = offspring_size[0]

        while k > 0:
            try:
                individual1_indx, individual2_indx = Crossover.generate_individuals(parents, offspring, ga_instance, k)
            except Exception:
                continue

            z_chromosomes = [Crossover.generate_gene_linear(0.5, 0.5, gene1, gene2, ga_instance.init_range_low,
                                                            ga_instance.init_range_high) for gene1, gene2
                             in zip(parents[individual1_indx], parents[individual2_indx])]

            v_chromosomes = [Crossover.generate_gene_linear(1.5, -0.5, gene1, gene2, ga_instance.init_range_low,
                                                            ga_instance.init_range_high) for gene1, gene2
                             in zip(parents[individual1_indx], parents[individual2_indx])]

            w_chromosomes = [Crossover.generate_gene_linear(-0.5, 1.5, gene1, gene2, ga_instance.init_range_low,
                                                            ga_instance.init_range_high) for gene1, gene2 in
                             zip(parents[individual1_indx], parents[individual2_indx])]

            fitness_values = [
                (ga_instance.fitness_func(ga_instance, z_chromosomes, None), z_chromosomes),
                (ga_instance.fitness_func(ga_instance, v_chromosomes, None), v_chromosomes),
                (ga_instance.fitness_func(ga_instance, w_chromosomes, None), w_chromosomes),
            ]

            sorted_chromosomes = sorted(fitness_values, key=lambda x: x[0], reverse=True)

            new_individuals = [chromosome for _, chromosome in sorted_chromosomes[:2]]

            offspring[k - 1] = new_individuals[0]
            offspring[k - 2] = new_individuals[1]
            k = k - 2

        return offspring

    @staticmethod
    def generate_gene_linear(coeff1, coeff2, gene1, gene2, start_interval, end_interval):
        result = coeff1 * gene1 + coeff2 * gene2
        if result < start_interval:
            return start_interval
        elif result > end_interval:
            return end_interval
        return result

    @staticmethod
    def get_gene_from_interval(start_interval, end_interval):
        """ Returns a random gene from the interval [start_interval, end_interval]. """
        gene = np.random.uniform(start_interval, end_interval)
        return gene

    @staticmethod
    def alfa_blend_cross_genes(chromosome1, chromosome2,
                               gene_min_value: float, gene_max_value: float,
                               offspring, actual_offspring_count_left) -> None:
        for idx, (gene1, gene2) in enumerate(zip(chromosome1, chromosome2)):
            alfa = np.random.rand()
            min_gene, max_gene = sorted((gene1, gene2))
            d = max_gene - min_gene
            interval_start = min_gene - alfa * d
            interval_end = max_gene + alfa * d
            if interval_start < gene_min_value:
                interval_start = gene_min_value
            if interval_end > gene_max_value:
                interval_end = gene_max_value
            offspring[actual_offspring_count_left - 1, idx] = Crossover.get_gene_from_interval(interval_start,
                                                                                               interval_end)
            offspring[actual_offspring_count_left - 2, idx] = Crossover.get_gene_from_interval(interval_start,
                                                                                               interval_end)

    @staticmethod
    def alfa_blend_crossover(parents, offspring_size, ga_instance):
        offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        offspring_count = offspring_size[0]

        while offspring_count > 0:
            try:
                individual1_indx, individual2_indx = Crossover.generate_individuals(parents, offspring, ga_instance,
                                                                                    offspring_count)
            except Exception:
                continue

            Crossover.alfa_blend_cross_genes(parents[individual1_indx], parents[individual2_indx],
                                             ga_instance.init_range_low, ga_instance.init_range_high, offspring,
                                             offspring_count)
            offspring_count = offspring_count - 2
        return offspring

    @staticmethod
    def alfa_beta_blend_cross_genes(chromosome1, chromosome2,
                                    gene_min_value: float, gene_max_value: float,
                                    offspring, actual_offspring_count_left) -> None:
        for idx, (gene1, gene2) in enumerate(zip(chromosome1, chromosome2)):
            alfa = np.random.rand()
            beta = np.random.rand()
            min_gene, max_gene = sorted((gene1, gene2))
            d = max_gene - min_gene
            interval_start = min_gene - alfa * d
            interval_end = max_gene + beta * d
            if interval_start < gene_min_value:
                interval_start = gene_min_value
            if interval_end > gene_max_value:
                interval_end = gene_max_value
            offspring[actual_offspring_count_left - 1, idx] = Crossover.get_gene_from_interval(interval_start,
                                                                                               interval_end)
            offspring[actual_offspring_count_left - 2, idx] = Crossover.get_gene_from_interval(interval_start,
                                                                                               interval_end)

    @staticmethod
    def alfa_beta_blend_crossover(parents, offspring_size, ga_instance):
        offspring = numpy.empty(offspring_size, dtype=ga_instance.gene_type[0])
        offspring_count = offspring_size[0]

        while offspring_count > 0:
            try:
                individual1_indx, individual2_indx = Crossover.generate_individuals(parents, offspring, ga_instance,
                                                                                    offspring_count)
            except Exception:
                continue

            Crossover.alfa_beta_blend_cross_genes(parents[individual1_indx], parents[individual2_indx],
                                                  ga_instance.init_range_low, ga_instance.init_range_high,
                                                  offspring, offspring_count)
            offspring_count = offspring_count - 2
        return offspring
