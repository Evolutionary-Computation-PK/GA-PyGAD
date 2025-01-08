import numpy
import random


class Crossover:

    # TODO: Przykładowa implementacja do zmiany (możemy korzystać z atrybutów ga_instance)
    @staticmethod
    def arithmetic_crossover(parents, offspring_size, ga_instance):

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
    def mean_crossover(parents, offspring_size, ga_instance):
        pass

    @staticmethod
    def linear_crossover(parents, offspring_size, ga_instance):
        pass

    @staticmethod
    def alfa_blend_crossover(parents, offspring_size, ga_instance):
        pass

    @staticmethod
    def alfa_beta_blend_crossover(parents, offspring_size, ga_instance):
        pass
