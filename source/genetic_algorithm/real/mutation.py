import numpy
import random


class Mutation:

    # TODO: Przykładowa implementacja do zmiany (możemy korzystać z atrybutów ga_instance)
    @staticmethod
    def gaussian_mutation(offspring, ga_instance):

        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if ga_instance.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (ga_instance.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = ga_instance.mutation_by_space(offspring)
            else:
                offspring = ga_instance.mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (ga_instance.gene_space is None):
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = ga_instance.mutation_probs_by_space(offspring)
            else:
                offspring = ga_instance.mutation_probs_randomly(offspring)

        return offspring

    @staticmethod
    def uniform_mutation(offspring, ga_instance):
        pass
