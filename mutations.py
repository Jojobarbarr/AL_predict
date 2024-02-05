from scipy.stats import binom
import numpy as np


class Mutations:
    def __init__(self, rate) -> None:
        self.rate = rate

class PointMutations(Mutations):
    def __init__(self, rate) -> None:
        super().__init__(rate)
    
    def is_neutral(self):
        pass

class SmallInsertion(Mutations):
    def __init__(self, rate) -> None:
        super().__init__(rate)
    
    def is_neutral(self):
        pass



if __name__ == '__main__':
    np.random.seed(42)
    
    genome_size = int(10e8)

    point_mutation_rate = 10e-9
    small_insertion_rate = 10e-9
    small_deletion_rate = 10e-9
    dupplication_rate = 10e-9
    deletion_rate = 10e-9
    inversion_rate = 10e-9
    
    point_mutation = PointMutations(point_mutation_rate)
    small_insertion = SmallInsertion(small_insertion_rate)
    mutations = [point_mutation, small_insertion]

    total_mutation_rate = 0
    for mutation in mutations:
        total_mutation_rate += mutation.rate

    mutation_number = binom.rvs(genome_size, total_mutation_rate)
    print(f"mutation_number: {mutation_number}")
