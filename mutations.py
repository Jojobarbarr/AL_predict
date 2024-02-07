from ctypes import ArgumentError
from scipy.stats import binom
import random as rd
import numpy as np

from genome import Genome


class Mutation:
    def __init__(self, rate: float, name: str, DEBUG: bool=False) -> None:
        self.rate = rate
        self.name = name
        self.DEBUG = DEBUG

        self.starting_point = 0
        self.length = 0
    
    def __str__(self) -> str:
        return self.name
    
    def is_neutral(self, genome: Genome):
        pass

    def apply(self, genome: Genome):
        pass



class PointMutation(Mutation): # DONE
    def __init__(self, rate: float, DEBUG: bool=False) -> None:
        super().__init__(rate, "Point Mutation", DEBUG)
    
    def is_neutral(self, genome: Genome):
        # We know the proportion of non coding sequences, therefore,
        # we conduct a Bernoulli trial with parameter p = genome.nc_proportion
        if rd.random() > genome.nc_proportion:
            if self.DEBUG:
                print(f"Deleterious starting point...")
            return False
        return True



class SmallInsertion(Mutation): # DONE
    def __init__(self, rate: float, l_m: int, DEBUG: bool=False) -> None:
        super().__init__(rate, "Small Insertion", DEBUG)
        self.l_m = l_m

    def is_neutral(self, genome: Genome):
        # We know the proportion of non coding sequences, therefore,
        # we conduct a Bernoulli trial with parameter p = genome.nc_proportion
        if rd.random() > genome.nc_proportion:
            if self.DEBUG:
                print(f"Deleterious starting point...")
            return False
        if self.DEBUG:
            print(f"Deleterious starting point...")
        return True

    def apply(self, genome: Genome):
        # We know that mutation is neutral, therefore in a non coding segment.
        # The exact starting_point in the non coding segment is useless, 
        # we only need to know which promoters loci will be affected.
        self.starting_point = rd.choice(genome.loci)
        self.length = rd.randint(1, self.l_m)

        if self.DEBUG:
            print(f"Neutral mutation:\n"
                  f"\tStarting point: {self.starting_point}\n"
                  f"\tLength: {self.length}")
            
        genome.insert(self.starting_point, self.length)



class Deletion(Mutation):
    def __init__(self, rate: float, is_small: bool=False, l_m: int=-1, DEBUG: bool=False) -> None:
        super().__init__(rate, f"{'Small ' if is_small else ''}Deletion", DEBUG)
        self.is_small = is_small
        if self.is_small and l_m == -1:
            raise ValueError(f"You must provide l_m is is_small is True.")
        self.l_m = l_m

    def is_neutral(self, genome: Genome):
        # We know the proportion of non coding sequences, therefore,
        # we perform a Bernoulli trial with parameter p = genome.nc_proportion
        if rd.random() > genome.nc_proportion:
            if self.DEBUG:
                print(f"Deleterious starting point...")
            return False
        
        # We know the maximum length a neutral deletion could have, therefore,
        # we check the length
        if self.is_small:
            self.length = rd.randint(1, self.l_m)
        else:
            self.length = rd.randint(1, genome.length)

        # # DEBUG
        # self.length = 5
        # # /DEBUG

        if self.length > genome.max_length_neutral:
            if self.DEBUG:
                print(f"Deleterious length ({self.length})...")
            return False
        
        # To get a random starting point, we draw a random number that represent 
        # the absolute position in non coding genome.
        # We check the following promoter, and get the absolute position in whole genome that is:
        # absolute_position_in_non_coding + promoter_numbers_passed * gene_length
        self.starting_point = rd.randint(0, genome.z_nc - 1)
        
        # # DEBUG
        # self.starting_point = 12
        # # /DEBUG

        next_promoter_locus_index = genome.binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index * genome.gene_length

        # If the starting point is between the last promoter and ORI, then the binary research gives
        # len(genome.loci) as a result. The next promoter is then the first one. 
        if next_promoter_locus_index == len(genome.loci):
            next_promoter_locus = genome.length + genome.loci[0]
        else:
            next_promoter_locus = genome.loci[next_promoter_locus_index]

        # We know the beginning locus and the length of the deletion,
        # we can check if the end point is neutral.
        if self.length > next_promoter_locus - self.starting_point:
            if self.DEBUG:
                print(f"Deleterious ending point (starting point: {self.starting_point}, length: {self.length})...")
            return False
        return True
    
    def apply(self, genome: Genome):
        if self.DEBUG:
            print(f"Neutral mutation:\n"
                  f"\tStarting point: {self.starting_point}\n"
                  f"\tLength: {self.length}\n")
        # If deletion is between the last promoter and the first, we need to proceed with two steps:
        # - Deletion from starting point to ORI
        # - Deletion from ORI to first promoter
        # without deleting more than self.length
        if self.starting_point > genome.loci[-1]:
            end_deletion_length = min(genome.length - self.starting_point, self.length)
            genome.delete(genome.loci[-1], end_deletion_length)
            genome.delete(0, self.length - end_deletion_length)
        else:
            genome.delete(self.starting_point, self.length)





if __name__ == "__main__":
    # np.random.seed(42)
    # rd.seed(42)

    DEBUG = True

    g = 1000
    z_c = 1000 * g
    z_nc = 2000 * g

    genome = Genome(z_c, z_nc, g, DEBUG=DEBUG)

    point_mutation_rate = 10e-9
    small_insertion_rate = 10e-9
    small_deletion_rate = 10e-9
    dupplication_rate = 10e-9
    deletion_rate = 10e-9
    inversion_rate = 10e-9

    point_mutation = PointMutation(point_mutation_rate, DEBUG=DEBUG)

    if point_mutation.is_neutral(genome):
        point_mutation.apply(genome)
    
    small_insertion = SmallInsertion(small_insertion_rate, 6, DEBUG=DEBUG)

    print(genome)

    if small_insertion.is_neutral(genome):
        small_insertion.apply(genome)

    print(genome)

    deletion = Deletion(deletion_rate, DEBUG=DEBUG)

    if deletion.is_neutral(genome):
        deletion.apply(genome)

    print(genome)

    small_deletion = Deletion(small_deletion_rate, is_small=True, l_m=6, DEBUG=DEBUG)

    if small_deletion.is_neutral(genome):
        small_deletion.apply(genome)
    
    print(genome)


    

