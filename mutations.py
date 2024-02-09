import random as rd
from typing import Callable

from stats import MutationStatistics
from genome import Genome


class Mutation:
    def __init__(self, rate: float, type: str, genome: Genome, DEBUG: bool=False) -> None:
        """

        Args:
            rate (float): rate of occurence.
            type (str): mutation type.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        self.rate = rate
        self.type = type
        self.genome = genome
        self.stats = MutationStatistics()
        self.DEBUG = DEBUG

        self.insertion_locus = 0
        self.starting_point = 0
        self.length = 0

        self.is_small = False
        self.l_m = 0
    
    def __str__(self) -> str:
        """Print the type of the mutation.

        Returns:
            str: Type of the mutation.
        """
        return self.type
    
    def is_neutral(self):
        """If this method is called, it means a mutation occurs. The mutation counter is incremented.
        """
        self.stats.count += 1

    def apply(self, virtually: bool=False):
        """If this method is called, it means a mutation occurs and is neutral. The neutral mutation counter is incremented as well as the length sum.
        
        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        self.stats.neutral_count += 1
        self.stats.length_sum += self.length

    def Bernoulli(self, p: float) -> bool:
        """Perform a Bernoulli trial with parameter p, 0 <= p <= 1.

        Args:
            p (float): probability of success.

        Raises:
            ValueError: if p isn't in [0, 1].

        Returns:
            bool: the result of the Bernoulli trial.
        """
        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1. You provided p = {p}")
        
        if rd.random() < p:
            return True
        
        return False
    
    def set_length(self):
        """Set a random length for the mutation. If the mutation is a small one, the length is between 1 and self.l_m included.
        Otherwise, the length is between 1 ans self.genome.length.
        """
        if self.is_small:
            self.length = rd.randint(1, self.l_m)
        else:
            self.length = rd.randint(1, self.genome.length)
    
    def set_insertion_locus(self, upper_bound: int):
        """Choose a random insertion locus.

        Args:
            upper_bound (int): the upper bound of the random range.
        """
        self.insertion_locus = rd.randint(0, upper_bound) # both end points are included with rd.randint
    
    def map_local_to_absolute_locus(self, mapping: Callable[[int], int]):
        """Map local absolute position to the absolute position in genome. This mapping depends on the mutation type and is provided in argument 'mapping'.

        Args:
            mapping (Callable[[int], int]): mapping that depends on the mutation type.
        """
        insertion_locus_index = mapping(self.insertion_locus)
        if insertion_locus_index == len(self.genome.loci):
            insertion_locus_index = 0
        self.insertion_locus = self.genome.loci[insertion_locus_index]
    
    def ending_point_is_ok(self, next_promoter_locus_index: int) -> bool:
        """Checks if ending point is neutral. Ending point is neutral if length is less than distance between next promoter and starting point.

        Args:
            next_promoter_locus_index (int): next promoter locus index that allows to get next promoter absolute position.

        Returns:
            bool: Return False if ending point is deleterious.
        """
        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(self.genome.loci):
            # In circular genome case, first promoter locus is self.genome.loci[0] (mod self.genome.length).
            next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            next_promoter_locus = self.genome.loci[next_promoter_locus_index]

        if self.length <= next_promoter_locus - self.starting_point:
            return True
        
        return False



class PointMutation(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int=-1, DEBUG: bool=False) -> None:
        """

        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.. Defaults to False.
        """
        super().__init__(rate, "Point Mutation", genome, DEBUG)
    
    def is_neutral(self) -> bool:
        """Check if mutation is neutral. Point mutation is neutral if it affects a non coding base.
        Therefore, we conduct a Bernoulli trial with parameter p = self.genome.nc_proportion.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        return self.Bernoulli(self.genome.nc_proportion)



class SmallInsertion(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int, DEBUG: bool=False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            l_m (int): maximum length of the mutation.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, "Small Insertion", genome, DEBUG)
        self.is_small = True
        self.l_m = l_m

    def is_neutral(self) -> bool:
        """Check if mutation is neutral. Samll insertion is neutral if it inserts between two non coding bases, and right after a coding sequence
        Therefore, we conduct a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        return self.Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length) or self.DEBUG
    
    def apply(self, virtually: bool=False):
        """Apply the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply()
        self.set_insertion_locus(self.genome.z_nc + self.genome.g - 1)
        
        self.map_local_to_absolute_locus(self.genome.insertion_binary_search)

        self.set_length()

        if self.DEBUG:
            self.insertion_locus = 4
            self.length = 3
        
            print(f"Neutral", self,
                  f"\n\tStarting point: {self.insertion_locus}\n"
                  f"\tLength: {self.length}")
        if not virtually:
            self.genome.insert(self.insertion_locus, self.length)


class Deletion(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int=-1, DEBUG: bool=False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, f"{'Small ' if self.is_small else ''}Deletion", genome, DEBUG)

    def length_is_ok(self) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= self.genome.max_length_neutral:
            return True
        
        if self.DEBUG:
            print(f"Deleterious length ({self.length})...")
        return False
    
    def set_starting_point(self) -> int:
        """Sets deletion starting point. The starting poisition is a random base position in non coding genome. This position in non coding genome is then mapped
        to the absolute position in whole genome.

        Returns:
            int: the index of the next promoter locus.
        """
        self.starting_point = rd.randint(0, self.genome.z_nc - 1)
        next_promoter_locus_index = self.genome.deletion_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index * self.genome.gene_length
        return next_promoter_locus_index

    def is_neutral(self) -> bool:
        """Checks if mutation is neutral. Deletion is neutral if starting point is a non coding base AND length is less than distance to the next coding section.
        This method first checks if starting point is a deleterious locus by conducting a Bernoulli trial with parameter p = self.genome.z_nc_porportion.
        Then, checks if length is greater than the maximum non coding sequence.
        Then, checks if the ending point is deleterious.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        if not self.Bernoulli(self.genome.nc_proportion) and not self.DEBUG:
            return False
        
        self.set_length()
        
        if not self.length_is_ok():
            return False

        return self.ending_point_is_ok(self.set_starting_point()) or self.DEBUG
    
    def apply(self, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply()
        if self.DEBUG:
            self.starting_point = 15
            self.length = 4
            print(f"Neutral", self,
                  f"\n\tStarting point: {self.starting_point}\n"
                  f"\tLength: {self.length}\n")
            
        if not virtually:
            # If deletion is between the last promoter and the first, we need to proceed with two steps:
            # - Deletion from starting point to ORI
            # - Deletion from ORI to first promoter
            # without deleting more than self.length
            if self.starting_point > self.genome.loci[-1]:
                end_deletion_length = min(self.genome.length - self.starting_point, self.length)
                self.genome.delete(self.genome.loci[-1], end_deletion_length)
                self.genome.delete(0, self.length - end_deletion_length)
            else:
                self.genome.delete(self.starting_point, self.length)


class SmallDeletion(Deletion):
    def __init__(self, rate: float, genome: Genome, l_m: int, DEBUG: bool = False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            l_m (int): maximum length of the mutation.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        self.is_small = True
        super().__init__(rate, genome, l_m, DEBUG)
        self.l_m = l_m


class Duplication(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int=-1, DEBUG: bool = False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, "Duplication", genome, DEBUG)
    
    def set_starting_point(self) -> int:
        """Sets duplication starting point. The starting poisition is a random base position in genome as long as it is not a promoter. 
        This position in genome minus promoters is then mapped to the absolute position in whole genome.

        Returns:
            int: the index of the next promoter locus.
        """
        self.starting_point = rd.randint(0, self.genome.length - len(self.genome.loci) - 1)
        next_promoter_locus_index = self.genome.deletion_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index
        return next_promoter_locus_index


    def length_is_ok(self) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= self.genome.max_length_neutral + self.genome.gene_length - 1:
            return True
        
        if self.DEBUG:
            print(f"Deleterious length ({self.length})...")
        return False

    def is_neutral(self) -> bool:
        """Checks if mutation is neutral. Duplication is neutral if starting point is not a promoter AND
        length is less than distance to the next promoter AND insertion locus is neutral.
        This method first checks if starting point is a deleterious locus by conducting a Bernoulli trial with parameter p = len(self.genome.loci) / self.genome.length.
        Then, checks if insertion locus is neutral by conducting a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.
        Then, checks if the length is deleterious.
        Then, checks if the ending point is deleterious.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        if not self.Bernoulli(len(self.genome.loci) / self.genome.length):
            return False
        
        if not self.Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        
        self.set_length()

        if not self.length_is_ok():
            return False
        
        return self.ending_point_is_ok(self.set_starting_point())

    
        
    def apply(self, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply()
        self.set_insertion_locus(self.genome.z_nc + self.genome.g - 1)
        self.map_local_to_absolute_locus(self.genome.insertion_binary_search)

        if self.DEBUG:
            self.starting_point = 1
            self.length = 5
            self.insertion_locus = 2
            print(f"Neutral", self,
                  f"\n\tStarting point: {self.starting_point}\n"
                  f"\tLength: {self.length}\n"
                  f"\tInsertion locus: {self.insertion_locus}")
        if not virtually:  
            self.genome.insert(self.insertion_locus, self.length)
        
"""
class Inversion(Mutation): # TODO don't transpose !!!!
    def __init__(self, rate: float, DEBUG: bool = False) -> None:
        super().__init__(rate, "Inversion", DEBUG)
    
    def is_neutral(self, genome: Genome):
        if rd.random() > genome.nc_proportion:
            if self.DEBUG:
                print(f"Deleterious starting point...")
            return False
        if rd.random() > (genome.z_nc - 1) / genome.length:
            if self.DEBUG:
                print(f"Deleterious ending point...")
            return False
        if rd.random() > (genome.z_nc + genome.g) / genome.length:
            if self.DEBUG:
                print(f"Deleterious insertion locus...")
            return False
        return True
    
    def apply(self, genome: Genome):
        # To get random but different starting and ending point, we draw a random
        # number that represent the absolute position in non coding genome.
        # We check the following promoter, and get the absolute position in whole genome that is:
        # absolute_position_in_non_coding + promoter_numbers_passed * gene_length
        breakpoints = rd.sample(range(genome.z_nc), 2)

        # DEBUG
        breakpoints = [0, 2]
        # /DEBUG

        # Ensure starting point is before ending point
        if breakpoints[0] > breakpoints[1]:
            breakpoints[0], breakpoints[1] = breakpoints[1], breakpoints[0]
            
        next_promoter_locus_index_record = []
        for index, breakpoint in enumerate(breakpoints):
            next_promoter_locus_index_record.append(genome.deletion_binary_search(breakpoint))
            breakpoints[index] += next_promoter_locus_index_record[-1] * genome.gene_length

        self.starting_point, ending_point = breakpoints
        self.length = ending_point - self.starting_point

        print(self.starting_point, self.length, breakpoints)
        # If the inversion is on the same sequence of non coding genome, the structure is unchanged
        if next_promoter_locus_index_record[0] == next_promoter_locus_index_record[1]:
            return None
        if next_promoter_locus_index_record[0] == 0 and next_promoter_locus_index_record[1] == len(genome.loci):
            return None
        

        # We know this mutation is neutral.
        # There are z_nc + g positions for a neutral insertion.
        # Once the position in non coding genome is known, the binary 
        # search gives the minimum index of promoters affected.
        # TODO
        self.insertion_locus = rd.randint(0, genome.z_nc + genome.g - self.length - 1) # both end points are included with rd.randint
        
        # DEBUG
        self.insertion_locus = 10
        # /DEBUG

        self.insertion_locus = genome.loci[genome.inversion_binary_search(self.insertion_locus, self.starting_point, ending_point)]

        if self.DEBUG:
            print(f"Neutral", self,
                  f"\n\tStarting point of inversion: {self.starting_point}\n"
                  f"\tLength: {self.length}\n"
                  f"\tInsertion locus: {self.insertion_locus}")

        genome.inverse(self.starting_point, ending_point, self.insertion_locus)
        """



if __name__ == "__main__":
    pass



    

