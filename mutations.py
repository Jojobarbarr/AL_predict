import random as rd

from genome import Genome
from stats import MutationStatistics


class Mutation:
    def __init__(self, name: str, *args) -> None:
        self.name = name
        self.stats = MutationStatistics()

        self.insertion_locus = 0
        self.starting_point = 0
        self.length = 0

        self.is_small = False
        self.l_m = 0
    
    def __str__(self) -> str:
        return self.name
    
    def is_neutral(self, *args) -> bool:
        self.stats.count += 1
        return True

    def apply(self, genome: Genome, switched: bool=False, virtually: bool=False):
        self.stats.neutral_count += 1
        if switched:
            self.length = genome.length - self.length
        self.stats.length_sum += self.length
        self.stats.length_square_sum += self.length ** 2

    def Bernoulli(self, p: float) -> bool:
        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1. You provided p = {p}")
        
        if rd.random() < p:
            return True
        
        return False
    
    def set_length(self, genome: Genome):
        """Set a random length for the mutation. If the mutation is a small one, the length is between 1 and self.l_m included.
        Otherwise, the length is between 1 ans self.genome.length.
        """
        if self.is_small:
            self.length = rd.randint(1, self.l_m)
        else:
            self.length = rd.randint(1, genome.length)
    
    def set_insertion_locus(self, upper_bound: int):
        """Choose a random insertion locus.

        Args:
            upper_bound (int): the upper bound of the random range.
        """
        self.insertion_locus = rd.randint(0, upper_bound) # both end points are included with rd.randint
    
    def map_local_to_absolute_locus(self, genome: Genome):
        """Map local absolute position to the absolute position in genome. This mapping depends on the mutation type and is provided in argument 'mapping'.

        Args:
            mapping (Callable[[int], int]): mapping that depends on the mutation type.
        """
        insertion_locus_index = genome.insertion_binary_search(self.insertion_locus)
        if insertion_locus_index == len(genome.loci):
            self.insertion_locus = -1
        else:
            self.insertion_locus = genome.loci[insertion_locus_index]
    
    def ending_point_is_ok(self, genome: Genome, next_promoter_locus_index: int) -> bool:
        """Checks if ending point is neutral. Ending point is neutral if length is less than distance between next promoter and starting point.

        Args:
            next_promoter_locus_index (int): next promoter locus index that allows to get next promoter absolute position.

        Returns:
            bool: Return False if ending point is deleterious.
        """
        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(genome.loci):
            # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
            next_promoter_locus = genome.length + genome.loci[0]
        else:
            next_promoter_locus = genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - self.starting_point:
            return True
        
        return False
    
    def theory(self) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (0, 0)
    



class PointMutation(Mutation):
    def __init__(self) -> None:
        super().__init__("Point Mutation")
    
    def is_neutral(self, genome: Genome) -> bool:
        """Check if mutation is neutral. Point mutation is neutral if it affects a non coding base.
        Therefore, we conduct a Bernoulli trial with parameter p = genome.z_nc / genome.length.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """        
        super().is_neutral(genome)
        return self.Bernoulli(genome.z_nc / genome.length)
    
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (genome.z_nc / genome.length, 0)



class SmallInsertion(Mutation):
    def __init__(self, l_m: int=10) -> None:
        super().__init__("Small Insertion")
        self.is_small = True
        self.l_m = l_m

    def is_neutral(self, genome: Genome) -> bool:
        """Check if mutation is neutral. Samll insertion is neutral if it inserts between two non coding bases, and right after a coding sequence
        Therefore, we conduct a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral(genome)
        return self.Bernoulli((genome.z_nc + genome.g) / genome.length)
    
    def apply(self, genome: Genome, virtually: bool=False):
        """Apply the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply(genome)
        self.set_insertion_locus(genome.z_nc + genome.g - 1)
        
        self.map_local_to_absolute_locus(genome)

        self.set_length(genome)

        if not virtually:
            genome.insert(self.insertion_locus, self.length)
    
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            tuple[float, float]: mutation neutrality probability
        """
        return ((genome.z_nc + genome.g) / genome.length,
                (1 + self.l_m) / 2)
        


class Deletion(Mutation):
    def __init__(self) -> None:
        super().__init__("Deletion")

    def length_is_ok(self, genome: Genome) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= genome.max_length_neutral:
            return True
        
        return False
    
    def set_starting_point(self, genome: Genome) -> int:
        """Sets deletion starting point. The starting poisition is a random base position in non coding genome. This position in non coding genome is then mapped
        to the absolute position in whole genome.

        Returns:
            int: the index of the next promoter locus.
        """
        self.starting_point = rd.randint(0, genome.z_nc - 1)
        next_promoter_locus_index = genome.deletion_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index * genome.gene_length
        return next_promoter_locus_index

    def is_neutral(self, genome: Genome) -> bool:
        """Checks if mutation is neutral. Deletion is neutral if starting point is a non coding base AND length is less than distance to the next coding section.
        This method first checks if starting point is a deleterious locus by conducting a Bernoulli trial with parameter p = genome.z_nc / genome.length.
        Then, checks if length is greater than the maximum non coding sequence.
        Then, checks if the ending point is deleterious.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral(genome)
        if not self.Bernoulli(genome.z_nc / genome.length):
            return False
        
        self.set_length(genome)
        
        if not self.length_is_ok(genome):
            return False

        return self.ending_point_is_ok(genome, self.set_starting_point(genome))
    
    def apply(self, genome: Genome, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply(genome)
        if not virtually:
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
        
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            tuple[float, float]: mutation neutrality probability and length mean to expect
        """
        return (genome.z_nc * (genome.z_nc / genome.g + 1) / (2 * genome.length ** 2),
                genome.z_nc / (3 * genome.g))


class SmallDeletion(Deletion):
    def __init__(self, l_m: int=10) -> None:
        """
        Args:
            rate (float): rate of occurence.
            l_m (int): maximum length of the mutation.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__()
        self.is_small = True
        self.l_m = l_m
        self.name = "Small Deletion"
    
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return ((genome.z_nc - genome.g * (self.l_m - 1) / 2) / genome.length,
                ((genome.z_nc / genome.g) * (self.l_m + 1) / 2 + (1 - self.l_m ** 2) / 3) / (genome.z_nc / genome.g - (self.l_m - 1) / 2))

class Duplication(Mutation):
    def __init__(self) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__("Duplication")
    
    def length_is_ok(self, genome: Genome) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= genome.max_length_neutral:
            return True
        return False
    
    def set_starting_point(self, genome: Genome) -> int:
        """Sets duplication starting point. The starting poisition is a random base position in genome as long as it is not a promoter. 
        This position in genome minus promoters is then mapped to the absolute position in whole genome.

        Returns:
            int: the index of the next promoter locus.
        """
        self.starting_point = rd.randint(0, genome.length - genome.g - 1)
        next_promoter_locus_index = genome.duplication_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index
        return next_promoter_locus_index
    
    def ending_point_is_ok(self, genome: Genome, next_promoter_locus_index: int) -> bool:
        """Checks if ending point is neutral. Ending point is neutral if length is less than distance between next promoter and starting point.

        Args:
            next_promoter_locus_index (int): next promoter locus index that allows to get next promoter absolute position.

        Returns:
            bool: Return False if ending point is deleterious.
        """
        if (genome.loci[next_promoter_locus_index - 1] + genome.gene_length > self.starting_point and
        genome.orientation_list[next_promoter_locus_index - 1] == -1 and
        next_promoter_locus_index != 0):
            next_promoter_locus = genome.loci[next_promoter_locus_index - 1] + genome.gene_length - 1
        else:
            next_promoter_locus = 0
            if next_promoter_locus_index == len(genome.loci):
                # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
                next_promoter_locus_index = 0
                next_promoter_locus = genome.length
            if genome.orientation_list[next_promoter_locus_index] == -1:
                next_promoter_locus += genome.loci[next_promoter_locus_index] + genome.gene_length - 1
            else:
                next_promoter_locus += genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - self.starting_point:
            return True
        return False
    
    def is_neutral(self, genome: Genome) -> bool:
        """Checks if mutation is neutral. Duplication is neutral if starting point is not a promoter AND
        length is less than distance to the next promoter AND insertion locus is neutral.
        This method first checks if starting point is a deleterious locus by conducting a Bernoulli trial with parameter p = 1 - len(self.genome.loci) / self.genome.length.
        Then, checks if insertion locus is neutral by conducting a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.
        Then, checks if the length is deleterious.
        Then, checks if the ending point is deleterious.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral(genome)
        if not self.Bernoulli((genome.z_nc + genome.g) / genome.length):
            return False
        
        self.set_length(genome)

        if not self.length_is_ok(genome):
            return False
    
        if not self.Bernoulli(1 - genome.g / genome.length):
            return False
        
        return self.ending_point_is_ok(genome, self.set_starting_point(genome))

    
        
    def apply(self, genome: Genome, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        super().apply(genome)
        self.set_insertion_locus(genome.z_nc + genome.g - 1)
        self.map_local_to_absolute_locus(genome)

        if not virtually:
            genome.insert(self.insertion_locus, self.length)
    
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (((genome.z_nc + genome.g) * ((genome.z_c + genome.z_nc) / genome.g - 1)) / (2 * genome.length ** 2),
                (genome.z_c + genome.z_nc) / (3 * genome.g))
    
        

class Inversion(Mutation):
    def __init__(self) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__("Inversion")
    
    def set_breaking_locus(self, genome: Genome) -> bool:
        switched = False
        breaking_locus = rd.sample(range(0, genome.z_nc + genome.g), 2)
        if breaking_locus[1] < breaking_locus[0]:
            breaking_locus[0], breaking_locus[1] = breaking_locus[1], breaking_locus[0]
            switched = True
        next_promoter_locus_index_starting_point = genome.insertion_binary_search(breaking_locus[0])
        next_promoter_locus_index_ending_point = genome.insertion_binary_search(breaking_locus[1])
        self.starting_point = breaking_locus[0] + (genome.gene_length - 1) * next_promoter_locus_index_starting_point
        ending_point = breaking_locus[1] + (genome.gene_length - 1) * next_promoter_locus_index_ending_point
        self.length = ending_point - self.starting_point
        return switched



    def is_neutral(self, genome: Genome) -> bool:
        """Checks if mutation is neutral. Deletion is neutral if the two breakpoints are 
        in non coding section and different from each other.
        
        This method first checks if starting point is neutral by conducting a Bernoulli trial 
        with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.
        Then, checks if the ending point is neutral and different from the starting point by 
        conducting a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g - 1) / self.genome.length).

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral(genome)
        if not self.Bernoulli((genome.z_nc + genome.g) / genome.length):
            return False
        return self.Bernoulli((genome.z_nc + genome.g - 1) / (genome.length - 1))

    
    def apply(self, genome: Genome, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        switched = self.set_breaking_locus(genome)
        length = self.length
        super().apply(genome, switched=switched)
        
        if not virtually:
            genome.inverse(self.starting_point, length)
    
    def theory(self, genome: Genome) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (((genome.z_nc + genome.g) * (genome.z_nc + genome.g - 1)) / (genome.length * (genome.length - 1)),
                genome.length / 2)



    

