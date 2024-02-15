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
        self.stats.length_square_sum += self.length ** 2

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
    
    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return self.genome.z_nc / self.genome.length



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

        if not virtually:
            self.genome.insert(self.insertion_locus, self.length)
    
    def test(self, insertion_locus_nc_coord: int, answer: int):
        """Test the implementation.

        Args:
            insertion_locus_nc_coord (int): Absolute position of the insertion in the neutral space.
            answer (int): Expected self.insertion_locus value.
        """
        print(f"Insertion locus in non coding space: {insertion_locus_nc_coord}")

        self.insertion_locus = insertion_locus_nc_coord        
        self.map_local_to_absolute_locus(self.genome.insertion_binary_search)

        if answer != self.insertion_locus:
            raise ValueError(f"Mapping is wrong, it gave {self.insertion_locus} when it was supposed to give {answer}")
        
        print(f"Insertion locus after mapping: {self.insertion_locus}")

        self.set_length()

        print(f"Neutral", self,
              f"\n\tStarting point: {self.insertion_locus}\n"
              f"\tLength: {self.length}")
        
        self.genome.insert(self.insertion_locus, self.length)

    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (self.genome.z_nc + self.genome.g) / self.genome.length
        


class Deletion(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int=-1, DEBUG: bool=False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, "Deletion", genome, DEBUG)

    def length_is_ok(self) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= self.genome.max_length_neutral:
            return True
        
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
    
    def test(self, starting_point_nc_coord: int, answer: int):
        """Test the implementation.

        Args:
            starting_point_nc_coord (int): Absolute position of the starting_point in the neutral space.
            answer (int): Expected self.starting_point value.
        """
        if self.is_small:
            print("This is a small deletion")
        print(f"Insertion locus in neutral space: {starting_point_nc_coord}")

        self.length = 3

        self.starting_point = starting_point_nc_coord
        next_promoter_locus_index = self.genome.deletion_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index * self.genome.gene_length

        
        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(self.genome.loci):
            # In circular genome case, first promoter locus is self.genome.loci[0] (mod self.genome.length).
            next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            next_promoter_locus = self.genome.loci[next_promoter_locus_index]

        if not (self.length <= next_promoter_locus - self.starting_point):
            print(f"This mutation is deleterious due to bad ending point")
            print(self)
            return None


        if answer != self.starting_point:
            raise ValueError(f"Mapping is wrong, it gave {self.starting_point} when it was supposed to give {answer}")
        
        print(f"Starting point locus after mapping: {self.starting_point}")

        print(f"Neutral", self,
              f"\n\tStarting point: {self.starting_point}\n"
              f"\tLength: {self.length}")
        
        if self.starting_point > self.genome.loci[-1]:
            end_deletion_length = min(self.genome.length - self.starting_point, self.length)
            self.genome.delete(self.genome.loci[-1], end_deletion_length)
            self.genome.delete(0, self.length - end_deletion_length)
        else:
            self.genome.delete(self.starting_point, self.length)
        
    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return self.genome.z_nc * (self.genome.z_nc / self.genome.g + 1) / (2 * self.genome.length ** 2)


class SmallDeletion(Deletion):
    def __init__(self, rate: float, genome: Genome, l_m: int, DEBUG: bool = False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            l_m (int): maximum length of the mutation.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, genome, l_m, DEBUG)
        self.is_small = True
        self.l_m = l_m
        self.type = "Small Deletion"
    
    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (self.genome.z_nc - (self.l_m - 1) / 2) / self.genome.length


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
        self.starting_point = rd.randint(0, self.genome.length - self.genome.g - 1)
        next_promoter_locus_index = self.genome.duplication_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index
        return next_promoter_locus_index


    def length_is_ok(self) -> bool:
        """Checks if length is neutral.

        Returns:
            bool: If length is deleterious, return False.
        """
        if self.length <= self.genome.max_length_neutral + 2 * (self.genome.gene_length - 1):
            return True
        return False

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
            if self.genome.orientation_list[-1] == -1:
                next_promoter_locus = self.genome.length + self.genome.loci[0] + self.genome.gene_length - 1
            else:
                next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            if self.genome.orientation_list[next_promoter_locus_index] == -1:
                next_promoter_locus = self.genome.loci[next_promoter_locus_index] + self.genome.gene_length - 1
            else:
                next_promoter_locus = self.genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - self.starting_point:
            return True
        
        return False
    
    def is_neutral(self) -> bool:
        """Checks if mutation is neutral. Duplication is neutral if starting point is not a promoter AND
        length is less than distance to the next promoter AND insertion locus is neutral.
        This method first checks if starting point is a deleterious locus by conducting a Bernoulli trial with parameter p = 1 - len(self.genome.loci) / self.genome.length.
        Then, checks if insertion locus is neutral by conducting a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.
        Then, checks if the length is deleterious.
        Then, checks if the ending point is deleterious.

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        if not self.Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        
        self.set_length()

        if not self.length_is_ok():
            return False
    
        if not self.Bernoulli(1 - self.genome.g / self.genome.length):
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

        if not virtually:
            self.genome.insert(self.insertion_locus, self.length)
    
    def test(self, starting_point_nc_coord: int, answer1: int, insertion_locus_nc_coord: int, answer2: int):
        """Test the implementation.

        Args:
            starting_point_nc_coord (int): Absolute position of the starting point in the neutral space.
            answer1 (int): Expected self.starting_point value.
            insertion_locus_nc_coord (int): Absolute position of the insertion in the neutral space.
            answer1 (int): Expected self.insertion_locus value.
        """
        print(f"Starting point locus in neutral space: {starting_point_nc_coord}")

        self.length = 4

        self.starting_point = starting_point_nc_coord
        next_promoter_locus_index = self.genome.duplication_binary_search(self.starting_point)
        self.starting_point += next_promoter_locus_index

        print(f"Starting point locus after mapping: {self.starting_point}") 

        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(self.genome.loci):
            # In circular genome case, first promoter locus is self.genome.loci[0] (mod self.genome.length).
            next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            next_promoter_locus = self.genome.loci[next_promoter_locus_index]
        if not (self.length <= next_promoter_locus - self.starting_point):
            print(f"This mutation is deleterious due to bad ending point")
            print(self)
            return None

        

        if answer1 != self.starting_point:
            raise ValueError(f"Mapping is wrong, it gave {self.starting_point} when it was supposed to give {answer1}")
        
        print(f"Insertion locus in neutral space: {insertion_locus_nc_coord}")

        self.insertion_locus = insertion_locus_nc_coord
        self.map_local_to_absolute_locus(self.genome.insertion_binary_search)

        if answer2 != self.insertion_locus:
            raise ValueError(f"Mapping is wrong, it gave {self.insertion_locus} when it was supposed to give {answer2}")
        
        print(f"Insertion locus after mapping: {self.insertion_locus}")
        print(f"Neutral", self,
              f"\n\tStarting point: {self.starting_point}\n"
              f"\tLength: {self.length}\n"
              f"\tInsertion locus: {self.insertion_locus}")
        self.genome.insert(self.insertion_locus, self.length)
    
    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return ((self.genome.z_nc + self.genome.g) * ((self.genome.z_c + self.genome.z_nc) / self.genome.g - 1)) / (2 * self.genome.length ** 2)
    
        

class Inversion(Mutation):
    def __init__(self, rate: float, genome: Genome, l_m: int=-1, DEBUG: bool = False) -> None:
        """
        Args:
            rate (float): rate of occurence.
            genome (Genome): genome on which the mutation is applied.
            DEBUG (bool, optional): Flag to activate prints. Defaults to False.
        """
        super().__init__(rate, "Inversion", genome, DEBUG)
    
    def set_breaking_locus(self):
        switched = False
        breaking_locus = rd.sample(range(0, self.genome.z_nc + self.genome.g), 2)
        if breaking_locus[1] < breaking_locus[0]:
            breaking_locus[0], breaking_locus[1] = breaking_locus[0], breaking_locus[1]
            switched = True
        next_promoter_locus_index_starting_point = self.genome.insertion_binary_search(breaking_locus[0])
        next_promoter_locus_index_ending_point = self.genome.insertion_binary_search(breaking_locus[1])

        self.starting_point = breaking_locus[0] + (self.genome.gene_length - 1) * (next_promoter_locus_index_starting_point - 1)
        ending_point = breaking_locus[1] + (self.genome.gene_length - 1) * (next_promoter_locus_index_ending_point - 1)
        self.length = ending_point - self.starting_point
        if switched:
            self.length = self.genome.length - self.length
    



    def is_neutral(self) -> bool:
        """Checks if mutation is neutral. Deletion is neutral if the two breakpoints are 
        in non coding section and different from each other.
        
        This method first checks if starting point is neutral by conducting a Bernoulli trial 
        with parameter p = (self.genome.z_nc + self.genome.g) / self.genome.length.
        Then, checks if the ending point is neutral and different from the starting point by 
        conducting a Bernoulli trial with parameter p = (self.genome.z_nc + self.genome.g - 1) / self.genome.length).

        Returns:
            bool: True if mutation is neutral, False if it is deleterious.
        """
        super().is_neutral()
        if not self.Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length) and self.DEBUG:
            return False
        return self.Bernoulli((self.genome.z_nc + self.genome.g - 1) / (self.genome.length - 1)) or self.DEBUG

    
    def apply(self, virtually: bool=False):
        """Applies the mutation. If virtually is True, all the mutation characteristics are determined but it is not applied on the genome.

        Args:
            virtually (bool, optional): If True, mutation isn't applied. Defaults to False.
        """
        self.set_breaking_locus()
        super().apply()
        
        if not virtually:
            self.genome.inverse(self.starting_point, self.length)
    
    def test(self, starting_point_nc_coord: int, answer1: int, ending_point_nc_coord: int, answer2: int):
        """Test the implementation.

        Args:
            starting_point_nc_coord (int): Absolute position of the starting point in the neutral space.
            answer1 (int): Expected self.starting_point value.
            ending_point_nc_coord (int): Absolute position of the ending point in the neutral space.
            answer1 (int): Expected ending_point value.
        """
        print(f"Starting point locus in neutral space: {starting_point_nc_coord}")

        switched = False
        breaking_locus = [starting_point_nc_coord, ending_point_nc_coord]
        if breaking_locus[1] < breaking_locus[0]:
            breaking_locus[0], breaking_locus[1] = breaking_locus[0], breaking_locus[1]
            switched = True
        next_promoter_locus_index_starting_point = self.genome.insertion_binary_search(breaking_locus[0])
        next_promoter_locus_index_ending_point = self.genome.insertion_binary_search(breaking_locus[1])

        self.starting_point = breaking_locus[0] + (self.genome.gene_length - 1) * (next_promoter_locus_index_starting_point - 1)
        ending_point = breaking_locus[1] + (self.genome.gene_length - 1) * (next_promoter_locus_index_ending_point - 1)
        self.length = ending_point - self.starting_point
        if switched:
            self.length = self.genome.length - self.length

        if answer1 != self.starting_point:
            raise ValueError(f"Mapping is wrong, it gave {self.starting_point} when it was supposed to give {answer1}")
        
        print(f"Starting point locus after mapping: {self.starting_point}")

        print(f"Ending point locus in neutral space: {ending_point_nc_coord}")

        ending_point = ending_point_nc_coord
        next_promoter_locus_index_ending_point = self.genome.insertion_binary_search(ending_point)
        ending_point += (self.genome.gene_length - 1) * (next_promoter_locus_index_ending_point - 1)

        if answer2 != ending_point - self.starting_point:
            raise ValueError(f"Mapping is wrong, it gave {ending_point - self.starting_point} when it was supposed to give {answer2}")
        
        print(f"Ending point locus after mapping: {ending_point}")

        print(f"Neutral", self,
              f"\n\tStarting point: {self.starting_point}\n"
              f"\tLength: {ending_point - self.starting_point}")
        
        self.genome.inverse(self.starting_point, ending_point - self.starting_point)
    
    def theory(self) -> float:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return ((self.genome.z_nc + self.genome.g) * (self.genome.z_nc + self.genome.g - 1)) / (self.genome.length * (self.genome.length - 1))



    

