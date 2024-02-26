import random as rd

import numpy as np
import numpy.typing as npt

from stats import GenomeStatistics


class Genome:
    def __init__(self, g: int, z_c: int, z_nc: int, homogeneous: bool=False, orientation: bool=False, DEBUG: bool=False):
        """
        Args:
            g (int): number of coding segments.
            z_c (int): number of conding bases.
            z_nc (int): number of non coding bases.
            homogeneous (bool): if True, non coding sequences are all the same size.
            orientation (bool): if True, every genes are in the same direction.
            DEBUG (bool, optional): Flag to activate prints and explicit genome visualisation. Defaults to False.
        """
        if g == 1 and z_c == 1 and z_nc == 1:
            # Dummy genome, no calculation will be done on it.
            return None
        
        self.z_c = z_c
        self.z_nc = z_nc
        self.length = self.z_c + self.z_nc

        self.g = g
        
        self.homogeneous = homogeneous
        self.orientation = orientation

        self.DEBUG = DEBUG


        self.stats = GenomeStatistics()
        self.gene_length = self.z_c // self.g
        self.max_length_neutral = 0
        self.loci, self.orientation_list, self.genome = self.init_genome()
        self.loci_interval = np.empty(self.g)
        self.update_features()
        

    def __str__(self) -> str:
        """Print the representation of the genome.

        Returns:
            str: representation of the genome.
        """
        return (f"\nGenome:\n"
                f"g: {self.g}\n"
                f"z_c: {self.z_c}\n"
                f"z_nc: {self.z_nc}\n"
                f"length: {self.length}\n"
                f"homogeneous: {self.homogeneous}\n"
                f"gene_length: {self.gene_length}\n"
        )
    
    def clone(self):
        genome = Genome(1, 1, 1)
        genome.DEBUG = self.DEBUG
        genome.z_c = self.z_c
        genome.z_nc = self.z_nc
        genome.length = self.length
        genome.g = self.g
        genome.homogeneous = self.homogeneous
        genome.orientation = self.orientation
        genome.stats = self.stats.clone()
        genome.gene_length = self.gene_length
        genome.max_length_neutral = self.max_length_neutral
        genome.loci = self.loci.copy()
        genome.orientation_list = self.orientation_list.copy()
        genome.genome = np.empty(1)
        genome.loci_interval = self.loci_interval.copy()
        return genome

    
    def init_genome(self):
        """Create a random genome respecting the constraints given by the user.
        """
        if self.orientation:
            orientation = np.array([1 for _ in range(self.g)])
        else:
            orientation = np.array([rd.choice([1, -1]) for _ in range(self.g)])
        if not self.DEBUG:
            if not self.homogeneous:
                # To create non homogeneous genome, we start from a non coding genome. 
                # g random locus of insertion are selected, and the genes are inserted.
                loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
                loci = np.array([locus + (segment * self.gene_length) + 1 for segment, locus in enumerate(loci_of_insertion)])
                return loci, orientation, np.empty(1)
            
            # To create homogeneous genome, promoters are regularly disposed.
            distance_between_promoters = self.gene_length + (self.z_nc // self.g)
            loci = np.array([promoter * distance_between_promoters for promoter in range(self.g)])
            return loci, orientation, np.empty(1)

        ## Only executed in DEBUG mode
        if not self.homogeneous:
            loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
            loci = np.array([promoter + (segment * self.gene_length) + 1 for segment, promoter in enumerate(loci_of_insertion)])
        else:
            loci = np.array([promoter * (self.gene_length + (self.z_nc // self.g)) for promoter in range(self.g)])
            print(loci)
        genome = self.update_genome(loci)

        return loci, orientation, genome

    def compute_intervals(self):
        """Compute the intervals between the promoters.
        """
        self.loci_interval = np.array([self.loci[i] - self.loci[i-1] for i in range(1, len(self.loci))])
    
    def set_genome(self, z_c: int, z_nc: int, g: int, loci: npt.NDArray[np.int_], genome: npt.NDArray[np.int_], orientation: npt.NDArray[np.int_]=np.array([])):
        """Set manually an explicit genome. For DEBUG only.

        Args:
            z_c (int): number of conding bases.
            z_nc (int): number of non coding bases.
            g (int): number of coding segments.
            loci (npt.NDArray[np.int_]): The implicit genome description (absolute position of promoters).
            genome (npt.NDArray[np.int_]): The explicit genome description.
            orientation (npt.NDArray[np.int_], optional): The orientation of the genes. Defaults to np.array([]).
        """
        self.z_c = z_c
        self.z_nc = z_nc
        self.length = self.z_c + self.z_nc
        self.g = g
        self.homogeneous = False
        if orientation.size == 0:
            orientation = np.array([1 for _ in range(self.g)])

        self.DEBUG = True
        
        self.gene_length = self.z_c // self.g
        self.nc_proportion = self.z_nc / self.length
        self.max_length_neutral = 0
        self.loci, self.orientation_list, self.genome = loci, orientation, genome

    def insertion_binary_search(self, target: int) -> int:
        """Mapping of target to genome absolute position in insertion case.

        Args:
            target (int): Position in neutral space.

        Returns:
            int: next promoter locus index
        """
        left, right = 0, len(self.loci) - 1
        while left <= right:
            middle = (left + right) // 2
            if self.loci[middle] < target + middle * (self.gene_length - 1):
                left = middle + 1
            else:
                right = middle - 1
        return left

    def deletion_binary_search(self, target: int) -> int:
        """Mapping of target to genome absolute position in deletion case.

        Args:
            target (int): Position in neutral space.

        Returns:
            int: next promoter locus index
        """
        left, right = 0, len(self.loci) - 1
        while left <= right:
            middle = (left + right) // 2
            if self.loci[middle] <= target + middle * self.gene_length:
                left = middle + 1
            else:
                right = middle - 1
        return left
    
    def duplication_binary_search(self, target: int) -> int:
        """Mapping of target to genome absolute position in duplication case.

        Args:
            target (int): Position in neutral space.

        Returns:
            int: next promoter locus index
        """
        left, right = 0, len(self.loci) - 1
        while left <= right:
            middle = (left + right) // 2
            if self.loci[middle] <= target + middle:
                left = middle + 1
            else:
                right = middle - 1
        return left

    def insert(self, locus: int, length: int):
        """Insertion method. Shift all the affected promoters (>= locus) by length.

        Args:
            locus (int): first promoter affected by the insertion.
            length (int): length of the insertion.
        """
        self.z_nc += length
        locus_after_insertion = self.loci >= locus
        self.loci[locus_after_insertion] += length
        self.update_features()
        if self.DEBUG:
            self.genome = self.update_genome(self.loci)
    
    def inverse(self, locus: int, length: int):
        """Inverse method. Reverse the sequence between locus and locus + length.

        Args:
            locus (int): first promoter affected by the inversion.
            length (int): length of the inversion.
        """
        end_locus = locus + length
        locus_affected = np.logical_and(self.loci >= locus, self.loci < end_locus)
        self.loci[locus_affected] = (locus - 1) + (end_locus - (self.loci[locus_affected][::-1] + self.gene_length - 1))
        self.orientation_list[locus_affected] = -self.orientation_list[locus_affected][::-1]
        self.update_features()
        if self.DEBUG:
            self.genome = self.update_genome(self.loci)

    
    def delete(self, locus: int, length: int):
        """Deletion method. Shift all the affected promoters (> locus) by length.

        Args:
            locus (int): last promoter unaffected by the deletion.
            length (int): length of the deletion.
        """
        self.z_nc -= length
        locus_after_deletion = self.loci > locus
        self.loci[locus_after_deletion] -= length
        self.update_features()
        if self.DEBUG:
            self.genome = self.update_genome(self.loci)
    
    def update_features(self):
        """Compute some genome charasteristics from global attributes.
        """
        self.length = self.z_c + self.z_nc
        self.compute_intervals()
        self.max_length_neutral = self.loci_interval.min()
        distance = self.length - self.loci[-1] + self.loci[0]
        if distance < self.max_length_neutral:
            self.max_length_neutral = distance

    def update_genome(self, loci: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Update the explicit genome.

        Args:
            loci (npt.NDArray[np.int_]): array of the absolute position of the promoters.

        Returns:
            npt.NDArray[np.int_]: explicit genome.
        """
        genome = np.array([], dtype=np.int_)
        for locus in loci:
            genome = np.append(genome, [0] * (locus - len(genome)))
            genome = np.append(genome, [1] * self.gene_length)
        genome = np.append(genome, [0] * (self.length - len(genome)))
        return genome

    def compute_stats(self):
        """Compute the genome statistics.
        """
        self.stats.compute(self)
        self.stats_computed = True


if __name__ == "__main__":
    np.random.seed(42)
    rd.seed(42)

    DEBUG = True

    g = 1000
    z_c = 1000 * g
    z_nc = 2000 * g

    

    genome = Genome(z_c, z_nc, g, DEBUG)
    print(genome)

    # print(genome.binary_search(12))

    # genome.insert(1, 4)
    # print(genome)
    # genome.delete(5, 2)
    # print(genome)