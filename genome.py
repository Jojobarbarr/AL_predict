import numpy as np
import numpy.typing as npt
import random as rd

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
        print("Genome initialisation...")
        self.z_c = z_c
        self.z_nc = z_nc
        self.length = self.z_c + self.z_nc
        self.g = g
        self.homogeneous = homogeneous

        self.DEBUG = DEBUG
        
        self.gene_length = self.z_c // self.g
        self.nc_proportion = self.z_nc / self.length
        self.max_length_neutral = 0
        self.loci, self.genome = self.init_genome()
        
        self.update_features()
        print("Genome initialisation done.")

    def __str__(self) -> str:
        """Print the representation of the genome.

        Returns:
            str: representation of the genome.
        """
        if not self.DEBUG:
            return f"\tloci: {self.loci}"
        
        return (f"\nGenome:\n"
                f"loci: {self.loci}\n"
                f"genome: {self.genome}\n")
    
    def init_genome(self):
        """Create a random genome respecting the constraints given by the user.
        """
        if not self.DEBUG:
            if not self.homogeneous:
                # To create non homogeneous genome, we start from a non coding genome. 
                # g random locus of insertion are selected, and the genes are inserted.
                loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
                loci = np.array([locus + (segment * self.gene_length) + 1 for segment, locus in enumerate(loci_of_insertion)])
                # orientation = np.array([1 for locus in loci])
                return loci, np.empty(1)
            
            # To create homogeneous genome, promoters are regularly disposed.
            loci = np.array([promoter * (self.gene_length + (self.z_nc // self.g)) for promoter in range(self.g)])
            return loci, np.empty(1)

        ## Only executed in DEBUG mode
        if not self.homogeneous:
            loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
            loci = np.array([promoter + (segment * self.gene_length) + 1 for segment, promoter in enumerate(loci_of_insertion)])
            # orientation = np.array([1 for locus in loci])
        else:
            loci = np.array([promoter * (self.gene_length + (self.z_nc // self.g)) for promoter in range(self.g)])
            print(loci)
        genome = self.update_genome(loci)

        return loci, genome
    
    def set_genome(self, z_c: int, z_nc: int, g: int, loci: npt.NDArray[np.int_], genome: npt.NDArray[np.int_]):
        """Set manually an explicit genome. For DEBUG only.

        Args:
            z_c (int): number of conding bases.
            z_nc (int): number of non coding bases.
            g (int): number of coding segments.
            loci (npt.NDArray[np.int_]): The implicit genome description (absolute position of promoters).
            genome (npt.NDArray[np.int_]): The explicit genome description.
        """
        self.z_c = z_c
        self.z_nc = z_nc
        self.length = self.z_c + self.z_nc
        self.g = g
        self.homogeneous = False

        self.DEBUG = True
        
        self.gene_length = self.z_c // self.g
        self.nc_proportion = self.z_nc / self.length
        self.max_length_neutral = 0
        self.loci, self.genome = loci, genome

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
        """Mapping of target to genome absolute position in deletion case.

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
    
    ## NOT USED
    # def duplication_binary_search(self, target: int) -> int:
    #     """Mapping of target to genome absolute position in duplication case.

    #     Args:
    #         target (int): Position in neutral space.

    #     Returns:
    #         int: next promoter locus index
    #     """
    #     left, right = 0, len(self.loci) - 1
    #     while left <= right:
    #         middle = (left + right) // 2
    #         if self.loci[middle] <= target + middle:
    #             left = middle + 1
    #         else:
    #             right = middle - 1
    #     return left

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
    
    
    def parse(self):
        """Compute some genome characteristics that needs to loop through the genome.
        """
        self.max_length_neutral = 0
        prev_locus = 0
        for locus in self.loci:
            distance = locus - prev_locus - self.gene_length
            if distance > self.max_length_neutral:
                self.max_length_neutral = distance
            prev_locus = locus
        distance = self.length - self.loci[-1] - self.gene_length + self.loci[0]
        if distance > self.max_length_neutral:
            self.max_length_neutral = distance

    def update_features(self):
        """Compute some genome charasteristics from global attributes.
        """
        self.length = self.z_c + self.z_nc
        self.nc_proportion = self.z_nc / self.length
        self.parse()
        if self.DEBUG:
            print(f"New length: {self.length}\n"
                  f"New nc_proportion: {self.nc_proportion}\n"
                  f"New max_length_neutral: {self.max_length_neutral}")

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