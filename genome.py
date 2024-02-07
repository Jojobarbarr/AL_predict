import numpy as np
import random as rd

class Genome:
    def __init__(self, z_c: int, z_nc: int, g: int, DEBUG: bool=False) -> None:
        self.z_c = z_c
        self.z_nc = z_nc
        self.length = self.z_c + self.z_nc
        self.g = g

        self.DEBUG = DEBUG

        if z_c % g != 0:
            raise ValueError(f"z_c must be a multiple of g. z_c: {z_c} -- g: {g}")
        
        self.gene_length = self.z_c // self.g
        self.nc_proportion = self.z_nc / self.length
        self.max_length_neutral = 0
        if not self.DEBUG:
            self.loci = self.init_genome()
        else:
            self.g = 3
            self.gene_length = 2
            self.z_nc = 16
            self.z_c = self.g * self.gene_length
            self.length = self.g * self.gene_length + self.z_nc
            self.loci, self.genome = self.init_genome()
        
        self.update_features()
    
    def __str__(self) -> str:
        if not self.DEBUG:
            return f"\tloci: {self.loci}"
        
        return (f"\nGenome:\n"
                f"loci: {self.loci}\n"
                f"genome: {self.genome}\n")
    
    
    def init_genome(self):
        if not self.DEBUG:
            loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
            loci = np.array([locus + (segment * self.gene_length) + 1 for segment, locus in enumerate(loci_of_insertion)])
            orientation = np.array([1 for locus in loci])
            return loci

        # only executed in DEBUG mode

        loci_of_insertion = sorted(rd.sample(range(0, self.z_nc), self.g))
        loci = np.array([promoter + (segment * self.gene_length) + 1 for segment, promoter in enumerate(loci_of_insertion)])
        orientation = np.array([1 for locus in loci])
        genome = self.update_genome(loci)

        return loci, genome
    
        """
        nc_segment_size = self.z_nc // self.g
        first_promoter_position = 2
        # genome structure is: first PROMOTER at ORI + first_promoter_position, z_c // g long coding sequence, z_nc_init // g long non-coding sequence
        genome = np.array([first_promoter_position])
        genome = np.append(genome, [nc_segment_size for k in range(self.g - 1)])
        genome = np.append(genome, nc_segment_size - first_promoter_position)
        
        # visual_genome length is 90
        visual_genome = np.array([2, 20, 20, 18])
        """
    
    def binary_search(self, target):
        left, right = 0, len(self.loci) - 1
        while left <= right:
            middle = (left + right) // 2
            if self.loci[middle] - (middle * self.gene_length) <= target:
                left = middle + 1
            else:
                right = middle - 1
        return left

    def insert(self, locus: int, length: int):
        self.z_nc += length
        locus_after_insertion = self.loci >= locus
        self.loci[locus_after_insertion] += length
        self.update_features()
        if self.DEBUG:
            self.genome = self.update_genome(self.loci)

    
    def delete(self, locus: int, length: int):
        self.z_nc -= length
        locus_after_deletion = self.loci > locus
        self.loci[locus_after_deletion] -= length
        self.update_features()
        if self.DEBUG:
            self.genome = self.update_genome(self.loci)
    
    def parse(self):
        self.max_length_neutral = 0
        prev_locus = 0
        for locus in self.loci:
            distance = locus - prev_locus - self.gene_length
            if distance > self.max_length_neutral:
                self.max_length_neutral = distance
            prev_locus = locus

    def update_features(self):
        self.length = self.z_c + self.z_nc
        self.nc_proportion = self.z_nc / self.length
        self.parse()
        if self.DEBUG:
            print(f"New length: {self.length}\n"
                  f"New nc_proportion: {self.nc_proportion}\n"
                  f"New max_length_neutral: {self.max_length_neutral}")

    def update_genome(self, loci: np.ndarray):
        genome = np.array([])
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
    genome.insert(1, 4)
    print(genome)
    genome.delete(5, 2)
    print(genome)