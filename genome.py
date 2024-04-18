import random as rd
import traceback
import numpy as np

from stats import GenomeStatistics


class Genome:
    """Genome class.
    Contains all the method to manipulate the genome.
    """

    def __init__(
        self,
        g: int,
        z_c: int,
        z_nc: int,
        homogeneous: bool = False,
        orientation: bool = False,
    ) -> None:
        """Genome constructor.

        Args:
            g (int): Number of segments (one segment is one non coding segment + one coding segment) in the genome.
            z_c (int): Number of coding bases in the genome.
            z_nc (int): Number of non coding bases in the genome.
            homogeneous (bool, optional): If True, the genome is homogeneous: every non coding segments are the same size (+/- 1). Defaults to False.
            orientation (bool, optional): If True, all genes are in the same direction. Defaults to False.
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

        self.stats = GenomeStatistics()
        self.gene_length = self.z_c // self.g
        self.max_length_neutral = 0
        self.loci, self.orientation_list = self.init_genome()
        self.loci_interval = np.empty(self.g)
        self.update_features()

    def __str__(self) -> str:
        return (
            f"\nGenome:\n"
            f"g: {self.g}\n"
            f"z_c: {self.z_c}\n"
            f"z_nc: {self.z_nc}\n"
            f"length: {self.length}\n"
            f"homogeneous: {self.homogeneous}\n"
            f"gene_length: {self.gene_length}\n"
        )

    def clone(self) -> "Genome":
        """Deep clone of the genome.

        Returns:
            Genome: The clone of the self genome
        """
        genome = Genome(1, 1, 1)

        genome.z_c = self.z_c
        genome.z_nc = self.z_nc
        genome.length = self.length

        genome.g = self.g

        genome.homogeneous = self.homogeneous
        genome.orientation = self.orientation

        genome.gene_length = self.gene_length
        genome.max_length_neutral = self.max_length_neutral

        genome.stats = self.stats.clone()

        genome.loci = self.loci.copy()
        genome.orientation_list = self.orientation_list.copy()
        genome.loci_interval = self.loci_interval.copy()
        return genome

    def init_genome(self) -> tuple[np.ndarray, np.ndarray]:
        """Initialize the genome structure

        Returns:
            tuple[np.ndarray, np.ndarray]: (loci of the first base of genes, orientation of genes)
        """
        if self.orientation:
            orientation = np.array([1 for _ in range(self.g)], dtype=np.int_)
        else:
            orientation = np.array(
                [rd.choice([1, -1]) for _ in range(self.g)], dtype=np.int_
            )

        if not self.homogeneous:
            # To create non homogeneous genome, we start from a non coding genome.
            # g random locus of insertion are selected, and the genes are inserted.
            loci_of_insertion = sorted(rd.sample(range(0, self.z_nc + self.g), self.g))
            loci = np.array(
                [
                    locus + segment * (self.gene_length - 1)
                    for segment, locus in enumerate(loci_of_insertion)
                ],
                dtype=np.int_,
            )
            return loci, orientation

        # To create homogeneous genome, promoters are regularly disposed.
        distance_between_promoters = self.gene_length + (self.z_nc // self.g)
        loci = np.array(
            [promoter * distance_between_promoters for promoter in range(self.g)],
            dtype=np.int_,
        )
        return loci, orientation

    def compute_intervals(self):
        """Computes the number of non coding bases between the genes."""
        self.loci_interval = np.array(
            [
                self.loci[i] - self.loci[i - 1] - self.gene_length
                for i in range(len(self.loci))
            ],
            dtype=np.int_,
        )
        self.loci_interval[0] = (
            self.length + self.loci[0] - self.loci[-1] - self.gene_length
        )
        # if self.loci_interval[self.loci_interval < 0].any():
        #     print(self.loci)
        #     print(self.loci_interval)
        #     print(traceback.print_stack())
        #     raise ValueError("Negative interval detected.")

    def insertion_binary_search(
        self,
        target: int,
    ) -> int:
        """_summary_

        Args:
            target (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            int: _description_
        """
        if target < 0 or target > self.z_nc + self.g:
            raise ValueError("Target is out of the neutral space.")
        left, right = 0, len(self.loci) - 1
        while left <= right:
            middle = (left + right) // 2
            if self.loci[middle] < target + middle * (self.gene_length - 1):
                left = middle + 1
            else:
                right = middle - 1
        return left

    def insert(self, locus_index: int, length: int):
        self.z_nc += length
        self.loci[locus_index:] += length
        self.update_features()

    def inverse(self, locus: int, end_locus: int):
        # print(f"locus: {locus}, end_locus: {end_locus}")
        # print(f"loci: {self.loci}")
        locus_affected = np.logical_and(self.loci >= locus, self.loci < end_locus)
        self.loci[locus_affected] = (
            locus + end_locus - (self.loci[locus_affected][::-1] + self.gene_length)
        )
        self.orientation_list[locus_affected] = -self.orientation_list[locus_affected][
            ::-1
        ]
        self.update_features()
        # print(f"After inversion:")
        # print(f"loci: {self.loci}")
        # print(f"interval: {self.loci_interval}")

    def delete(self, locus: int, length: int):
        self.z_nc -= length
        if locus < self.loci[-1]:
            locus_after_deletion = self.loci > locus
            self.loci[locus_after_deletion] -= length
        self.update_features()

    def blend(self):
        nc_lengths = self.z_nc // self.g
        remaining_nc = self.z_nc % self.g
        distance_between_promoters = nc_lengths
        self.loci_interval = np.array(
            [distance_between_promoters for _ in range(self.g)], dtype=np.int_
        )
        remaining_insertions = rd.sample(range(0, self.g), remaining_nc)
        self.loci_interval[remaining_insertions] += 1

        self.loci = np.cumsum(self.loci_interval)
        self.loci = np.array(
            [locus + index * self.gene_length for index, locus in enumerate(self.loci)],
            dtype=np.int_,
        )

        self.orientation_list = np.array([1 for _ in range(self.g)], dtype=np.int_)
        self.update_features(skip_intervals=True)
        return self

    def update_features(self, skip_intervals: bool = False):
        self.length = self.z_c + self.z_nc
        if not skip_intervals:
            self.compute_intervals()
        self.max_length_neutral = self.loci_interval.max() + self.gene_length - 1
        if not self.orientation:
            self.max_length_neutral += self.gene_length - 1
            distance = self.length - self.loci[-1] + self.loci[0] + self.gene_length - 2
        else:
            distance = self.length - self.loci[-1] + self.loci[0] - 1
        if distance > self.max_length_neutral:
            self.max_length_neutral = distance

    def compute_stats(self):
        """Compute the genome statistics."""
        self.stats.compute(self)
