import random as rd
from typing import Callable
import numpy as np

from genome import Genome
from stats import MutationStatistics


class Mutation:
    """Mutation class is the base class for all mutation types.

    It contains the common methods and attributes for all mutation types.
    3 main methods are implemented in this class:
    - is_neutral: checks if the mutation is neutral or not.
    - apply: applies the mutation to the genome.
    - theory: returns the theoretical mutation neutrality probability from the mathematical model. Must be implemented in the derived classes.
    The others are helper methods for the main methods.

    Attributes:
        genome (Genome): the genome on which the mutation will be applied.
        stats (MutationStatistics): the statistics of the mutation.
        length (int): the length of the mutation.
        l_m (int): the maximum length of the mutation.
    """

    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Initializes the attributes of the Mutation class.

        Args:
            genome (Genome, optional): Genome object on which mutations are done. Defaults to Genome(1, 1, 1).
        """
        self.genome = genome
        self.stats = MutationStatistics()
        self.length = 0
        self.l_m = 10
        self.rng = np.random.default_rng()

    def __str__(
        self,
    ) -> str:
        return self.__class__.__name__

    def is_neutral(
        self,
    ) -> bool:
        """Counts the number of mutations and returns True.

        Returns:
            bool: True
        """
        self.stats.count += 1
        return True

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Counts the number of neutral mutations and their length.

        Args:
            virtually (bool, optional): Used in derived classes. Defaults to False.
        """
        self.stats.neutral_count += 1
        self.stats.length_sum += self.length
        self.stats.length_square_sum += self.length**2
        return False

    def theory(
        self,
    ) -> tuple[float, float]:
        """Returns the neutral proportion and length mean predicted by the mathematical model.

        Raises:
            NotImplementedError: This method must be implemented in the derived classes.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        raise NotImplementedError(
            f"Derived class {self.__class__.__name__} of Mutation must implement theory method."
        )

    def _pick_segment(
        self,
    ) -> int:
        if self.genome.loci_interval.all() == 0:
            return False
        segment = self.rng.choice(
            len(self.genome.loci_interval),
            p=self.genome.loci_interval / self.genome.z_nc,
        )
        return segment

    def _bernoulli(
        self,
        p: float,
    ) -> bool:
        """Performs a bernoulli trial with probability p.

        Args:
            p (float): parameter of the bernoulli trial.

        Raises:
            ValueError: ensure that p is between 0 and 1.

        Returns:
            bool: return True if the trial is successful, False otherwise.
        """
        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1. You provided p = {p}")
        if rd.random() < p:
            return True
        return False

    def _set_length(
        self,
    ) -> int:
        """Sets the random length of the mutation.

        Returns:
            int: the length of the mutation.
        """
        if "Small" in self.__class__.__name__:
            return rd.randint(1, self.l_m)
        return rd.randint(1, self.genome.length)

    def _pick_locus_in_neutral_space(
        self,
        max_locus: int,
    ) -> int:
        """Picks a random locus in the neutral space.

        Args:
            max_locus (int): neutral space upper bound. (lower bound is 0)

        Returns:
            int: a random locus in the neutral space.
        """
        # both end points are included with rd.randint
        return rd.randint(0, max_locus)

    def _get_next_promoter_index(
        self,
        locus: int,
        binary_search: Callable[[int], int],
    ) -> int:
        """Maps the neutral space to the genome space.

        Find the absolute position of the next promoter index after locus.

        Args:
            locus (int): the locus in the neutral space.
            binary_search (Callable[[int], int]): binary search mapping neutral space to genome space. Depends on the mutation type.

        Returns:
            int: the absolute position of the next promoter index after locus.
        """
        insertion_locus_index = binary_search(locus)
        return insertion_locus_index

    def _length_is_ok(
        self,
    ) -> bool:
        """Ensures thath length isn't greater than the maximum neutral length in self.genome.

        Returns:
            bool: True if length is ok, False otherwise.
        """
        if self.length <= self.genome.max_length_neutral:
            return True
        return False


class PointMutation(Mutation):
    """Point Mutation, on base is replaced by another."""

    def is_neutral(
        self,
    ) -> bool:
        """Checks if the mutation is neutral or not.

        Point mutation is neutral if mutation happens on a non coding base.
        A bernoulli trial is performed with probability z_nc/length.

        Returns:
            bool: True if the mutation is neutral, False otherwise.
        """
        super().is_neutral()
        return self._bernoulli(self.genome.z_nc / self.genome.length)

    def theory(
        self,
    ) -> tuple[float, float]:
        """Theoretical mutation neutrality proportion and length mean from the mathematical model.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        return (self.genome.z_nc / self.genome.length, 0)


class SmallInsertion(Mutation):
    """Small Insertion, a small sequence is inserted in the genome."""

    def __init__(
        self,
        l_m,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Specifies the maximum length of the insertion.

        Args:
            l_m (int): maximum length of the insertion.
            genome (Genome, optional): Genome object. Defaults to Genome(1, 1, 1).
        """
        super().__init__(genome)
        self.l_m = l_m

    def is_neutral(
        self,
    ) -> bool:
        """Checks if the small insertion is neutral or not.

        Small insertion is neutral if insertion locus happens between two non conding base or right after or before a coding sequence.
        A bernoulli trial is performed with probability (z_nc + g) / length.

        Returns:
            bool: True if the small insertion is neutral, False otherwise.
        """
        super().is_neutral()
        return self._bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length)

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Applies the small insertion to the genome.

        Sets the insertion locus and the length of the insertion.

        Args:
            virtually (bool, optional): If True, the genome isn't modified. Defaults to False.
        """
        insertion_locus = self._pick_locus_in_neutral_space(
            self.genome.z_nc + self.genome.g - 1
        )
        next_promoter_index = self._get_next_promoter_index(
            insertion_locus, self.genome.insertion_binary_search
        )
        self.length = self._set_length()
        if not virtually:
            self.genome.insert(next_promoter_index, self.length)
        super().apply()
        return True

    def theory(
        self,
    ) -> tuple[float, float]:
        """Theoretical small insertion neutrality proportion and length mean from the mathematical model.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        return (
            (self.genome.z_nc + self.genome.g) / self.genome.length,
            (1 + self.l_m) / 2,
        )


class Deletion(Mutation):
    """Deletion, a sequence is deleted from the genome."""

    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Initializes the starting locus of the deletion.

        Args:
            genome (Genome, optional): Genome object. Defaults to Genome(1, 1, 1).
        """
        super().__init__(genome)
        self.starting_locus = 0
        self.orientation = 1

    def is_neutral(
        self,
    ) -> bool:
        super().is_neutral()
        if not self._bernoulli(self.genome.z_nc / self.genome.length):
            return False
        self.length = self._set_length()
        if not self._length_is_ok():
            return False
        segment = int(self._pick_segment())
        # print(f"length: {self.length}")
        # print(f"segment: {segment}")
        # print(f"loci: {self.genome.loci}")
        # print(f"interval: {self.genome.loci_interval}")
        if self.length > self.genome.loci_interval[segment]:
            return False
        if segment == 0:
            first_neutral_locus = 0 - (
                self.genome.length
                - self.genome.loci[segment - 1]
                - self.genome.gene_length
            )
        else:
            first_neutral_locus = (
                self.genome.loci[segment - 1] + self.genome.gene_length
            )
        last_neutral_locus = self.genome.loci[segment]
        # print(f"first_neutral_locus: {first_neutral_locus}")
        # print(f"last_neutral_locus: {last_neutral_locus}")
        self.starting_locus = self.rng.integers(
            first_neutral_locus,
            last_neutral_locus,
        )
        # print(f"starting_locus: {self.starting_locus}")
        if self._bernoulli(1 / 2):
            # deletion is forward
            if self.starting_locus + self.length - 1 > last_neutral_locus:
                return False
            self.orientation = 1
            return True
        # deletion is backward
        if self.starting_locus - self.length + 1 < first_neutral_locus:
            return False
        self.starting_locus = self.starting_locus - self.length + 1
        self.orientation = -1
        return True

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Applies the deletion to the genome.

        Args:
            virtually (bool, optional): If True, the genome isn't modified. Defaults to False.
        """
        # print(f"orientation: {self.orientation}")
        # print(f"length: {self.length}")
        # print(f"loci: {self.genome.loci}")
        # print(f"interval: {self.genome.loci_interval}")
        if not virtually:
            if self.starting_locus < 0:
                self.starting_locus += self.genome.length
            if self.starting_locus > self.genome.loci[-1]:
                end_deletion_length = min(
                    self.genome.length - self.starting_locus, self.length
                )
                self.genome.delete(self.genome.loci[-1], end_deletion_length)
                self.genome.delete(0, self.length - end_deletion_length)
            else:
                self.genome.delete(self.starting_locus, self.length)
        super().apply()
        return True

    def theory(
        self,
    ) -> tuple[float, float]:
        """Theoretical deletion neutrality proportion and length mean from the mathematical model.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        return (
            self.genome.z_nc
            * (self.genome.z_nc / self.genome.g + 1)
            / (2 * self.genome.length**2),
            self.genome.z_nc / (3 * self.genome.g),
        )


class SmallDeletion(Deletion):
    """Small Deletion, a sequence is deleted from the genome with a maximum length l_m."""

    def __init__(
        self,
        l_m,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Specifies the maximum length of the insertion.

        Args:
            l_m (int): maximum length of the insertion.
            genome (Genome, optional): Genome object. Defaults to Genome(1, 1, 1).
        """
        super().__init__(genome)
        self.l_m = l_m

    def theory(
        self,
    ) -> tuple[float, float]:
        """Theoretical small deletion neutrality proportion and length mean from the mathematical model.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        return (
            (self.genome.z_nc - self.genome.g * (self.l_m - 1) / 2)
            / self.genome.length,
            (
                (self.genome.z_nc / self.genome.g) * (self.l_m + 1) / 2
                + (1 - self.l_m * self.l_m) / 3
            )
            / (self.genome.z_nc / self.genome.g - (self.l_m - 1) / 2),
        )


class Duplication(Mutation):
    """Duplication, a sequence is copied then inserted elsewhere in the genome."""

    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Initializes the starting locus of the duplication.

        Args:
            genome (Genome, optional): Genome object. Defaults to Genome(1, 1, 1).
        """
        super().__init__(genome)
        self.starting_locus = 0
        self.orientation = 1

    def _first_neutral_locus(
        self,
        segment: int,
        reverse: bool = False,
    ) -> int:
        """Returns the first neutral locus in the segment.

        Args:
            segment (int): the segment index.
            reverse (bool, optional): If True, the orientation is reversed. Defaults to False.

        Returns:
            int: the first neutral locus in the segment.
        """
        if segment == 0:
            if reverse:
                return 0 - (
                    self.genome.length
                    - self.genome.loci[segment - 1]
                    - self.genome.gene_length
                )
            return 0 - (self.genome.length - self.genome.loci[segment - 1] - 1)
        if reverse:
            return self.genome.loci[segment - 1] + self.genome.gene_length
        return self.genome.loci[segment - 1] + 1

    def is_neutral(
        self,
    ) -> bool:
        """Checks if the duplication is neutral or not.

        - Duplication is neutral if insertion locus happens between two non conding base or right after or before a coding sequence.
            A bernoulli trial is performed with probability (z_nc + g) / length.
        - Duplication is neutral if length is less than the maximum neutral length in self.genome.
        - Duplication is neutral if starting locus is not a promoter.
            A bernoulli trial is performed with probability 1 - g / length.
        - Duplication is neutral if ending point is ok (there are no promoters between ending locus and starting locus).

        Returns:
            bool: True if the duplication is neutral, False otherwise.
        """
        super().is_neutral()
        if not self._bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        self.length = self._set_length()
        if not self._length_is_ok():
            return False
        segment = self._pick_segment()
        corrector = 1
        first_neutral_reverse = False
        last_neutral_reverse = False
        if self.genome.orientation_list[segment - 1] == -1:
            corrector -= 1
            first_neutral_reverse = True
        if self.genome.orientation_list[segment] == -1:
            corrector += 1
            last_neutral_reverse = True
        neutral_length = self.genome.loci_interval[segment] + corrector * (
            self.genome.gene_length - 1
        )
        if self.length > neutral_length:
            return False
        first_neutral_locus = self._first_neutral_locus(segment, first_neutral_reverse)
        if last_neutral_reverse:
            last_neutral_locus = self.genome.loci[segment] + self.genome.gene_length - 1
        last_neutral_locus = self.genome.loci[segment]

        self.starting_locus = self.rng.integers(
            first_neutral_locus,
            last_neutral_locus,
        )
        if self._bernoulli(1 / 2):
            # duplication is forward
            if self.starting_locus + self.length - 1 > last_neutral_locus:
                return False
            self.orientation = 1
            return True
        # deletion is backward
        if self.starting_locus - self.length + 1 < first_neutral_locus:
            return False
        self.orientation = -1
        self.starting_locus = self.starting_locus - self.length + 1
        return True

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Applies the duplication to the genome.

        Sets the insertion locus.

        Args:
            virtually (bool, optional): If True, the genome isn't modified. Defaults to False.
        """
        insertion_locus = self._pick_locus_in_neutral_space(
            self.genome.z_nc + self.genome.g - 1
        )
        next_promoter_index = self._get_next_promoter_index(
            insertion_locus, self.genome.insertion_binary_search
        )
        if not virtually:
            self.genome.insert(next_promoter_index, self.length)
        super().apply()
        return True

    def theory(
        self,
    ) -> tuple[float, float]:
        """Theoretical duplication neutrality proportion and length mean from the mathematical model.

        Returns:
            tuple[float, float]: (neutral proportion, length mean)
        """
        return (
            (
                (self.genome.z_nc + self.genome.g)
                * ((self.genome.z_c + self.genome.z_nc) / self.genome.g - 1)
            )
            / (2 * self.genome.length**2),
            (self.genome.z_c + self.genome.z_nc) / (3 * self.genome.g),
        )


class Inversion(Mutation):
    """Inversion, a sequence is reverted."""

    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        """Initializes the starting locus of the inversion.

        Args:
            genome (Genome, optional): Genome object. Defaults to Genome(1, 1, 1).
        """
        super().__init__(genome)
        self.starting_locus = 0
        self.ending_locus = 0
        self.orientation = 0

    def is_neutral(
        self,
    ) -> bool:
        """Checks if the inversion is neutral or not.

        - Inversion is neutral if starting locus happens between two non conding base or right after or before a coding sequence.
            A bernoulli trial is performed with probability (z_nc + g) / length.
        - Inversion is neutral if ending locus happens between two non conding base or right after or before a coding sequence. The ending locus is different from the starting locus.
            A bernoulli trial is performed with probability (z_nc + g - 1) / (length - 1).

        Returns:
            bool: True if the inversion is neutral, False otherwise.
        """
        super().is_neutral()
        if not self._bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        self.length = self._set_length()
        segment = int(self._pick_segment())
        if segment == 0:
            first_neutral_locus = 0 - (
                self.genome.length
                - self.genome.loci[segment - 1]
                - self.genome.gene_length
            )
        else:
            first_neutral_locus = (
                self.genome.loci[segment - 1] + self.genome.gene_length
            )
        last_neutral_locus = self.genome.loci[segment]
        self.starting_locus = self.rng.integers(
            first_neutral_locus,
            last_neutral_locus,
        )
        if self.starting_locus < 0:
            self.starting_locus += self.genome.length
        # print(f"\nGenome length: {self.genome.length}")
        if self._bernoulli(1 / 2):
            # deletion is forward
            # print("Forward")
            endpoint = self.starting_locus + self.length - 1
            # print(f"Endpoint before check: {endpoint}")
            if endpoint >= self.genome.length:
                endpoint -= self.genome.length
        else:
            # deletion is backward
            # print("Backward")
            endpoint = self.starting_locus - self.length + 1
            # print(f"Endpoint before check: {endpoint}")
            if endpoint < 0:
                endpoint += self.genome.length
        # print(f"endpoint: {endpoint}")
        # print(f"loci: {self.genome.loci}")
        mask_less_than_endpoint = self.genome.loci[self.genome.loci <= endpoint]
        mask_greater_than_endpoint = self.genome.loci[self.genome.loci > endpoint]
        if len(mask_less_than_endpoint) == 0:
            min_endpoint = self.genome.loci[-1] + self.genome.gene_length - 1
        else:
            min_endpoint = mask_less_than_endpoint[-1] + self.genome.gene_length - 1

        if len(mask_greater_than_endpoint) == 0:
            max_endpoint = self.genome.loci[0]
        else:
            max_endpoint = mask_greater_than_endpoint[0]

        # print(f"Endpoint: {endpoint}")
        # print(f"Min endpoint: {min_endpoint}")
        # print(f"Max endpoint: {max_endpoint}")
        if min_endpoint < endpoint:
            # print("Min ok")
            if max_endpoint > endpoint:
                # print("Max ok")
                self.ending_locus = endpoint
                if self.ending_locus < self.starting_locus:
                    self.starting_locus, self.ending_locus = (
                        self.ending_locus,
                        self.starting_locus,
                    )
                return True
        return False

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Applies the inversion to the genome.

        Sets the breaking locus. If starting locus is greater than ending locus, the inversion is reverted. Length is then genome.length - length.

        Args:
            virtually (bool, optional): If True, the genome isn't modified. Defaults to False.
        """
        genome_structure_changed = True
        if not virtually:
            self.genome.inverse(self.starting_locus, self.length)
        super().apply()
        return genome_structure_changed

    def theory(
        self,
    ) -> tuple[float, float]:
        """Returns the theoretical inversion neutrality probability from the mathematical model.

        Returns:
            float: inversion neutrality probability
        """
        return (
            (
                (self.genome.z_nc + self.genome.g)
                * (self.genome.z_nc + self.genome.g - 1)
            )
            / (self.genome.length * (self.genome.length - 1)),
            self.genome.length / 2,
        )
