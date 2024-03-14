import random as rd
from typing import Callable

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

    def _ending_point_is_ok(
        self,
        starting_locus: int,
        next_promoter_locus_index: int,
    ) -> bool:
        """Ensures that the ending point of the mutation is ok (for mutation types that need it, like deletion, inversion, duplication)

        Args:
            starting_locus (int): the starting locus in genome.
            next_promoter_locus_index (int): the next promoter index in genome.

        Returns:
            bool: True if ending point is ok, False otherwise.
        """
        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(self.genome.loci):
            # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
            next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            next_promoter_locus = self.genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - starting_locus:
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

    def is_neutral(
        self,
    ) -> bool:
        """Checks if the deletion is neutral or not.

        - Deletion is neutral if starting locus is a non coding base.
            A bernoulli trial is performed with probability z_nc / length.
        - Deletion is neutral if length is less than the maximum neutral length in self.genome.
        - Deletion is neutral if all bases between starting locus and ending point are non coding.

        Returns:
            bool: True if the deletion is neutral, False otherwise.
        """
        super().is_neutral()
        if not self._bernoulli(self.genome.z_nc / self.genome.length):
            return False
        self.length = self._set_length()
        if not self._length_is_ok():
            return False

        self.starting_locus = self._pick_locus_in_neutral_space(self.genome.z_nc - 1)
        next_promoter_locus_index = self._get_next_promoter_index(
            self.starting_locus, self.genome.deletion_binary_search
        )
        self.starting_locus += next_promoter_locus_index * self.genome.gene_length
        return self._ending_point_is_ok(self.starting_locus, next_promoter_locus_index)

    def apply(
        self,
        virtually: bool = False,
    ) -> bool:
        """Applies the deletion to the genome.

        Args:
            virtually (bool, optional): If True, the genome isn't modified. Defaults to False.
        """
        if not virtually:
            # If deletion is between the last promoter and the first, we need to proceed with two steps:
            # - Deletion from starting point to ORI
            # - Deletion from ORI to first promoter
            # without deleting more than self.length
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
                + (1 - self.l_m**2) / 3
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

    def _ending_point_is_ok(
        self,
        starting_locus: int,
        next_promoter_locus_index: int,
    ) -> bool:
        """Ensures that the ending point of the mutation is ok.

        Overrides the method in Mutation class. Ending point is ok if the bases copied are not promoters.

        Args:
            starting_locus (int): the starting locus in genome.
            next_promoter_locus_index (int): the next promoter index in genome.

        Returns:
            bool: True if ending point is ok, False otherwise.
        """
        if next_promoter_locus_index == 0:
            next_promoter_locus = self.genome.loci[0]
            if self.genome.orientation_list[0] == -1:
                next_promoter_locus += self.genome.gene_length - 1
            if self.length <= next_promoter_locus - starting_locus:
                return True
            return False

        previous_promoter_locus_index = next_promoter_locus_index - 1
        if self.genome.orientation_list[previous_promoter_locus_index] == -1:
            next_promoter_locus = (
                self.genome.loci[previous_promoter_locus_index]
                + self.genome.gene_length
            )
            if next_promoter_locus > starting_locus:
                if self.length <= next_promoter_locus - starting_locus:
                    return True
                return False
            if next_promoter_locus_index != len(self.genome.loci):
                if self.genome.orientation_list[next_promoter_locus_index] == -1:
                    next_promoter_locus = (
                        self.genome.loci[next_promoter_locus_index]
                        + self.genome.gene_length
                        - 1
                    )
                else:
                    next_promoter_locus = self.genome.loci[next_promoter_locus_index]
                if self.length <= next_promoter_locus - starting_locus:
                    return True
                return False

            # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
            next_promoter_locus_index = 0
            next_promoter_locus = self.genome.length
            if self.genome.orientation_list[next_promoter_locus_index] == -1:
                next_promoter_locus += (
                    self.genome.loci[next_promoter_locus_index]
                    + self.genome.gene_length
                    - 1
                )

        else:
            next_promoter_locus = 0
            if next_promoter_locus_index == len(self.genome.loci):
                # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
                next_promoter_locus_index = 0
                next_promoter_locus = self.genome.length
            if self.genome.orientation_list[next_promoter_locus_index] == -1:
                next_promoter_locus += (
                    self.genome.loci[next_promoter_locus_index]
                    + self.genome.gene_length
                    - 1
                )
            else:
                next_promoter_locus += self.genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - starting_locus:
            return True
        return False

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
        if not self._bernoulli(1 - self.genome.g / self.genome.length):
            return False

        self.starting_locus = self._pick_locus_in_neutral_space(
            self.genome.length - self.genome.g - 1
        )
        next_promoter_locus_index = self._get_next_promoter_index(
            self.starting_locus, self.genome.duplication_binary_search
        )
        self.starting_locus += next_promoter_locus_index
        return self._ending_point_is_ok(self.starting_locus, next_promoter_locus_index)

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
        try:
            return self._bernoulli(
                (self.genome.z_nc + self.genome.g - 1) / (self.genome.length - 1)
            )
        except ZeroDivisionError:
            # Genome lentgh is 1
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
        switched = False
        breaking_locus = rd.sample(range(0, self.genome.z_nc + self.genome.g), 2)
        if breaking_locus[1] < breaking_locus[0]:
            breaking_locus[0], breaking_locus[1] = breaking_locus[1], breaking_locus[0]
            switched = True
        next_promoter_locus_index_starting_locus = self._get_next_promoter_index(
            breaking_locus[0], self.genome.insertion_binary_search
        )
        next_promoter_locus_index_ending_point = self._get_next_promoter_index(
            breaking_locus[1], self.genome.insertion_binary_search
        )
        if (
            next_promoter_locus_index_ending_point
            == next_promoter_locus_index_starting_locus
        ):
            self.length = breaking_locus[1] - breaking_locus[0]
            genome_structure_changed = False

        else:
            self.starting_locus = (
                breaking_locus[0]
                + (self.genome.gene_length - 1)
                * next_promoter_locus_index_starting_locus
            )
            ending_point = (
                breaking_locus[1]
                + (self.genome.gene_length - 1) * next_promoter_locus_index_ending_point
            )
            self.length = ending_point - self.starting_locus
            if not virtually:
                self.genome.inverse(self.starting_locus, self.length)
        if switched:
            self.length = self.genome.length - self.length
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
