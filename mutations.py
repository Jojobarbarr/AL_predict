import random as rd
from typing import Callable

from genome import Genome
from stats import MutationStatistics


class Mutation:
    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
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
        self.stats.count += 1
        return True

    def apply(
        self,
        virtually: bool = False,
    ) -> None:
        self.stats.neutral_count += 1
        self.stats.length_sum += self.length
        self.stats.length_square_sum += self.length**2

    def _Bernoulli(
        self,
        p: float,
    ) -> bool:
        if p < 0 or p > 1:
            raise ValueError(f"p must be between 0 and 1. You provided p = {p}")
        if rd.random() < p:
            return True
        return False

    def _set_length(
        self,
    ):
        if "Small" in self.__class__.__name__:
            return rd.randint(1, self.l_m)
        return rd.randint(1, self.genome.length)

    def _pick_locus_in_neutral_space(
        self,
        max_locus: int,
    ):
        # both end points are included with rd.randint
        return rd.randint(0, max_locus)

    def _get_next_promoter_index(
        self,
        locus: int,
        binary_search: Callable[[int], int],
    ):
        insertion_locus_index = binary_search(locus)
        return insertion_locus_index

    def _length_is_ok(
        self,
    ) -> bool:
        if self.length <= self.genome.max_length_neutral:
            return True
        return False

    def _ending_point_is_ok(
        self,
        starting_locus: int,
        next_promoter_locus_index: int,
    ) -> bool:
        # Handle the case when starting point is between last promoter and ORI.
        if next_promoter_locus_index == len(self.genome.loci):
            # In circular genome case, first promoter locus is genome.loci[0] (mod genome.length).
            next_promoter_locus = self.genome.length + self.genome.loci[0]
        else:
            next_promoter_locus = self.genome.loci[next_promoter_locus_index]
        if self.length <= next_promoter_locus - starting_locus:
            return True
        return False

    def theory(
        self,
    ) -> tuple[float, float]:
        raise NotImplementedError(
            f"Daugther class {self.__class__.__name__} of Mutation must implement theory method."
        )


class PointMutation(Mutation):
    def is_neutral(
        self,
    ) -> bool:
        super().is_neutral()
        return self._Bernoulli(self.genome.z_nc / self.genome.length)

    def theory(
        self,
    ) -> tuple[float, float]:
        return (self.genome.z_nc / self.genome.length, 0)


class SmallInsertion(Mutation):
    def __init__(
        self,
        l_m: int = 10,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        super().__init__(genome)
        self.l_m = l_m

    def is_neutral(
        self,
    ) -> bool:
        super().is_neutral()
        return self._Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length)

    def apply(
        self,
        virtually: bool = False,
    ):
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

    def theory(
        self,
    ) -> tuple[float, float]:
        return (
            (self.genome.z_nc + self.genome.g) / self.genome.length,
            (1 + self.l_m) / 2,
        )


class Deletion(Mutation):
    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        super().__init__(genome)
        self.starting_locus = 0

    def is_neutral(
        self,
    ) -> bool:
        super().is_neutral()
        if not self._Bernoulli(self.genome.z_nc / self.genome.length):
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

    def apply(self, virtually: bool = False):
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

    def theory(
        self,
    ) -> tuple[float, float]:
        return (
            self.genome.z_nc
            * (self.genome.z_nc / self.genome.g + 1)
            / (2 * self.genome.length**2),
            self.genome.z_nc / (3 * self.genome.g),
        )


class SmallDeletion(Deletion):
    def __init__(
        self,
        l_m: int = 10,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        super().__init__(genome)
        self.l_m = l_m

    def theory(
        self,
    ) -> tuple[float, float]:
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
    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        super().__init__(genome)
        self.starting_locus = 0

    def _ending_point_is_ok(
        self,
        starting_locus: int,
        next_promoter_locus_index: int,
    ) -> bool:
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
        super().is_neutral()
        if not self._Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        self.length = self._set_length()
        if not self._length_is_ok():
            return False
        if not self._Bernoulli(1 - self.genome.g / self.genome.length):
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
    ):
        insertion_locus = self._pick_locus_in_neutral_space(
            self.genome.z_nc + self.genome.g - 1
        )
        next_promoter_index = self._get_next_promoter_index(
            insertion_locus, self.genome.insertion_binary_search
        )
        if not virtually:
            self.genome.insert(next_promoter_index, self.length)
        super().apply()

    def theory(
        self,
    ) -> tuple[float, float]:
        return (
            (
                (self.genome.z_nc + self.genome.g)
                * ((self.genome.z_c + self.genome.z_nc) / self.genome.g - 1)
            )
            / (2 * self.genome.length**2),
            (self.genome.z_c + self.genome.z_nc) / (3 * self.genome.g),
        )


class Inversion(Mutation):
    def __init__(
        self,
        genome: Genome = Genome(1, 1, 1),
    ) -> None:
        super().__init__(genome)
        self.starting_locus = 0

    def is_neutral(
        self,
    ) -> bool:
        super().is_neutral()
        if not self._Bernoulli((self.genome.z_nc + self.genome.g) / self.genome.length):
            return False
        return self._Bernoulli(
            (self.genome.z_nc + self.genome.g - 1) / (self.genome.length - 1)
        )

    def apply(
        self,
        virtually: bool = False,
    ):
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

    def theory(
        self,
    ) -> tuple[float, float]:
        """Returns the theoretical mutation neutrality probability from the mathematical model.

        Returns:
            float: mutation neutrality probability
        """
        return (
            (
                (self.genome.z_nc + self.genome.g)
                * (self.genome.z_nc + self.genome.g - 1)
            )
            / (self.genome.length * (self.genome.length - 1)),
            self.genome.length / 2,
        )
