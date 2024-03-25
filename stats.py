import numpy as np
from typing import Any


class Statistics:
    def __init__(self) -> None:
        pass

    def mean(self, sum: int | float, population: int | float) -> float:
        if population == 0:
            print("Empty population.")
            return 0
        return sum / population

    def mean_estimator_variance(self, variance: float, number_of_samples: int) -> float:
        if number_of_samples == 0:
            print("Empty or one individual sample.")
            return 0
        return (variance / number_of_samples) ** 0.5

    def variance(
        self, square_sum: int | float, population: int | float, mean: float
    ) -> float:
        if population <= 1:
            print("Empty or one individual population.")
            return 0
        return square_sum / (population - 1) - mean**2


class MutationStatistics(Statistics):
    def __init__(self):
        self.count = 0
        self.neutral_count = 0
        self.length_sum = 0
        self.length_square_sum = 0
        self.d_stats = {}

    def __str__(self) -> str:
        return str(self.d_stats)

    def compute(self, theory: tuple[float, float] = (0, 0)):
        neutral_mean = self.mean(self.neutral_count, self.count)
        neutral_std = self.variance(self.neutral_count, self.count, neutral_mean)
        neutral_mean_std = self.mean_estimator_variance(neutral_std, self.count)

        lenght_mean = self.mean(self.length_sum, self.neutral_count)
        length_std = self.variance(
            self.length_square_sum, self.neutral_count, lenght_mean
        )
        length_mean_std = self.mean_estimator_variance(length_std, self.neutral_count)
        self.d_stats = {
            "Total mutations": self.count,
            "Neutral mutations": self.neutral_count,
            "Neutral mutations proportion": neutral_mean,
            "Neutral mutations standard deviation of proportion estimator": neutral_mean_std,
            "Neutral probability theory": theory[0],
            "Length mean": lenght_mean,
            "Length standard deviation of mean estimator": length_mean_std,
            "Length mean theory": theory[1],
            "Length standard deviation": length_std**0.5,
        }


class GenomeStatistics(Statistics):
    def __init__(
        self,
        nc_proportion: float = 0,
        intervals_between_loci: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(
            (0), dtype=int
        ),
        d_stats: dict[str, Any] = {
            "Non coding proportion": 0,
            "Non coding length list": 0,
        },
    ) -> None:
        self.nc_proportion = nc_proportion
        self.intervals_between_loci = intervals_between_loci
        self.d_stats = d_stats

    def clone(self):
        nc_proportion = self.nc_proportion
        intervals_between_loci = self.intervals_between_loci.copy()
        d_stats = self.d_stats.copy()
        clone = GenomeStatistics(nc_proportion, intervals_between_loci, d_stats)
        return clone

    def compute(self, genome) -> None:
        self.nc_proportion = genome.z_nc / genome.length
        self.intervals_between_loci = np.sort(genome.loci_interval)
        self.d_stats = {
            "Non coding proportion": self.nc_proportion,
            "Non coding length list": self.intervals_between_loci,
        }
