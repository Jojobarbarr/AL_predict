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
    
    def variance(self, square_sum: int | float, population: int | float, mean: float) -> float:
        if population <= 1:
            print("Empty or one individual population.")
            return 0
        return square_sum / (population - 1) - mean ** 2

class MutationStatistics(Statistics):
    def __init__(self):
        self.count = 0
        self.neutral_count = 0
        self.length_sum = 0
        self.length_square_sum = 0
        self.d_stats = {}

    def __str__(self) -> str:
        return str(self.d_stats)
    
    def compute(self, theory: float=0):
        neutral_mean = self.neutral_mean()
        neutral_std = self.neutral_variance(neutral_mean)
        neutral_mean_std = self.mean_estimator_variance(neutral_std, self.count)
        lenght_mean = self.length_mean()
        length_std = self.length_variance(lenght_mean)
        length_mean_std = self.mean_estimator_variance(length_std, self.neutral_count)
        self.d_stats = {
            "Total mutations": self.count,
            "Neutral mutations": self.neutral_count,
            "Neutral mutations proportion": neutral_mean,
            "Neutral mutations standard deviation of proportion estimator": neutral_mean_std,
            "Neutral probability theory": theory,
            "Length mean": lenght_mean,
            "Length standard deviation of mean estimator": length_mean_std,
            "Length standard deviation": length_std ** 0.5,
        }

    def neutral_mean(self) -> float:
        return super().mean(self.neutral_count, self.count)
        
    
    def neutral_variance(self, mean: float) -> float:
        return super().variance(self.neutral_count, self.count, mean)
        

    def length_mean(self) -> float:
        return super().mean(self.length_sum, self.neutral_count)
    
    def length_variance(self, mean: float) -> float:
        return super().variance(self.length_square_sum, self.neutral_count, mean)
