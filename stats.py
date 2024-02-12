class Statistics:
    def __init__(self) -> None:
        pass
    
    def mean(self, sum: int | float, population: int | float) -> float:
        return sum / population

class MutationStatistics(Statistics):
    def __init__(self):
        self.count = 0
        self.neutral_count = 0
        self.length_sum = 0
        self.d_stats = {}

    def __str__(self) -> str:
        return str(self.d_stats)
    
    def compute(self):
        self.d_stats = {
            "Total mutations": self.count,
            "Neutral mutations": self.neutral_count,
            "Neutral mutations proportion": self.neutral_mean(),
            "Length mean": self.length_mean(),
        }

    def neutral_mean(self) -> float:
        return super().mean(self.neutral_count, self.count)
    
    def length_mean(self) -> float:
        return super().mean(self.length_sum, self.neutral_count)
