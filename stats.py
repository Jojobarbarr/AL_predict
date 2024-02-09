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

    def __str__(self) -> str:
        return (f"\n\nTotal mutations: {self.count}\n"
                f"Neutral mutations: {self.neutral_count}\n"
                f"Neutral mutations proportion: {self.neutral_mean()}\n"
                f"Length mean: {self.length_mean()}")
    
    def neutral_mean(self) -> float:
        return super().mean(self.neutral_count, self.count)
    
    def length_mean(self) -> float:
        return super().mean(self.length_sum, self.neutral_count)
