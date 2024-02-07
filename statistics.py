import numpy as np

class Statistic:
    def __init__(self) -> None:
        pass

    def mean(self, array: np.ndarray):
        return array.mean()
    
    def variance(self, array: np.ndarray):
        array_length = len(array)
        return array.var() * array_length / (array_length - 1) # Bessel correction for unbiased estimation