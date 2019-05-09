import numpy as np


class MovingAverage:
    def __init__(self, train_series, order=3):
        self.order = order
        self.train_series = train_series

    def fit(self):
        pass

    def preprocessing(self):
        pass

    def predict(self, step_size):
        average = np.mean(self.train_series[-self.order:])
        return [average] * step_size


