from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsDifferenceRegressor:
    def __init__(self, train_series, k=3):
        self.train_series = train_series
        self.k = k

    def fit(self):
        pass

    def preprocessing(self):
        pass

    def distance(self, x, y):
        return euclidean_distances([x], [y])[0][0]

    # Difference of the time series. difference(t) = observation(t) - observation(t-1)
    def difference_series(self):
        return [self.train_series[i + 1] - self.train_series[i] for i in range(len(self.train_series) - 1)]

    def predict(self, step_size):
        # Calculates the difference series.
        difference = self.difference_series()
        to_predict = difference[-step_size:]
        window = difference[:-step_size]
        # Due to the division, some chunks can have a length different from step_size.
        chunks = [window[i:(i + step_size)] for i in range(len(window)) if len(window) >= (i + step_size)]
        # Filtering the chunks with length different from step_size
        distances = np.array(list(map(lambda x: euclidean_distances([x], [to_predict])[0][0], chunks)))
        k_nearest_chunks_indexes = np.argpartition(distances, range(min(len(chunks), self.k)))[:min(len(chunks), self.k)]
        k_nearest_chunks = [chunks[chunk_index] for chunk_index in k_nearest_chunks_indexes]

        mean_of_differences = np.mean(k_nearest_chunks, axis=0)
        inverted = np.array(self.train_series[-step_size:])
        predicted = [None] * step_size
        last_value = self.train_series[-1]
        for index, value in enumerate(mean_of_differences):
            predicted[index] = last_value + value
            last_value = predicted[index]
        return predicted



