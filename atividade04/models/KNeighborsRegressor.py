from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsRegressor:
    def __init__(self, train_series, k=3):
        self.train_series = train_series
        self.k = k

    def fit(self):
        pass

    def distance(self, x, y):
        return euclidean_distances([x], [y])[0][0]

    def predict(self, step_size):
        to_predict = self.train_series[-step_size:]
        f = lambda x, y: x + (x % y) if (x + (x % y) % y) == 0 else x - (x % y)
        window = self.train_series[:-step_size]
        window = window[:f(len(window), step_size)]
        # Due to the division, some chunks can have a length different from step_size.
        chunks = [chunk for chunk in np.split(window, len(window) / step_size) if len(chunk) == step_size]
        # Filtering the chunks with length different from step_size
        distances = np.array(list(map(lambda x: mean_squared_error(x, to_predict), chunks)))
        k_nearest_chunks_indexes = np.argpartition(distances, range(min(len(chunks), self.k)))[:min(len(chunks), self.k)]
        k_nearest_chunks = [chunks[chunk_index] if (chunk_index + 1 == len(chunks)) else chunks[chunk_index + 1]
                            for chunk_index in k_nearest_chunks_indexes]
        return np.mean(k_nearest_chunks, axis=0)



