from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsRegressor:
    def __init__(self, train_series, k=1):
        self.train_series = train_series
        self.k = k

    def fit(self):
        pass

    def preprocessing(self):
        pass

    def distance(self, x, y):
        return euclidean_distances([x], [y])[0][0]

    def predict(self, step_size):
        to_predict = self.train_series[-step_size:]
        window = self.train_series[:-step_size]
        # Due to the division, some chunks can have a length different from step_size.
        chunks = [window[i:(i + step_size)] for i in range(len(window)) if len(window) >= (i + step_size)]
        # Filtering the chunks with length different from step_size
        distances = np.array(list(map(lambda x: mean_squared_error(x, to_predict), chunks)))
        k_nearest_chunks_indexes = np.argpartition(distances, range(min(len(chunks), self.k)))[:min(len(chunks), self.k)]
        k_nearest_chunks = [chunks[chunk_index] for chunk_index in k_nearest_chunks_indexes]
        return np.mean(k_nearest_chunks, axis=0)



