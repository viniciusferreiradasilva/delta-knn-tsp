from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsDifferenceRegressor:
    def __init__(self, train_series, k=3, comparison_range=1):
        self.train_series = train_series
        self.k = k
        self.comparison_range = comparison_range

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
        # Ensures that the comparison_range is not lesser than the step size.
        comparison_range = max(self.comparison_range, step_size)
        # Separates the last comparison_range from the prediction window.
        to_predict = difference[-comparison_range:]
        window = difference[:-comparison_range]
        # Divides the window into chunks with size equals to the self.comparison_range.
        chunks = [window[i:(i + comparison_range)] for i in range(len(window) - comparison_range + 1)]
        # Calculates the distances between the chunks and the tp_predict array.
        distances = np.array(list(map(lambda x: self.distance(x, to_predict), chunks)))
        # Selects the indexes of the k nearest chunks.
        k_nearest_chunks_indexes = np.argpartition(
            distances, range(min(len(chunks), self.k)))[:min(len(chunks), self.k)]
        # Calculates the indexes of the lasts elements of the chunks.
        lasts = [chunk_index + comparison_range - 1 for chunk_index in k_nearest_chunks_indexes]
        # Finds the next chunks of the nearest chunks to the to_predict horizon.
        k_nearest_next_chunks = [difference[last + 1:last + step_size + 1] for last in lasts]
        mean_of_differences = np.mean(k_nearest_next_chunks, axis=0)
        predicted = [None] * step_size
        last_value = self.train_series[-1]
        for index, value in enumerate(mean_of_differences):
            predicted[index] = last_value + value
            last_value = predicted[index]
        return predicted




