from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsRegressorAVG:
    def __init__(self, train_series, k=3, comparison_range=5):
        self.train_series = train_series
        self.k = k
        self.comparison_range = comparison_range

    def fit(self):
        pass

    def distance(self, x, y):
        return euclidean_distances([x], [y])[0][0]

    def predict(self, step_size):
        # Ensures that the comparison_range is not lesser than the step size.
        comparison_range = max(self.comparison_range, step_size)
        # Separates the last comparison_range from the prediction window.
        to_predict = self.train_series[-comparison_range:]
        window = self.train_series[:-comparison_range]
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
        k_nearest_next_chunks = [self.train_series[last + 1:last + step_size + 1] for last in lasts]
        w_avg = np.average(k_nearest_next_chunks, axis=0, weights=range(self.k, 0, -1))
        w = [x**2 for x in list(range(self.k, 0, -1))]
        w_avg2 = np.average(k_nearest_next_chunks, axis=0, weights=w)
        return w_avg



