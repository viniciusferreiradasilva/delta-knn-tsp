from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KNeighborsRegressorAVG:
    def __init__(self, train_series, k=3):
        self.train_series = train_series
        self.k = k

    def fit(self):
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
        
        w_avg = np.average(k_nearest_chunks, axis=0, weights=range(self.k, 0, -1))
        
        w = [x**2 for x in list(range(self.k, 0, -1))] 
        
        w_avg2 = np.average(k_nearest_chunks, axis=0, weights=w)

        #print(w_avg, np.mean(k_nearest_chunks, axis=0))
        
        return w_avg



