from sklearn.metrics import mean_squared_error
import math
import numpy as np


# Calculates the mean squared error using the sklearn function.
def mse(y, predicted):
    return mean_squared_error(y, predicted)


# Returns a list containing all the absolute error between y and predicted.
def absolute_errors(y, predicted):
    return list(map(lambda i, j: math.fabs((i - j)), y, predicted))


# Returns a list containing all the absolute error between y and predicted.
def squared_errors(y, predicted):
    return list(map(lambda i, j: pow((i - j), 2), y, predicted))


# Calculates TU.
def tu(y, predicted):
    predictor = 0
    naive_predictor = 0
    for i in range(1, len(y)):
        predictor += (y[i] - predicted[i]) * (y[i] - predicted[i])
        naive_predictor += (y[i] - y[i - 1]) * (y[i] - y[i - 1])
    return predictor / naive_predictor


# Calculates POCID.
def pocid(y, predicted):
    value = 0
    for i in range(1, len(y)):
        if((predicted[i] - predicted[i - 1]) * (y[i] - y[i - 1])) > 0:
            value += 1
    return value / len(y)
