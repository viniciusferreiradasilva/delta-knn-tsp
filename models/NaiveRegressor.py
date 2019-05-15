
class NaiveRegressor:
    def __init__(self, train_series, p):
        self.train_series = train_series

    def fit(self):
        pass

    def preprocessing(self):
        pass

    def predict(self, step_size):
        return self.train_series[-step_size:]


