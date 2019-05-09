from statsmodels.tsa.ar_model import AR


class AutoRegression:
    def __init__(self, train_series):
        self.train_series = train_series
        self.ar = AR(train_series)

    def fit(self):
        self.ar = self.ar.fit(disp=0)

    def preprocessing(self):
        pass

    def predict(self, step_size):
        return self.ar.predict(start=len(self.train_series), end=(len(self.train_series) + step_size - 1),
                               dynamic=False)


