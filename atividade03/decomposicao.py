# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

series = pd.read_csv('input/all_stocks_5yr.csv', header=0).dropna()
# series = series.loc[series['Name'] == 'NVDA']['open'][1:100]
series = series.loc[series['Name'] == 'NVDA']['open'][1:200]
series.index = pd.DatetimeIndex(freq='B', start=0, periods=len(series - 1))
result = seasonal_decompose(series, model='multiplicative')
plt.figure(figsize=(12,6))
plt.subplots_adjust(hspace=.3, wspace=0.3)
plt.subplot(3, 2, 1)
plt.plot(range(len(result.trend)), result.trend, label='trend')
plt.plot(range(len(result.trend)), series, label='open')
plt.legend()
plt.title("Decomposição NVDA")
plt.subplot(3, 2, 3)
plt.plot(range(len(result.seasonal)), result.seasonal, label='sazonalidade')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(range(len(result.resid)), result.resid, label='resíduo')
plt.legend()

plt.subplot(4, 2, 2)
series = pd.read_csv('input/all_stocks_5yr.csv', header=0).dropna()
series = series.loc[series['Name'] == 'CHK']['open'][1:200]
series.index = pd.DatetimeIndex(freq='B', start=0, periods=len(series - 1))
result = seasonal_decompose(series, model='multiplicative')
plt.subplot(3, 2, 2)
plt.plot(range(len(result.trend)), result.trend, label='trend')
plt.plot(range(len(result.trend)), series, label='open')
plt.title("Decomposição CHK")
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(range(len(result.seasonal)), result.seasonal, label='sazonalidade')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(range(len(result.resid)), result.resid, label='resíduo')
plt.legend()

plt.savefig('decompose.eps', format='eps')