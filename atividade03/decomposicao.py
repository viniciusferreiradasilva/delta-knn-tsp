# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

stock = 'NVDA'
series = pd.read_csv('input/all_stocks_5yr.csv', header=0).dropna()
# series = series.loc[series['Name'] == 'NVDA']['open'][1:100]
series = series.loc[series['Name'] == stock]['open'][1:500]
series.index = pd.DatetimeIndex(freq='B', start=0, periods=len(series - 1))
result = seasonal_decompose(series, model='multiplicative')
plt.figure(figsize=(12, 6))
plt.subplots_adjust(hspace=.3, wspace=0.3)
plt.subplot(3, 1, 1)
plt.plot(range(len(result.trend)), series, label='open', color='gray', linewidth=2)
plt.plot(range(len(result.trend)), result.trend, label='Trend')
plt.legend()
plt.title("Decomposição " + stock)
plt.subplot(3, 1, 2)
plt.plot(range(len(result.seasonal)), result.seasonal, label='Sazonalidade')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(range(len(result.resid)), result.resid, label='Resíduo')
plt.legend()

plt.savefig('output/' + stock + '_decompose.eps', format='eps')