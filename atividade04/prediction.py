# Implementação da atividade 04 de Aplicações de Algoritmos de Aprendizado de Máquina.
from models.AutoRegression import AutoRegression
from models.MovingAverage import MovingAverage
from models.KNeighborsRegressor import KNeighborsRegressor
from models.KNeighborsRegressorAVG import KNeighborsRegressorAVG
import measures
import math

import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Ferramenta de regressão.', formatter_class=argparse.RawTextHelpFormatter)
# Input file argument.
parser.add_argument('--input', required=True, type=str,
                    help='Uma string que representa um .csv que contém os dados S&P500 para uma ação.')
# Stock name argument.
parser.add_argument('--stock_name', required=True, type=str,
                    help='Uma string que representa o nome da ação.')
# From index argument.
parser.add_argument('--from_index', type=int, default=0,
                    help='Índice do dataset que representa desde onde vai a predição.')
# To index argument.
parser.add_argument('--to_index', type=int,
                    help='Índice do dataset que representa até onde vai a predição.')
# Data field argument.
parser.add_argument('--field', type=str, default='open',
                    help='Uma string que representa para qual campo se quer a regressão.')
# Window size argument.
parser.add_argument('--window', type=int, default=20,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')
# Horizon size argument.
parser.add_argument('--steps', type=int, default=5,
                    help='um valor inteiro que representa o número de passos futuros que serão previstos.')
# Regressor choice argument.
parser.add_argument('--regressor', type=int, default=0,
                    help='um valor inteiro que representa qual regressor será utilizado na predição:\n'
                         '0 - Autoregressão\n'
                         '1 - Média Móvel\n'
                         '2 - kNN\n'
                         '3 - kNN modificado.'
                         '4 - kNN Sliding Window')
# Regressor arguments args.
parser.add_argument('--args', required=False, nargs='+', default=None,
                    help='Lista de parâmetros para o algoritmo de regressão. Cada algoritmo necessita de parâmetros'
                         'diferentes para a sua execução: --args arg_1 arg_2 ... arg_n')

args = parser.parse_args()
# Loads the .csv into a pandas dataframe.
df_full = pd.read_csv(args.input).dropna()
# Select rows with the wanted name.
df = df_full.loc[df_full['Name'] == args.stock_name]
# Sets where the prediction begins.
from_index = args.from_index
# Sets where the prediction ends.
if args.to_index:
    to_index = args.to_index
else:
    to_index = len(df)
# Sets which field will be used.
field = args.field
# Sets the size of the windows.
window_size = args.window
# Sets the size of the steps (or horizon).
step_size = args.steps
# Retrieves the time series of the field.
series = df[from_index:to_index][field]
# Series values.
y = [None] * (len(series) - window_size)
# Predicted series values.
predicted = [None] * (len(series) - window_size)
regressor = [AutoRegression, MovingAverage, KNeighborsRegressor, KNeighborsRegressorAVG][args.regressor]
# Retrieves the regression algorithm parameters.
if args.args:
    algorithm_args = tuple(map(int, args.args))
else:
    algorithm_args = None

for i in range(0, (len(series) - window_size), step_size):
    # Retrieves the slice of the series according to the window.
    train_series = series[i:(i + window_size)].values
    # Creates the model.
    model = regressor(train_series)
    model.fit()
    # Predict the step_size horizon in the series.
    prediction = model.predict(step_size)
    predicted[i:(i + step_size)] = prediction

y = series[window_size:].values
predicted = predicted[:len(y)]

plt.subplot(2, 1, 1)
plt.plot(range(len(predicted)), predicted, label='predito')
plt.plot(range(len(y)), y, label='y')
plt.legend()
plt.title("Predição de " + regressor.__name__ + " para " + args.stock_name)
plt.xlabel("day")
plt.ylabel("open")

# Error metrics.
print('mean_square_error:', measures.mse(y, predicted))
print('TU:', measures.tu(y, predicted))
print('pocid:', measures.pocid(y, predicted))
errors = measures.absolute_errors(y, predicted)

# Plots the error graph.
plt.subplot(2, 1, 2)
plt.plot(range(len(errors)), errors, label='erro')
plt.legend()
plt.title("Erros de " + regressor.__name__ + " para " + args.stock_name + " (erro médio = " + '{:.3f}'.format(measures.mse(y, predicted)) + ")")
plt.xlabel("day")
plt.ylabel("erro")
plt.subplots_adjust(hspace=.5)
plt.show()
# plt.savefig('output/' + regressor.__name__ + '_' + args.stock_name+'.eps', format='eps')
