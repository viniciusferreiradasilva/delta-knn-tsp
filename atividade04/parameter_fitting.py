# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import measures
from models.AutoRegression import AutoRegression
from models.MovingAverage import MovingAverage
from models.KNeighborsRegressor import KNeighborsRegressor
from models.KNeighborsRegressorAVG import KNeighborsRegressorAVG
from models.KNeighborsDifferenceRegressor import KNeighborsDifferenceRegressor

# Instancia o parser.
parser = argparse.ArgumentParser(description='Ferramenta de ajuste do tamanho da janela de regressão.', formatter_class=argparse.RawTextHelpFormatter)

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

# From window arg.
parser.add_argument('--from_window', type=int, default=10,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')

# To window arg.
parser.add_argument('--to_window', type=int, default=20,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')

# From window arg.
parser.add_argument('--from_parameter', type=int, default=3,
                    help='um valor inteiro que representa de onde começa parâmetro para o algoritmo de predição.')

# To window arg.
parser.add_argument('--to_parameter', type=int, default=3,
                    help='um valor inteiro que representa até onde vai o parâmetro para o algoritmo de predição.')

# Prediction horizon steps.
parser.add_argument('--steps', required=False, nargs='+', default=[1, 3, 5],
                    help='Lista dos steps ao qual se deseja testar. --steps step1 step2 step3 ...')

# Regressor choice argument.
parser.add_argument('--regressor', type=int, default=0,
                    help='um valor inteiro que representa qual regressor será utilizado na predição:\n'
                         '0 - Autoregressão\n'
                         '1 - Média Móvel\n'
                         '2 - kNN.')
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
# Retrieves the time series of the field.
series = df[from_index:to_index][field]
regressor = [AutoRegression, MovingAverage, KNeighborsRegressor, KNeighborsRegressorAVG,
             KNeighborsDifferenceRegressor][args.regressor]
output_file = 'output/' + regressor.__name__ + '_' + args.stock_name+'.csv'
f = open(output_file, 'w')
print("Executando ", regressor.__name__, "em", args.stock_name)
f.write('step,window,parameter,mse,tu,pocid\n')
for step in list(map(int, args.steps)):
    for parameter in range(args.from_parameter, args.to_parameter + 1):
        tus = np.empty(args.to_window - args.from_window)
        for index, window_size in enumerate(range(args.from_window, args.to_window)):
            if window_size - step > step:
                y = [None] * (len(series) - window_size)
                predicted = [None] * (len(series) - window_size)
                for i in range(0, (len(series) - window_size), step):
                    train_series = series[i:(i + window_size)].values
                    # Creates the model.
                    model = regressor(train_series, *[parameter])
                    model.fit()
                    # Predict the step_size horizon in the series.
                    prediction = model.predict(step)
                    predicted[i:(i + step)] = prediction
                y = series[window_size:]
                predicted = predicted[:len(y)]
                f.write(str(step) + ',' + str(window_size) + ',' + str(parameter) + ',' +
                        str(measures.mse(y.values, predicted)) + ',' + str(measures.tu(y.values, predicted)) + ',' +
                        str(measures.pocid(y.values, predicted)) + '\n')
f.close()
