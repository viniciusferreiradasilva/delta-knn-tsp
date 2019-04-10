# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Instancia o parser.
parser = argparse.ArgumentParser(description='Ferramenta de regressão.', formatter_class=argparse.RawTextHelpFormatter)

# Argumento para o arquivo de entrada.
parser.add_argument('--input', required=True, type=str,
                    help='Uma string que representa um .csv que contém os dados S&P500 para uma ação.')

# Argumento para o arquivo de entrada.
parser.add_argument('--stock_name', required=True, type=str,
                    help='Uma string que representa o nome da ação.')

# Argumento para o arquivo de entrada.
parser.add_argument('--from_index', type=int, default=0,
                    help='Índice do dataset que representa desde onde vai a predição.')

# Argumento para o arquivo de entrada.
parser.add_argument('--to_index', type=int,
                    help='Índice do dataset que representa até onde vai a predição.')

# Argumento para o arquivo de entrada.
parser.add_argument('--field', type=str, default='open',
                    help='Uma string que representa para qual campo se quer a regressão.')

# Argumento para o arquivo de entrada.
parser.add_argument('--from_window', type=int, default=10,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')

# Argumento para o arquivo de entrada.
parser.add_argument('--to_window', type=int, default=20,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')

# Required clustering algorithm configs.
parser.add_argument('--steps', required=True, nargs='+', default=[1, 5, 10],
                    help='Lista dos steps ao qual se deseja testar. --configs step1 step2 step3 ...')


args = parser.parse_args()

# Carrega o arquivo .csv em um dataframe do pandas.
df_full = pd.read_csv(args.input).dropna()
# Seleciona apenas as linhas que contém o nome da ação passado como parâmetro.
df = df_full.loc[df_full['Name'] == args.stock_name]

print('dataframe com ', len(df), ' linhas carregadas.')
# Seta onde começa a predição.
from_index = args.from_index
# Seta até onde vai a predição.
if(args.to_index):
    to_index = args.to_index
else:
    to_index = len(df)
# Seta o nome do campo que será utilizado na regressão.
field = args.field
# Recupera a série do dataframe de acordo com o nome fornecido como argumento.
series = df[from_index:to_index][field]

for step in list(map(int, args.steps)):
    mean_square_errors = np.empty(args.to_window - args.from_window)
    for index, window_size in enumerate(range(args.from_window, args.to_window)):
        # print('Computando ', (index + 1), ' de ', args.to_window - args.from_window)
        # Arrays que armazenam os valores reais.
        y = [None] * (len(series) - window_size)
        # Arrays que armazenam os valores preditos.
        predicted = [None] * (len(series) - window_size)
        for i in range(0, (len(series) - window_size), step):
            # Recupera uma fatia da série temporal de acordo com o tamanho da janela.
            train_series = series[i:(i + window_size)].values

            # Cria um modelo de autoregressão.
            model = AR(train_series).fit()
            # Prediz os step passos na série temporal.
            prediction = model.predict(start=len(train_series), end=(len(train_series) + step - 1), dynamic=False)
            # Armazena o valor predito.
            predicted[i:(i + step)] = prediction

        y = series[window_size:]
        predicted = predicted[:len(y)]
        mean_square_errors[index] = mean_squared_error(y.values, predicted)

    print("melhor janela:", (args.from_window + np.argmin(mean_square_errors)), " erro:", np.min(mean_square_errors))
    print("pior janela:", (args.from_window + np.argmax(mean_square_errors)), "erro:", np.max(mean_square_errors))
    print("mean:", np.mean(mean_square_errors))
    print("std:", np.std(mean_square_errors))
    print("*" * 50)
    plt.plot(range(args.from_window, args.to_window), mean_square_errors, label='erros para step=' + str(step))

plt.legend()
plt.title("Erro quadrático médio por janela")
plt.xlabel("Tamanho da Janela")
plt.ylabel("Valor de MSE")
plt.ylim(0, np.mean(series))
# plt.xlim(args.from_window, 100)
plt.savefig('output/' + args.stock_name+'.eps', format='eps')
# plt.show()
