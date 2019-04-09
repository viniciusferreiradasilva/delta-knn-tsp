# Implementação da atividade 03 de Aplicações de Algoritmos de Aprendizado de Máquina.
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

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
parser.add_argument('--window', type=int, default=20,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')

# Argumento para o arquivo de entrada.
parser.add_argument('--steps', type=int, default=5,
                    help='um valor inteiro que representa o número de passos futuros que serão previstos.')

# Argumento para o modelo de regressão.
parser.add_argument('--alg', type=int, default=0,
                    help='Um valor inteiro que representa qual algoritmo deve ser usado. 0 - AR\n1 - ARIMA\n2 - SARIMA.')

args = parser.parse_args()

# Carrega o arquivo .csv em um dataframe do pandas.
df_full = pd.read_csv(args.input).dropna()

# Seleciona apenas as linhas que contém o nome da ação passado como parâmetro.
df = df_full.loc[df_full['Name'] == args.stock_name]

# Seta onde começa a predição.
from_index = args.from_index

# Seta até onde vai a predição.
if(args.to_index):
    to_index = args.to_index
else:
    to_index = len(df)
# Seta o nome do campo que será utilizado na regressão.
field = args.field

# Seta o tamanho das janelas que será utilizada na regressão.
window_size = args.window

# Seta o tamanho do passo que será predito.
step_size = args.steps

# Recupera a série do dataframe de acordo com o nome fornecido como argumento.
series = df[from_index:to_index][field]

# Arrays que armazenam os valores reais.
y = [None] * (len(series) - window_size)

# Arrays que armazenam os valores preditos.
predicted = [None] * (len(series) - window_size)

# Recupera o algoritmo que vai ser utilizado na regressão.
algorithm = [AR, ARIMA][args.alg]


for i in range(0, (len(series) - window_size), step_size):
    # Recupera uma fatia da série temporal de acordo com o tamanho da janela.
    train_series = series[i:(i + window_size)].values

    # Cria um modelo de autoregressão.
    # model = algorithm(train_series)
    # model = algorithm(train_series, order=(1, 0, 0))
    model = ARIMA(train_series, order=(1, 0, 0))
    # Ajusta o modelo de acordo com a faixa da série.
    model = model.fit(disp=0)

    # Prediz os step_size passos na série temporal.
    prediction = model.predict(start=len(train_series), end=(len(train_series) + step_size - 1), dynamic=False)
    # Armazena o valor predito.
    predicted[i:(i + step_size)] = prediction
    # print("lag:", model.k_ar)

y = series[window_size:].values
predicted = predicted[:len(y)]

# Calcula o erro quadrático médio.
mean_square_error = mean_squared_error(y, predicted)

# Calcula a lista contendo os erros absolutos.
absolute_errors = list(map(lambda x, y: math.fabs((x - y)), y, predicted))

# Calcula o TU.
predictor = 0
naive_predictor = 0
for i in range(1, len(y)):
    predictor += (y[i] - predicted[i]) * (y[i] - predicted[i])
    naive_predictor += (y[i] - y[i - 1]) * (y[i] - y[i - 1])
tu = predictor / naive_predictor

# Calcula o POCID.
pocid = 0
for i in range(1, len(y)):
    if((predicted[i] - predicted[i - 1]) * (y[i] - y[i - 1])) > 0:
        pocid += 1
pocid /= len(y)

# Plota o gráfico diferenciando os valores reais e os valores preditos.
plt.subplot(2, 1, 1)
plt.plot(range(len(predicted)), predicted, label='predito')
plt.plot(range(len(y)), y, label='y')
plt.legend()
plt.title("Resultados de AutoRegressão para " + args.stock_name)
plt.xlabel("day")
plt.ylabel("open")

# Métricas de erro.
print('mean_square_error:', mean_square_error)
print('TU:', tu)
print('pocid:', pocid)

# Plota o gráfico do erro quadratico para cada ponto.
plt.subplot(2, 1, 2)
plt.plot(range(len(absolute_errors)), absolute_errors, label='erro')
plt.legend()
plt.title("Erros na AutoRegressão para " + args.stock_name + " (erro médio = " + '{:.3f}'.format(mean_square_error) + ")")
plt.xlabel("day")
plt.ylabel("erro")
plt.subplots_adjust(hspace=.5)
plt.show()
# plt.savefig('output/' + args.stock_name+'_main.eps', format='eps')
