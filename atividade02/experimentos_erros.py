# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

for i in range(0, (len(series) - window_size), step_size):
    # Recupera uma fatia da série temporal de acordo com o tamanho da janela.
    train_series = series[i:(i + window_size)].values
    # Cria um modelo de autoregressão.
    model = AR(train_series).fit()
    # Prediz os step_size passos na série temporal.
    prediction = model.predict(start=len(train_series), end=(len(train_series) + step_size - 1), dynamic=False)
    # Armazena o valor predito.
    predicted[i:(i + step_size)] = prediction
    # print("lag:", model.k_ar)

y = series[window_size:]
predicted = predicted[:len(y)]

# Plota o gráfico diferenciando os valores reais e os valores preditos.
plt.subplot(2, 1, 1)
plt.plot(range(len(predicted)), predicted, label='predito')
plt.plot(range(len(y)), y, label='y')
plt.legend()
plt.title("Resultados de AutoRegressão para " + args.stock_name)
plt.xlabel("day")
plt.ylabel("open")

# Calcula o erro quadrático médio.
mean_square_error = mean_squared_error(y.values, predicted)
# Plota o gráfico do erro quadratico para cada ponto.
plt.subplot(2, 1, 2)
errors = list(map(lambda x, y: (x - y) * (x - y), y, predicted))
plt.plot(range(len(errors)), errors, label='erro')
plt.legend()
plt.title("Erros na AutoRegressão para " + args.stock_name + " (erro médio = " + '{:.3f}'.format(mean_square_error) + ")")
plt.xlabel("day")
plt.ylabel("erro")
plt.subplots_adjust(hspace=.5)
plt.show()
# plt.savefig(args.stock_name+'_main.eps', format='eps')
