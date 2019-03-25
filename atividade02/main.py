# Implementação da atividade 02 de Aplicações de Algoritmos de Aprendizado de Máquina.

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

import argparse
import pandas as pd
import matplotlib.pyplot as plt


# Instancia o parser.
parser = argparse.ArgumentParser(description='Ferramenta de regressão.', formatter_class=argparse.RawTextHelpFormatter)

# Argumento para o arquivo de entrada.
parser.add_argument('--input', required=True, type=str,
                    help='Uma string que representa um .csv que contém os dados S&P500 para uma ação.')

# Argumento para o arquivo de entrada.
parser.add_argument('--field', type=str, default='open',
                    help='Uma string que representa para qual campo se quer a regressão.')

# Argumento para o arquivo de entrada.
parser.add_argument('--window', type=int, default=20,
                    help='um valor inteiro que representa o tamanho da janela que será utilizada para o treinamento'
                         'na predição.')


args = parser.parse_args()

# Carrega o arquivo .csv em um dataframe do pandas.
df = pd.read_csv(args.input).dropna()
print('dataframe with ', len(df), ' rows loaded.')
# Seta o nome do campo que será utilizado na regressão.
field = args.field
# Seta o tamanho das janelas que será utilizada na regressão.
window_size = args.window

# Recupera a série do dataframe de acordo com o nome fornecido como argumento.
series = df[field]

# Arrays que armazenam os valores reais e preditos.
y = [None] * (len(series) - window_size)
predicted = [None] * (len(series) - window_size)

for i in range(len(series) - window_size):
    # Recupera uma fatia da série temporal de acordo com o tamanho da janela.
    train_series = series[i:(i + window_size)].values
    # Cria um modelo de autoregressão.
    model = AR(train_series).fit()
    # Prediz o valor de i + window_size, que corresponde ao valor do passo seguinte.
    prediction = model.predict(start=len(train_series), end=(len(train_series)), dynamic=False)
    # Armazena o valor real.
    y[i] = series[i + window_size]
    # Armazena o valor predito.
    predicted[i] = prediction[0]


# Calcula o erro quadrático médio.
error = mean_squared_error(y, predicted)
# Plota o gráfico diferenciando os valores reais e os valores preditos.
plt.plot(range(len(predicted)), predicted, label='predicted')
plt.plot(range(len(y)), y, label='y')
plt.legend()
plt.title("Resultados de AR para " + (args.input.split('/')[1].split('.')[0]) + " (erro: " + str(error) + ")")
plt.xlabel("day")
plt.ylabel("open")
plt.show()