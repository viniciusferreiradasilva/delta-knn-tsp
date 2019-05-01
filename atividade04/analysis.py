import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# stock = "NVDA"
# stock = "CHK"
# stock = "AES"
stock = "PCLN"

df = pd.read_csv('output/KNeighborsRegressor_' + stock + '.csv')
df = df[df['step'] == 1]
print(df.nsmallest(n=1, columns='mse'))

df = pd.read_csv('output/KNeighborsRegressor_' + stock + '.csv')
df = df[df['step'] == 3]
print(df.nsmallest(n=1, columns='mse'))

df = pd.read_csv('output/KNeighborsRegressor_' + stock + '.csv')
df = df[df['step'] == 5]
print(df.nsmallest(n=1, columns='mse'))

