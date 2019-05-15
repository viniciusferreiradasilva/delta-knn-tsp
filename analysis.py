import pandas as pd
import numpy as np

df = pd.read_csv('../output/AutoRegression_NVDA.csv')
df = df[df['step'] == 1]
print(df.nsmallest(n=1, columns='mse'))
