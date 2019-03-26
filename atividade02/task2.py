# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:39:44 2019

@author: Lucas
"""

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

listyear = list(range(2013,2019))
listmonth = list(range(1,13))

df = pd.read_csv('all_stocks_5yr.csv')


            
first = 20

linregression = []
y = []


df2 = df.query("Name == 'NVDA'")[['open','volume', 'high', 'low', 'close']]

            
for act in range(first, len(df2)):
    a = df2[:act]
    
    b = a[['open','volume', 'high', 'low']]
    c = a['close']
    
    X_train = b[:act-1]#.values()
    y_train = c[:act-1]#.values()
    
    X_test = b[act-1:act]#.values()
    y_test = c[act-1:act]#.values()
    
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
        
    y_prediction = regr.predict(X_test)
    
    print(act, 'previsao lin', y_prediction[0], 'y', y_test.values[0])
    
    linregression.append(y_prediction[0])
    y.append(y_test.values[0])
    
    

k = list(range(len(y)))

plt.plot(k, linregression, label='predicted')    
plt.plot(k, y, label='y')

plt.xlabel('day')
plt.ylabel('close')

mse = mean_squared_error(y, linregression)

plt.title('Resultados de LR para NVDA, MSE: %f' % (mse) )

plt.legend()

    
plt.show()
    
