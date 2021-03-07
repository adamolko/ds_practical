# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:23:02 2020

@author: Daniel
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
#to dataframe to work with
series2_cleaned = pd.read_pickle('series2_cleaned.pkl') 


#################################################
#trying out tree without concept features
train = series2_cleaned.iloc[:1700]
test = series2_cleaned.iloc[1700:]

clf = tree.DecisionTreeRegressor()

X = train.iloc[:,1:27]
y = train.iloc[:,0]

clf = clf.fit(X, y)

#get test data in correct form & predict
X = test.iloc[:,1:27]
y = test.iloc[:,0]
y = y.reset_index()

pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
#result = pd.concat([y, pred_y], axis=1, sort=False)
rms = mean_squared_error(y["t"], pred_y, squared=False)
##########################################################











#################################################
#trying out tree without concept features
train = series2_cleaned.iloc[:1700]
test = series2_cleaned.iloc[1700:]

clf = tree.DecisionTreeRegressor()

X = train.iloc[:,1:]
y = train.iloc[:,0]

clf = clf.fit(X, y)

#get test data in correct form & predict
X = test.iloc[:,1:]
y = test.iloc[:,0]
y = y.reset_index()

pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
#result = pd.concat([y, pred_y], axis=1, sort=False)
rms_with_concepts = mean_squared_error(y["t"], pred_y, squared=False)
##########################################################






################################################
#now try a recursive window for training/testing
#For now let's only use last 30 observations for testing & only predict one-step ahead each time
count_rows = series2_cleaned.shape[0] 
list_rmse = []
for i in range(10, 0, -1):
    index = count_rows - i
    train = series2_cleaned.iloc[:index]
    test = series2_cleaned.iloc[[index],:]
    
    clf = tree.DecisionTreeRegressor()
    #get train data in correct form & fit tree
    X = train.iloc[:,1:]
    #print(X.dtypes)
    y = train.iloc[:,0]
    clf = clf.fit(X, y)
    
    #get test data in correct form & predict
    X = test.iloc[:,1:]
    y = test.iloc[:,0]
    y = y.reset_index()
    
    pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
    #result = pd.concat([y, pred_y], axis=1, sort=False)
    current_rmse = mean_squared_error(y["t"], pred_y, squared=False)
    list_rmse.append(current_rmse)

print(sum(list_rmse) / len(list_rmse))



count_rows = series2_cleaned.shape[0] 
list_rmse = []
for i in range(10, 0, -1):
    index = count_rows - i
    train = series2_cleaned.iloc[:index]
    test = series2_cleaned.iloc[[index],:]
    
    clf = tree.DecisionTreeRegressor()
    #get train data in correct form & fit tree
    X = train.iloc[:,1:27]
    #print(X.dtypes)
    y = train.iloc[:,0]
    clf = clf.fit(X, y)
    
    #get test data in correct form & predict
    X = test.iloc[:,1:27]
    y = test.iloc[:,0]
    y = y.reset_index()
    
    pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
    #result = pd.concat([y, pred_y], axis=1, sort=False)
    current_rmse = mean_squared_error(y["t"], pred_y, squared=False)
    list_rmse.append(current_rmse)

print(sum(list_rmse) / len(list_rmse))


























#ARIMA/ARMA STUFF
# =============================================================================
# plot_acf(series2["time_series_2"]) 
# train_model = ARIMA(series2["time_series_2"], order=(2, 0, 2))
# fit_model = train_model.fit()
# print(fit_model.summary())
# 
# fit_model.plot_predict(dynamic=False)
# plt.show()
# =============================================================================
