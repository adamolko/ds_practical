# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:37:56 2020

@author: Daniel



testing stuff on data from M3 competition, to see if our data is the problem
"""



import graphviz 
import pylab
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn import tree
import ruptures as rpt
from sklearn.metrics import mean_squared_error
import functions
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


#Get the monthly data from the M3 competition (lots of paper online on that data)
data  = pd.read_excel("data/M3C.xls", sheet_name=2)

#for now just choose one of those timeseries
series = data.iloc[657,:].to_frame()
#only keep values
series = series.iloc[6:,:]
series = series.rename(columns={657: "t"})
series = series.dropna()

series["t"] = pd.to_numeric(series["t"])

lags = pd.concat([series["t"].shift(1), series["t"].shift(2), series["t"].shift(3)], axis=1)
series["t-1"]= lags.iloc[:,0]
series["t-2"]= lags.iloc[:,1]
series["t-3"]= lags.iloc[:,2]
series["t-4"]= lags.iloc[:,2]
series["t-5"]= lags.iloc[:,2]
series["t-6"]= lags.iloc[:,2]
series = series[6:]

series = series.iloc[:,0:5]
#series["intercept"] = 1
series = series.reset_index(drop=True)


signal = series.to_numpy()

algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(series.to_numpy())
my_bkps = algo.predict(pen=10000000000000)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(series.to_numpy())
my_bkps = algo.predict(pen=3)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()


series = functions.transform_bkps_to_features(my_bkps, series)


dummies_concept = pd.get_dummies(series[['concept']])
series = series.iloc[:,0:3]
series = pd.concat([series, dummies_concept], axis=1, sort=False)






count_rows = series.shape[0] 
list_rmse = []
for i in range(10, 0, -1):
    index = count_rows - i
    train = series.iloc[:index]
    test = series.iloc[[index],:]
    
    clf = tree.DecisionTreeRegressor(max_depth=5)
    #get train data in correct form & fit tree
    X = series.iloc[:,1:3]
    #print(X.dtypes)
    y = series.iloc[:,0]
    clf = clf.fit(X, y)
    
    #get test data in correct form & predict
    X = test.iloc[:,1:3]
    y = test.iloc[:,0]
    y = y.reset_index()
    
    pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
    #result = pd.concat([y, pred_y], axis=1, sort=False)
    current_rmse = mean_squared_error(y["t"], pred_y, squared=False)
    list_rmse.append(current_rmse)

print(sum(list_rmse) / len(list_rmse))

dot_data = tree.export_graphviz(clf, out_file=None,                
                      filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render(directory= "results", filename="no_concepts", format="png")


count_rows = series.shape[0] 
list_rmse = []
for i in range(10, 0, -1):
    index = count_rows - i
    train = series.iloc[:index]
    test = series.iloc[[index],:]
    
    clf = tree.DecisionTreeRegressor(max_depth=5)
    #get train data in correct form & fit tree
    X = series.iloc[:,1:]
    #print(X.dtypes)
    y = series.iloc[:,0]
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








dot_data = tree.export_graphviz(clf, out_file=None,                
                      filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render(directory= "results", filename="with_concepts", format="png")















plot_acf(series["t"]) 
train_model = ARIMA(series["t"], order=(10, 0, 0))
fit_model = train_model.fit()
print(fit_model.summary())
# 
# fit_model.plot_predict(dynamic=False)
# plt.show()