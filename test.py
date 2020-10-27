# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:17:36 2020

@author: Daniel
"""


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import ruptures as rpt
from sklearn import tree
from sklearn.metrics import mean_squared_error
from math import sqrt

#load the data, and clean time series values 
data  = pd.read_csv("data.csv", sep=";")
print(data.dtypes)
data.columns = data.columns.str.lower()

data["time_series_1"] = data["time_series_1"].str.replace(",", ".")
data["time_series_2"] = data["time_series_2"].str.replace(",", ".")
data["time_series_1"] = pd.to_numeric(data["time_series_1"])
data["time_series_2"] = pd.to_numeric(data["time_series_2"])

#data['year'], data['month'], data['day'] = data['posting_date'].str[:4], data['posting_date'].str[4:6],  data['posting_date'].str[6:8]
#data = data.astype({"year": "int", "month": "int", "day": "int"})



datelist  = pd.date_range(start="2015-01-01",end="2019-11-04").to_frame(index=False, name="date")


#now transform date in datetime and split away day, month & year
data = data.astype({"posting_date": "string"})
data["date"] = pd.to_datetime(data["posting_date"])


series1 = data.groupby('date', as_index=False).agg({"time_series_1": "sum"})
series1 = pd.merge(datelist, series1, on="date", how="left").fillna(0)
series1["day"], series1["month"],series1["year"] = series1["date"].dt.day, series1["date"].dt.month, series1["date"].dt.year
series1["day_of_week"] = series1["date"].dt.day_name()
series1 = series1.rename(columns={"time_series_1": "t"})


series2 = data.groupby('date', as_index=False).agg({"time_series_2": "sum"})
series2 = pd.merge(datelist, series2, on="date", how="left").fillna(0)
series2["day"], series2["month"],series2["year"] = series2["date"].dt.day, series2["date"].dt.month, series2["date"].dt.year
series2["day_of_week"] = series2["date"].dt.day_name()
series2 = series2.rename(columns={"time_series_2": "t"})


#lets work with series2 for now
#get lagged values
lags = pd.concat([series2["t"].shift(1), series2["t"].shift(2), series2["t"].shift(3), series2["t"].shift(4), series2["t"].shift(5)], axis=1)
series2["t-1"]= lags.iloc[:,0]
series2["t-2"]= lags.iloc[:,1]
series2["t-3"]= lags.iloc[:,2]
series2["t-4"]= lags.iloc[:,3]
series2["t-5"]= lags.iloc[:,4]

series2 = series2.iloc[5:1730]
series2 = series2.reset_index(drop=True)

#get variables in correct type
series2["day_of_week"] = series2["day_of_week"].astype('category')
series2["month"] = series2["month"].astype('category')
print(series2.dtypes)






#################################################
#trying out tree
train = series2.iloc[:1715]
test = series2.iloc[1715:]

clf = tree.DecisionTreeRegressor()

#get train data in correct form & fit tree
X = pd.get_dummies(train[['month','day_of_week']])
Temp_X = train.iloc[:,6:11]
X = pd.concat([X, Temp_X], axis=1, sort=False)
#print(X.dtypes)
y = train.iloc[:,1]
clf = clf.fit(X, y)

#get test data in correct form & predict
X = pd.get_dummies(test[['month','day_of_week']])
Temp_X = test.iloc[:,6:11]
X = pd.concat([X, Temp_X], axis=1, sort=False)
y = test.iloc[:,1]
y = y.reset_index()

pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
#result = pd.concat([y, pred_y], axis=1, sort=False)
rms = mean_squared_error(y["t"], pred_y, squared=False)
##########################################################








##### trying out ruptures 
algo = rpt.Pelt(model="ar", params={"order": 10}, min_size=5).fit(series2["t"].values)
my_bkps = algo.predict(pen=10000000)
fig, (ax,) = rpt.display(series2["t"].values, my_bkps, figsize=(10, 6))
plt.show()

algo = rpt.Pelt(model="normal", min_size=5).fit(series2["t"].values)
my_bkps = algo.predict(pen=1)
fig, (ax,) = rpt.display(series2["t"].values, my_bkps, figsize=(10, 6))
plt.show()

algo = rpt.Pelt(model="rbf", min_size=10).fit(series2["t"].values)
my_bkps = algo.predict(pen=3)
fig, (ax,) = rpt.display(series2["t"].values, my_bkps, figsize=(10, 6))
plt.show()

######################
#get concept features
#bkps have indices of breakpoints stored
list_concepts = []
count_rows = series2.shape[0] 
current_concept = 1
for x in range(1, count_rows+1):
    if (x in my_bkps): 
        current_concept+=1
    list_concepts.append(current_concept)
############################
    
#now try tree again:
series2["concept"] = list_concepts
series2["concept"] = series2["concept"].astype("category")

train = series2.iloc[:1715]
test = series2.iloc[1715:]

clf = tree.DecisionTreeRegressor()
#get train data in correct form & fit tree
X = pd.get_dummies(train[['month','day_of_week',"concept"]])
Temp_X = train.iloc[:,6:11]
X = pd.concat([X, Temp_X], axis=1, sort=False)
#print(X.dtypes)
y = train.iloc[:,1]
clf = clf.fit(X, y)

#get test data in correct form & predict
X = pd.get_dummies(test[['month','day_of_week',"concept"]])
Temp_X = test.iloc[:,6:11]
X = pd.concat([X, Temp_X], axis=1, sort=False)
y = test.iloc[:,1]
y = y.reset_index()

pred_y = pd.DataFrame(clf.predict(X)).rename(columns={0: "pred_y"})
#result = pd.concat([y, pred_y], axis=1, sort=False)
rms_with_concepts = mean_squared_error(y["t"], pred_y, squared=False)
























#here plotting the data


ArithmeticErrorfig, ax = plt.subplots()
ax.plot(series1["date"],series1["time_series_1"])
ax.ticklabel_format(useOffset=False, style='plain', axis="y")
#ax.get_yaxis().set_major_formatter(
    #plt.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()


ArithmeticErrorfig, ax = plt.subplots()
ax.plot(series2["date"],series2["time_series_2"])
ax.ticklabel_format(useOffset=False, style='plain', axis="y")
#ax.get_yaxis().set_major_formatter(
    #plt.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()


plt.plot(series1["date"], series1["time_series_1"])
plt.plot(series2)




#ARIMA/ARMA STUFF
plot_acf(series2["time_series_2"]) 
train_model = ARIMA(series2["time_series_2"], order=(2, 0, 2))
fit_model = train_model.fit()
print(fit_model.summary())

fit_model.plot_predict(dynamic=False)
plt.show()