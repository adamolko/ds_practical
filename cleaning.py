# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:17:36 2020

@author: Daniel
"""


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import math

import ruptures as rpt

import functions

#load the data, and clean time series values 
data  = pd.read_csv("data.csv", sep=";")
print(data.dtypes)
data.columns = data.columns.str.lower()

data["time_series_1"] = data["time_series_1"].str.replace(",", ".")
data["time_series_2"] = data["time_series_2"].str.replace(",", ".")
data["time_series_1"] = pd.to_numeric(data["time_series_1"])
data["time_series_2"] = pd.to_numeric(data["time_series_2"])


#according to forecasting paper
#   Statistical and Machine Learning forecasting methods: Concerns and ways forward
#we should apply a log or Box-Cox transformation for best forecasting results (both available in python)
#BUT: we have negative values, so how should we deal with that?

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
lags = pd.concat([series2["t"].shift(1), series2["t"].shift(2), series2["t"].shift(3),
                  series2["t"].shift(4), series2["t"].shift(5), series2["t"].shift(6), 
                  series2["t"].shift(7)], axis=1)
series2["t-1"]= lags.iloc[:,0]
series2["t-2"]= lags.iloc[:,1]
series2["t-3"]= lags.iloc[:,2]
series2["t-4"]= lags.iloc[:,3]
series2["t-5"]= lags.iloc[:,4]
series2["t-6"]= lags.iloc[:,5]
series2["t-7"]= lags.iloc[:,6]

series2 = series2.iloc[7:1730]
series2 = series2.reset_index(drop=True)

#get variables in correct type
series2["day_of_week"] = series2["day_of_week"].astype('category')
series2["month"] = series2["month"].astype('category')
print(series2.dtypes)



dummies = pd.get_dummies(series2[['month','day_of_week']])
series2 = pd.concat([series2, dummies], axis=1, sort=False)




#############################################################
# trying out ruptures 
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

###
# probably the most suitable one should be "Piecewise linear regression" (see 4.1.2. in paper)

#first drop all columns we dont need
signal = series2.drop(columns=["date", "day", "month", "year", "day_of_week"])
#signal["intercept"] = 1
#since we have a regression here, we should probably add an intercept and drop one dummy per category to avoid multicollinearity
signal = signal.drop(columns=["month_1", "day_of_week_Monday"])
signal = signal.drop(columns=["t-7","t-6","t-5","t-4","t-3"])
signal = signal[270:]
signal = signal.iloc[:,0:3]

signal = signal.to_numpy()




#define algorithm with cost function and execute
algo = rpt.Pelt(model="linear", min_size=5).fit(signal)




my_bkps = algo.predict(pen=100000)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()
#TODO: something weird going on, why do I get breakpoints at the each 305 steps?

#TODO: try out different penalizations, e.g. through:
# pen_values = np.logspace(0, 3, 10)
# algo = rpt.Pelt().fit(signal)
# bkps_list = [algo.predict(pen=pen) for pen in pen_values]
#





################################################################

#get concept features
#bkps have indices of breakpoints stored
#call own designed function for that
series2 = functions.transform_bkps_to_features(my_bkps, series2)


dummies_concept = pd.get_dummies(series2[['concept']])
series2 = pd.concat([series2, dummies_concept], axis=1, sort=False)
series2.to_pickle('series2.pkl')    
print(series2.columns.values)



#save dataframe for models later
#remove some of the attributes that are not needed
series2_cleaned = series2.drop(columns=["date", "day", "month", "year", "day_of_week"])
series2_cleaned.to_pickle('series2_cleaned.pkl')    


















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




