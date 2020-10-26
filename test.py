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

#load the data, and clean time series values 
data  = pd.read_csv("data.csv", sep=";")
print(data.dtypes)
data.columns = data.columns.str.lower()

data["time_series_1"] = data["time_series_1"].str.replace(",", ".")
data["time_series_2"] = data["time_series_2"].str.replace(",", ".")
data["time_series_1"],data["time_series_2"] = pd.to_numeric(data["time_series_1"]), pd.to_numeric(data["time_series_2"])

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
series1 = series1.rename(columns={"time_series_1": "value"})


series2 = data.groupby('date', as_index=False).agg({"time_series_2": "sum"})
series2 = pd.merge(datelist, series2, on="date", how="left").fillna(0)
series2["day"], series2["month"],series2["year"] = series2["date"].dt.day, series2["date"].dt.month, series2["date"].dt.year
series2["day_of_week"] = series2["date"].dt.day_name()
series2.rename(columns={"time_series_2": "value"})














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