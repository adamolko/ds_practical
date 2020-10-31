# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:34:17 2020

@author: Daniel

testing stuff on simulated data
"""


from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions
import ruptures as rpt

###########################
#Simulation 1
arparams = np.array([.75, -.25, .5, -.2])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1] # add zero-lag
#ma = np.r_[1, maparams] # add zero-lag
AR_object = ArmaProcess(ar, ma)
AR_object.isstationary
simulated_data1 = AR_object.generate_sample(nsample=1000, scale=0.1)
plt.plot(simulated_data1)

simulated_data1 = pd.DataFrame(simulated_data1).rename(columns={0: "t"})
simulated_data1 = functions.autocorrelations_in_window(10, simulated_data1)
simulated_data1 = functions.partial_autocorrelations_in_window(10, simulated_data1)
simulated_data1 = functions.features_in_window(10, simulated_data1)
simulated_data1 = functions.oscillation_behaviour_in_window(10, simulated_data1)


lags = pd.concat([simulated_data1["t"].shift(1), simulated_data1["t"].shift(2), 
                  simulated_data1["t"].shift(3),simulated_data1["t"].shift(4)], axis=1)
simulated_data1["t-1"]= lags.iloc[:,0]
simulated_data1["t-2"]= lags.iloc[:,1]
simulated_data1["t-3"]= lags.iloc[:,2]
simulated_data1["t-4"]= lags.iloc[:,3]
simulated_data1 = simulated_data1[10:]
stand = functions.standardize(simulated_data1.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']])
simulated_data1.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']] = stand


#series = series.iloc[:,0:5]
simulated_data1["intercept"] = 1
simulated_data1 = simulated_data1.reset_index(drop=True)



signal = simulated_data1.loc[:,["t", "t-1", "t-2", "t-3", "t-4", "intercept"]]
signal = signal.to_numpy()

algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(signal)
my_bkps = algo.predict(pen=1)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()

signal = simulated_data1.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=10)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()



###########################


###########################
#Simulation 2
arparams = np.array([])
maparams = np.array([.5])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
AR_object = ArmaProcess(ar, ma)
AR_object.isstationary
simulated_data_first_half = AR_object.generate_sample(nsample=500, scale=0.1)

arparams  = np.array([.75, -.25, .5, -.2])
maparams = np.array([.5])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
AR_object = ArmaProcess(ar, ma)
AR_object.isstationary
simulated_data_second_half = AR_object.generate_sample(nsample=500, scale=0.1)

simulated_data2 = np.concatenate((simulated_data_first_half, simulated_data_second_half))
plt.plot(simulated_data2)


simulated_data2 = pd.DataFrame(simulated_data2).rename(columns={0: "t"})
simulated_data2 = functions.autocorrelations_in_window(10, simulated_data2)
simulated_data2 = functions.partial_autocorrelations_in_window(10, simulated_data2)
simulated_data2 = functions.features_in_window(10, simulated_data2)
simulated_data2 = functions.oscillation_behaviour_in_window(10, simulated_data2)



lags = pd.concat([simulated_data2["t"].shift(1), simulated_data2["t"].shift(2), 
                  simulated_data2["t"].shift(3),simulated_data2["t"].shift(4)], axis=1)
simulated_data2["t-1"]= lags.iloc[:,0]
simulated_data2["t-2"]= lags.iloc[:,1]
simulated_data2["t-3"]= lags.iloc[:,2]
simulated_data2["t-4"]= lags.iloc[:,3]
simulated_data2 = simulated_data2[10:]
stand = functions.standardize(simulated_data2.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']])
simulated_data2.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']] = stand


#series = series.iloc[:,0:5]
simulated_data2["intercept"] = 1
simulated_data2 = simulated_data2.reset_index(drop=True)



signal = simulated_data2.loc[:,["t", "t-1", "t-2", "t-3", "t-4", "intercept"]]
signal = signal.to_numpy()

algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(signal)
my_bkps = algo.predict(pen=1)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()

signal = simulated_data2.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=30)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()





###########################