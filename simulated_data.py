# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:34:17 2020

@author: Daniel
"""


from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
lags = pd.concat([simulated_data1["t"].shift(1), simulated_data1["t"].shift(2), 
                  simulated_data1["t"].shift(3),simulated_data1["t"].shift(4)], axis=1)
simulated_data1["t-1"]= lags.iloc[:,0]
#simulated_data1["t-2"]= lags.iloc[:,1]
#simulated_data1["t-3"]= lags.iloc[:,2]
#simulated_data1["t-4"]= lags.iloc[:,3]

simulated_data1 = simulated_data1[4:]

simulated_data1 = simulated_data1.to_numpy()

algo = rpt.Pelt(model="rbf", min_size=5, jump=1).fit(simulated_data1)
my_bkps = algo.predict(pen=3)
fig, (ax,) = rpt.display(simulated_data1[:,0], my_bkps, figsize=(10, 6))
plt.show()


algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(simulated_data1)
my_bkps = algo.predict(pen=1)
fig, (ax,) = rpt.display(simulated_data1[:,0], my_bkps, figsize=(10, 6))
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
lags = pd.concat([simulated_data2["t"].shift(1), simulated_data2["t"].shift(2), 
                  simulated_data2["t"].shift(3),simulated_data2["t"].shift(4)], axis=1)
simulated_data2["t-1"]= lags.iloc[:,0]
simulated_data2["t-2"]= lags.iloc[:,1]
simulated_data2["t-3"]= lags.iloc[:,2]
simulated_data2["t-4"]= lags.iloc[:,3]


simulated_data2 = simulated_data2[4:]

simulated_data2 = simulated_data2.to_numpy()

algo = rpt.Pelt(model="rbf", min_size=5, jump=1).fit(simulated_data2)
my_bkps = algo.predict(pen=3)
fig, (ax,) = rpt.display(simulated_data2[:,0], my_bkps, figsize=(10, 6))
plt.show()


algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(simulated_data2)
my_bkps = algo.predict(pen=1)
fig, (ax,) = rpt.display(simulated_data2[:,0], my_bkps, figsize=(10, 6))
plt.show()





###########################