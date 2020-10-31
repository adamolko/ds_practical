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
simulated_data1 = functions.mutual_info(10, simulated_data1)
simulated_data1 = simulated_data1[10:]
stand = functions.standardize(simulated_data1.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']])
simulated_data1.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']] = stand


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
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

signal = simulated_data1.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=5)
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
simulated_data2 = functions.mutual_info(10, simulated_data2)
simulated_data2 = simulated_data2[10:]
stand = functions.standardize(simulated_data2.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']])
simulated_data2.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']] = stand


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
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

signal = simulated_data2.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=10)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()





###########################
#create similar artificial datasets as in FEDD paper (could also try the exact same)
#they used linear and non-linear model time series
#for both categories they created 3 groups of different models (with different coefficients)
#for each group they have a an abrupt & a gradual drift
#for each dataset they have 3 drifts in there

#start with linear models and abrupt drift


list_y = []
list_y.append(1)
list_y.append(0.5)
list_y.append(1.5)
list_y.append(1.2)
alpha_1 = 0.9
alpha_2 = -0.2
alpha_3 = 0.8
alpha_4 = -0.5

sigma = 0.1 # mean and standard deviation

for x in range(4,1000,1):
    error = np.random.normal(0, sigma, 1)
    new_y = alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  alpha_3 * list_y[x-3] +  alpha_4 * list_y[x-4] + error[0]
    list_y.append(new_y)

del list_y[0:4]




#-------------------------------------------------------------------------

##########
#Linear 1 - Abrupt
#Concept 1
lin1_abrupt = []
n_obs=500
starting_values = [0, 0, 0, 1, 0.5, 1.5, 1.2]
list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]
sigma = 0.5
lin1_abrupt = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin1_abrupt[493:]
list_alphas = [-0.3, 1.4, 0.4, -0.5, 0, 0, 0]   
sigma = 1.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin1_abrupt = [*lin1_abrupt, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin1_abrupt[993:]
list_alphas = [1.5, -0.4, -0.3, 0.2, 0, 0, 0]   
sigma = 2.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin1_abrupt = [*lin1_abrupt, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin1_abrupt[1493:]
list_alphas = [-0.1, 1.4, 0.4, -0.7, 0, 0, 0]   
sigma = 3.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin1_abrupt = [*lin1_abrupt, *new_data]

plt.plot(lin1_abrupt)

lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)


series = lin1_abrupt.copy()

##########

##########
#Linear 2 - Abrupt
#Concept 1
lin2_abrupt = []
n_obs=500
starting_values = [0, 0.8, 0.2, 1, 0.5, 1.5, 1.2]
list_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]
sigma = 0.5
lin2_abrupt = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin2_abrupt[493:]
list_alphas = [-0.1, 1.2, 0.4, 0.3, -0.2, -0.6, 0]   
sigma = 1.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin2_abrupt = [*lin2_abrupt, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin2_abrupt[993:]
list_alphas = [1.2, -0.4, -0.3, 0.7, -0.6, 0.4, 0]   
sigma = 2.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin2_abrupt = [*lin2_abrupt, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin2_abrupt[1493:]
list_alphas = [-0.1, 1.1, 0.5, 0.2, -0.2, -0.5, 0]   
sigma = 3.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin2_abrupt = [*lin2_abrupt, *new_data]

plt.plot(lin2_abrupt)
##########


##########
#Linear 3 - Abrupt
#Concept 1
lin3_abrupt = []
n_obs=500
starting_values = [0, 0, 0, 0, 0, 0.5, 1]
list_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
sigma = 0.5
lin3_abrupt = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin3_abrupt[493:]
list_alphas = [1.5, -0.5, 0, 0, 0, 0, 0]   
sigma = 1.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin3_abrupt[993:]
list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]   
sigma = 2.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin3_abrupt[1493:]
list_alphas = [0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4]   
sigma = 3.5
new_data = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]

plt.plot(lin3_abrupt)
##########











##########
#Linear 1 - Incremental
#Concept 1
lin1_inc = []
n_obs=500
starting_values = [0, 0, 0, 1, 0.5, 1.5, 1.2]
list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]
sigma = 0.5
lin1_inc = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin1_inc[493:]
list_alphas = [-0.3, 1.4, 0.4, -0.5, 0, 0, 0]  
list_old_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0] 
sigma = 1.5
sigma_old = 0.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin1_inc = [*lin1_inc, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin1_inc[993:]
list_alphas = [1.5, -0.4, -0.3, 0.2, 0, 0, 0]   
list_old_alphas = [-0.3, 1.4, 0.4, -0.5, 0, 0, 0]  
sigma = 2.5
sigma_old = 1.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin1_inc = [*lin1_inc, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin1_inc[1493:]
list_alphas = [-0.1, 1.4, 0.4, -0.7, 0, 0, 0]   
list_old_alphas = [1.5, -0.4, -0.3, 0.2, 0, 0, 0]   
sigma = 3.5
sigma_old = 2.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin1_inc = [*lin1_inc, *new_data]

plt.plot(lin1_inc)
##########

##########
#Linear 2 - Incremental
#Concept 1
lin2_inc = []
n_obs=500
starting_values = [0, 0.8, 0.2, 1, 0.5, 1.5, 1.2]
list_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]
sigma = 0.5
lin2_inc = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin2_inc[493:]
list_alphas = [-0.1, 1.2, 0.4, 0.3, -0.2, -0.6, 0]
list_old_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]   
sigma = 1.5
sigma_old = 0.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin2_inc = [*lin2_inc, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin2_inc[993:]
list_alphas = [1.2, -0.4, -0.3, 0.7, -0.6, 0.4, 0]   
list_old_alphas = [-0.1, 1.2, 0.4, 0.3, -0.2, -0.6, 0]
sigma = 2.5
sigma_old = 1.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin2_inc = [*lin2_inc, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin2_inc[1493:]
list_alphas = [-0.1, 1.1, 0.5, 0.2, -0.2, -0.5, 0]  
list_old_alphas = [1.2, -0.4, -0.3, 0.7, -0.6, 0.4, 0]  
sigma = 3.5
sigma_old = 2.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin2_inc = [*lin2_inc, *new_data]

plt.plot(lin2_inc)
##########


##########
#Linear 3 - Incremental
#Concept 1
lin3_abrupt = []
n_obs=500
starting_values = [0, 0, 0, 0, 0, 0.5, 1]
list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
sigma = 0.5
lin3_abrupt = functions.simulate_ar_data(n_obs, sigma, list_alphas, starting_values)
#Concept 2
n_obs=500
starting_values = lin3_abrupt[493:]
list_alphas = [1.5, -0.5, 0, 0, 0, 0, 0]   
list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
sigma = 1.5
sigma_old = 0.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]
##########
#Concept 3
n_obs=500
starting_values = lin3_abrupt[993:]
list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]   
list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
sigma = 2.5
sigma_old = 1.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]
##########
#Concept 4
n_obs=500
starting_values = lin3_abrupt[1493:]
list_alphas = [0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4]   
list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
sigma = 3.5
sigma_old = 2.5
speed = 30
new_data = functions.simulate_ar_data_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
lin3_abrupt = [*lin3_abrupt, *new_data]

plt.plot(lin3_abrupt)
##########

