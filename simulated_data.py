# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:34:17 2020

@author: Daniel

testing stuff on simulated data
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import functions
import create_simdata
import ruptures as rpt

import ray
ray.init(address='auto', _redis_password='5241590000000000', include_dashboard=False)
assert ray.is_initialized() == True
from ray import tune
#ray.shutdown()






result = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.linear1_abrupt)
result2 = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.nonlinear1_abrupt)





# =============================================================================
# futures = [functions.analysis_rbf.remote(penalization = i, iterations = 5, data_creation_function = create_simdata.linear1_abrupt) for i in range(4)]
# 
# print(ray.get(futures)) # [0, 1, 4, 9]
# 
# 
# 
# @ray.remote
# def f(x):
#     return x * x
# 
# futures = [f.remote(i) for i in range(4)]
# print(ray.get(futures)) # [0, 1, 4, 9]
# =============================================================================


list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear2_abrupt]


def objective(pen, function):
    return functions.analysis_rbf(penalization = pen, iterations = 30, size_concepts=250, data_creation_function = function)


def training_function(config):
    # Hyperparameters
    pen = config["pen"]
# =============================================================================
#    Might be able to do something like this for the different datasets
#    for step in range(10):
#         # Iterative training function - can be any arbitrary training procedure.
#         intermediate_score = objective(step, alpha, beta)
#         # Feed the score back back to Tune.
#         tune.report(mean_loss=intermediate_score)
# =============================================================================
    for function in list_data_functions:
        intermediate_result = objective(pen, function)
 
    #Feed the score back back to Tune.
    tune.report(miss_detection_rate =  intermediate_result[0],
                detection_rate = intermediate_result[1], average_delay = intermediate_result[2])
    
analysis = tune.run(
    training_function,
    config={
        "pen": tune.quniform(1, 100, 1),
        #"beta": tune.choice([1, 2, 3]) 
    },
    #num_samples=16,
    num_samples=100)
    #resources_per_trial={"cpu": 2, "gpu": 0.1})
    #resources_per_trial={"gpu": 0.1})

df = analysis.results_df













#-------------------------------------------------------------------------

lin1_abrupt = create_simdata.linear1_abrupt()
lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)

series = pd.DataFrame({"t":lin1_abrupt})
series = functions.autocorrelations_in_window(10, series)
series = functions.partial_autocorrelations_in_window(10, series)
series = functions.features_in_window(10, series)
series = functions.oscillation_behaviour_in_window(10, series)












nonlinear2_abrupt = create_simdata.nonlinear2_abrupt()

plt.plot(nonlinear2_abrupt)

lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)


signal = lin1_abrupt.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=20)
fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
plt.show()



lin1_abrupt = create_simdata.linear1_abrupt()
lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt) #cuts out the first 10 observations
signal = lin1_abrupt.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

signal = lin1_abrupt.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
bkps = algo.predict(pen=30)



fig, (ax,) = rpt.display(signal[:,0], bkps, figsize=(10, 6))
plt.show()


test = functions.analysis_rbf(penalization = 20, iterations = 1, data_creation_function = create_simdata.linear1_abrupt, size_concepts=250)
