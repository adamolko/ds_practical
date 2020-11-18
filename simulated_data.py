# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:34:17 2020

@author: Daniel

testing stuff on simulated data
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import functions
import create_simdata
import ruptures as rpt

import ray
ray.init(address='auto', _redis_password='5241590000000000', include_dashboard=False)
assert ray.is_initialized() == True
from ray import tune
#ray.shutdown()


create_simdata.nonlinear3_abrupt()



result = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.linear1_abrupt)
result2 = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.nonlinear1_abrupt)


result = functions.analysis_linear(penalization=30, iterations = 10, data_creation_function = create_simdata.linear1_abrupt,
                                    size_concepts = 200, obs_amount_beyond_window=0)



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

#switch this, based on what kind of data to use
list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]


def objective(pen, function):
    #return functions.analysis_rbf(penalization = pen, iterations = 10, size_concepts=200, 
    #                            data_creation_function = function, obs_amount_beyond_window=5)
    return functions.analysis_linear(penalization = pen, iterations = 10, size_concepts=200, 
                                  data_creation_function = function, obs_amount_beyond_window=5)


def training_function(config):
    # Hyperparameters
    pen = config["pen"]
    #function = config["datafunction"]
    #function = random.choice(list_data_functions)
# =============================================================================
#    Might be able to do something like this for the different datasets
#    for step in range(10):
#         # Iterative training function - can be any arbitrary training procedure.
#         intermediate_score = objective(step, alpha, beta)
#         # Feed the score back back to Tune.
#         tune.report(mean_loss=intermediate_score)
# =============================================================================
    avg_prec = 0;
    avg_rec = 0;
    avg_del = 0;
    for function in list_data_functions:
        intermediate_result = objective(pen, function)
        avg_prec += intermediate_result[0]
        avg_rec +=  intermediate_result[1]
        avg_del += intermediate_result[2]
    
    avg_prec = avg_prec/3
    avg_rec = avg_rec/3
    avg_del = avg_del/3
    #function = create_simdata.linear1_abrupt
    #intermediate_result = objective(pen, function)
    tune.report(precision =  avg_prec,
            recall = avg_rec, average_delay = avg_del)
 
    #Feed the score back back to Tune.
    
    
analysis = tune.run(
    training_function,
    config={
        "pen": tune.quniform(0, 500, 1),
        #"datafunction": tune.choice(list_data_functions),
    },
    #num_samples=16,
    num_samples=100)
    #resources_per_trial={"cpu": 2, "gpu": 0.1})
    #resources_per_trial={"gpu": 0.1})

df2 = analysis.results_df

#F1 = 2 * (precision * recall) / (precision + recall)
df2["f1"] = 2*(df2["precision"]*df2["recall"])/(df2["precision"]+df2["recall"])
df2["f1"].fillna(0, inplace=True)


#change name here
df2.to_pickle("results/linear/result_hyperpara_opt_linear_abrupt_200_300.pkl") 


#df2 = pd.read_pickle("results/result_hyperpara_opt_linearabrupt_complete.pkl")





#Plot 1:
ax = df2.plot.scatter(x='config.pen', y='recall', label='Recall', color='Green',);
df2.plot.scatter(x='config.pen', y='precision', color='Orange', label='Precision', ax=ax);
plt.xlabel("Penalization")
plt.ylabel("Rate")
#change name here:
#plt.savefig("results/nonlinear_inc_complete_recall_vs_prec.png", dpi=150)
#plt.savefig("results/linear/linear_abrupt_complete_recall_vs_prec.png", dpi=150)
plt.savefig("results/linear/linear_abrupt_complete_recall_vs_prec.png", dpi=150)
#Plot 2:


ax = df2.plot.scatter(x='config.pen', y='f1', color='Green',);
plt.xlabel("Penalization")
#change name here:
#plt.savefig("results/linear/linear_abrupt_complete_f1.png", dpi=150)
plt.savefig("results/linear/linear_abrupt_complete_f1.png", dpi=150)













#Define function for hyperparameter optimization:


def hyperparameter_opt(name, list_functions, beyond_window, pen_min, pen_max, rounding, window_preprocessing=10):

    def objective(pen, function):
        return functions.analysis_rbf(penalization = pen, iterations = 10, size_concepts=200, 
                                 data_creation_function = function, obs_amount_beyond_window=beyond_window,
                                 windowsize_preprocessing = window_preprocessing)
        #return functions.analysis_linear(penalization = pen, iterations = 20, size_concepts=200, 
                                    # data_creation_function = function, obs_amount_beyond_window=beyond_window)
    
    
    def training_function(config):
        # Hyperparameters
        pen = config["pen"]
        avg_prec = 0;
        avg_rec = 0;
        avg_del = 0;
        for function in list_functions:
            intermediate_result = objective(pen, function)
            avg_prec += intermediate_result[0]
            avg_rec +=  intermediate_result[1]
            avg_del += intermediate_result[2]
        
        avg_prec = avg_prec/3
        avg_rec = avg_rec/3
        avg_del = avg_del/3

        tune.report(precision =  avg_prec,
                recall = avg_rec, average_delay = avg_del)
     

        
        
    analysis = tune.run(
        training_function,
        config={
            "pen": tune.quniform(pen_min, pen_max, rounding),
        },
        num_samples=100)
    
    df2 = analysis.results_df

    #F1 = 2 * (precision * recall) / (precision + recall)
    df2["f1"] = 2*(df2["precision"]*df2["recall"])/(df2["precision"]+df2["recall"])
    df2["f1"].fillna(0, inplace=True)


    #change name here
    df2.to_pickle( "results" + name + ".pkg")
    
    #Plot 1:
    ax = df2.plot.scatter(x='config.pen', y='recall', label='Recall', color='Green',);
    df2.plot.scatter(x='config.pen', y='precision', color='Orange', label='Precision', ax=ax);
    plt.xlabel("Penalization")
    plt.ylabel("Rate")
    plt.savefig( "results" + name + "_recall_vs_prec.png", dpi=150)
    
    #Plot 2:
    ax = df2.plot.scatter(x='config.pen', y='f1', color='Green',);
    plt.xlabel("Penalization")
    plt.savefig( "results" + name + "_f1.png", dpi=150)
    




#run in one more time between 8 and 14 and then determine best parameter

list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
name = "/rbf/result_hyperpara_opt_linear_abrupt_8_14"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=8, pen_max=14, rounding=0.05)

list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
name = "/rbf/result_hyperpara_opt_nonlinear_abrupt_8_14"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=8, pen_max=14, rounding=0.05)

list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
name = "/rbf/result_hyperpara_opt_linear_inc_8_14"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=8, pen_max=14, rounding=0.05)

list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]
name = "/rbf/result_hyperpara_opt_nonlinear_inc_8_14"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=8, pen_max=14, rounding=0.05)


#try it out with larger window size (25) in preprocessing

list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
name = "/rbf/window_test/result_hyperpara_opt_linear_abrupt_0_50"
beyond_window = 10
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=0, pen_max=50, rounding=0.5,
                   window_preprocessing=20)

list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
name = "/rbf/window_test/result_hyperpara_opt_nonlinear_abrupt_0_50"
beyond_window = 10
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=0, pen_max=50, rounding=0.5,
                   window_preprocessing=20)

list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
name = "/rbf/window_test/result_hyperpara_opt_linear_inc_0_50"
beyond_window = 15
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=0, pen_max=50, rounding=0.5,
                   window_preprocessing=20)

list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]
name = "/rbf/window_test/result_hyperpara_opt_nonlinear_inc_0_50"
beyond_window = 15
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=0, pen_max=50, rounding=0.5,
                   window_preprocessing=20)








list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
name = "/linear/result_hyperpara_opt_linear_abrupt_200_400"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=200, pen_max=400, rounding=0.5)

list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
name = "/linear/result_hyperpara_opt_nonlinear_abrupt_200_400"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=200, pen_max=400, rounding=0.5)

list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
name = "/linear/result_hyperpara_opt_linear_inc_200_400"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=200, pen_max=400, rounding=0.5)

list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]
name = "/linear/result_hyperpara_opt_nonlinear_inc_200_400"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=200, pen_max=400, rounding=0.5)



#get best hyperparameter:
df = pd.read_pickle("results/rbf/window_test/result_hyperpara_opt_linear_abrupt_0_50.pkg")
df.loc[df['f1'].idxmax()] 

ax = df.plot.scatter(x='config.pen', y='f1', color='Green',);
plt.xlabel("Penalization")
#plt.savefig( "results" + name + "_f1.png", dpi=150)


df = pd.read_pickle("results/rbf/result_hyperpara_opt_linear_inc_8_14.pkg")
df.loc[df['f1'].idxmax()] 

df = pd.read_pickle("results/rbf/result_hyperpara_opt_nonlinear_abrupt_8_14.pkg")
df.loc[df['f1'].idxmax()] 

df = pd.read_pickle("results/rbf/result_hyperpara_opt_nonlinear_inc_8_14.pkg")
df.loc[df['f1'].idxmax()] 

##--> there is no definitive best result, max f1 score is between 11.5 and 13.5
# Also when looking at plot, there is still quite a lot of variation
# Might have to run analysis with iteration size a lot larger than 20, but that is just unfeasible
# therefore a value around 12, should still get us "fine" results



#Next step:
#run analysis exactly one time with optimal parameter :)


























#-------------------------------------------------------------------------
#testing area here:


lin1_abrupt = create_simdata.linear1_abrupt()
lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)

series = pd.DataFrame({"t":lin1_abrupt})
series = functions.autocorrelations_in_window(10, series)
series = functions.partial_autocorrelations_in_window(10, series)
series = functions.features_in_window(10, series)
series = functions.oscillation_behaviour_in_window(10, series)



timeseries = create_simdata.linear1_abrupt()

timeseries = functions.ada_preprocessing(timeseries, delay_correction=2)







nonlinear2_abrupt_raw = create_simdata.nonlinear3_abrupt()
nonlinear2_abrupt = functions.preprocess_timeseries(nonlinear2_abrupt_raw, windowsize=20)

plt.plot(nonlinear2_abrupt)

lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)


signal = nonlinear2_abrupt.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()

algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps = algo.predict(pen=18)
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
