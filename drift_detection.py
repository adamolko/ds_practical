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
import pickle
import statistics

import functions
import create_simdata
import ruptures as rpt

#Parallelization stuff
import ray
ray.init(address='auto', _redis_password='5241590000000000', include_dashboard=False)
assert ray.is_initialized() == True
from ray import tune
ray.shutdown()


#test analysis with random time series
result = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.linear1_abrupt)
result = functions.analysis_rbf(penalization=30, iterations = 10, data_creation_function = create_simdata.nonlinear1_abrupt)
result = functions.analysis_linear(penalization=30, iterations = 10, data_creation_function = create_simdata.linear1_abrupt,
                                    size_concepts = 200, obs_amount_beyond_window=0)


#Hyperparameter optimization - lots of stuff had to be done manually :( 


#Define function for hyperparameter optimization:
def hyperparameter_opt(name, list_functions, beyond_window, pen_min, pen_max, rounding, window_preprocessing=10):

    #change if needed
    def objective(pen, function):
        #return functions.analysis_rbf(penalization = pen, iterations = 10, size_concepts=200, 
         #                        data_creation_function = function, obs_amount_beyond_window=beyond_window,
         #                        windowsize_preprocessing = window_preprocessing)
        #return functions.analysis_linear(penalization = pen, iterations = 20, size_concepts=200, 
                                    # data_creation_function = function, obs_amount_beyond_window=beyond_window)
        return functions.analysis_l2(penalization = pen, iterations = 10, size_concepts=200, 
                                     data_creation_function = function, obs_amount_beyond_window=beyond_window)
    
    
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
    
    
#1)rbf

#Code can be applied to a varierty of ranges for the penalization
#this is the code after we narrowed it down to a tighter range for the RBF kernel
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


#rbf with larger window size (25) in preprocessing

list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
name = "/rbf/window_test/result_hyperpara_opt_linear_abrupt_10_30"
beyond_window = 10
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=10, pen_max=30, rounding=0.05,
                   window_preprocessing=20)

list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
name = "/rbf/window_test/result_hyperpara_opt_nonlinear_abrupt_10_30"
beyond_window = 10
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=10, pen_max=30, rounding=0.05,
                   window_preprocessing=20)

list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
name = "/rbf/window_test/result_hyperpara_opt_linear_inc_10_30"
beyond_window = 15
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=10, pen_max=30, rounding=0.05,
                   window_preprocessing=20)

list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]
name = "/rbf/window_test/result_hyperpara_opt_nonlinear_inc_10_30"
beyond_window = 15
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=10, pen_max=30, rounding=0.05,
                   window_preprocessing=20)


#2)linear

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



#3)l2
p_min = 0
p_max = 500
list_data_functions = [create_simdata.linear1_abrupt, create_simdata.linear2_abrupt, create_simdata.linear3_abrupt]
name = "/l2/result_hyperpara_opt_linear_abrupt_0_500"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=p_min, pen_max=p_max, rounding=0.5)

list_data_functions = [create_simdata.nonlinear1_abrupt, create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt]
name = "/l2/result_hyperpara_opt_nonlinear_abrupt_0_500"
beyond_window = 0
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=p_min, pen_max=p_max, rounding=0.5)

list_data_functions = [create_simdata.linear1_inc, create_simdata.linear2_inc, create_simdata.linear3_inc]
name = "/l2/result_hyperpara_opt_linear_inc_0_500"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=p_min, pen_max=p_max, rounding=0.5)

list_data_functions = [create_simdata.nonlinear1_inc, create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]
name = "/l2/result_hyperpara_opt_nonlinear_inc_0_500"
beyond_window = 5
hyperparameter_opt(name, list_data_functions, beyond_window, pen_min=p_min, pen_max=p_max, rounding=0.5)



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


#-------------------------------------------------------
#Final analysis used in report
#Analysis run exactly once with optimal hyperparameter for each type of model (100 time series per model)

#create data
functions.create_final_data()

#1) RBF
#define function
def final_analysis(dataset, name, window_preprocessing=10, iterations=100, pen=12):

    result = functions.analysis_rbf_final(penalization = pen, iterations = iterations, size_concepts=200, 
                           dataset = dataset, obs_amount_beyond_window=0,
                             windowsize_preprocessing = window_preprocessing)
    identified_bkps_total = result[0]
    not_detected_bkps_total =  result[1]
    miss_detected_bkps_total = result[2]
    delays_score_total = result[3]

    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
        
    #F1 = 2 * (precision * recall) / (precision + recall)    
    f1 = 2*(precision * recall) / (precision + recall) 

    results = [precision, recall, average_delay, f1]
    with open("results/final/" + name + ".data", 'wb') as filehandle:
        pickle.dump(results, filehandle)

    
#linear ones
name = "linear1"
final_analysis(name, name)
name = "linear2"
final_analysis(name, name)
name = "linear3"
final_analysis(name, name)

#non-linear ones
name = "nonlinear1"
final_analysis(name, name)
name = "nonlinear2"
final_analysis(name, name)
name = "nonlinear3"
final_analysis(name, name)

with open('results/final/linear1.data', 'rb') as filehandle:
    linear1 = pickle.load(filehandle)
with open('results/final/linear2.data', 'rb') as filehandle:
    linear2 = pickle.load(filehandle)
with open('results/final/linear3.data', 'rb') as filehandle:
    linear3 = pickle.load(filehandle)

with open('results/final/nonlinear1.data', 'rb') as filehandle:
    nonlinear1 = pickle.load(filehandle)
with open('results/final/nonlinear2.data', 'rb') as filehandle:
    nonlinear2 = pickle.load(filehandle)
with open('results/final/nonlinear3.data', 'rb') as filehandle:
    nonlinear3 = pickle.load(filehandle)


#now for different window sizes of feature:

#linear ones
name = "linear1_window15" ; dataset="linear1"
final_analysis(dataset, name, 15)
name = "linear2_window15"; dataset="linear2"
final_analysis(dataset, name, 15)
name = "linear3_window15"; dataset="linear3"
final_analysis(dataset, name, 15)

#non-linear ones
name = "nonlinear1_window15"; dataset="nonlinear1"
final_analysis(dataset, name, 15)
name = "nonlinear2_window15"; dataset="nonlinear2"
final_analysis(dataset, name, 15)
name = "nonlinear3_window15"; dataset="nonlinear3"
final_analysis(dataset, name, 15)

with open('results/final/linear1_window15.data', 'rb') as filehandle:
    linear1 = pickle.load(filehandle)
with open('results/final/linear2_window15.data', 'rb') as filehandle:
    linear2 = pickle.load(filehandle)
with open('results/final/linear3_window15.data', 'rb') as filehandle:
    linear3 = pickle.load(filehandle)

with open('results/final/nonlinear1_window15.data', 'rb') as filehandle:
    nonlinear1 = pickle.load(filehandle)
with open('results/final/nonlinear2_window15.data', 'rb') as filehandle:
    nonlinear2 = pickle.load(filehandle)
with open('results/final/nonlinear3_window15.data', 'rb') as filehandle:
    nonlinear3 = pickle.load(filehandle)
    
    
#linear ones
name = "linear1_window12" ; dataset="linear1"
final_analysis(dataset, name, 12)
name = "linear2_window12"; dataset="linear2"
final_analysis(dataset, name, 12)
name = "linear3_window12"; dataset="linear3"
final_analysis(dataset, name, 12)

#non-linear ones
name = "nonlinear1_window12"; dataset="nonlinear1"
final_analysis(dataset, name, 12)
name = "nonlinear2_window12"; dataset="nonlinear2"
final_analysis(dataset, name, 12)
name = "nonlinear3_window12"; dataset="nonlinear3"
final_analysis(dataset, name, 12)

with open('results/final/linear1_window12.data', 'rb') as filehandle:
    linear1 = pickle.load(filehandle)
with open('results/final/linear2_window12.data', 'rb') as filehandle:
    linear2 = pickle.load(filehandle)
with open('results/final/linear3_window12.data', 'rb') as filehandle:
    linear3 = pickle.load(filehandle)

with open('results/final/nonlinear1_window12.data', 'rb') as filehandle:
    nonlinear1 = pickle.load(filehandle)
with open('results/final/nonlinear2_window12.data', 'rb') as filehandle:
    nonlinear2 = pickle.load(filehandle)
with open('results/final/nonlinear3_window12.data', 'rb') as filehandle:
    nonlinear3 = pickle.load(filehandle)
    

###
#2)linear:
def final_analysis_linear(dataset, name, window_preprocessing=10, iterations=100, pen=300):

    result = functions.analysis_linear_final(penalization = pen, iterations = iterations, size_concepts=200, 
                           dataset = dataset, obs_amount_beyond_window=0,
                             windowsize_preprocessing = window_preprocessing)
    identified_bkps_total = result[0]
    not_detected_bkps_total =  result[1]
    miss_detected_bkps_total = result[2]
    delays_score_total = result[3]

    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
        
    #F1 = 2 * (precision * recall) / (precision + recall)    
    f1 = 2*(precision * recall) / (precision + recall) 

    results = [precision, recall, average_delay, f1]
    with open("results/final_linear/" + name + ".data", 'wb') as filehandle:
        pickle.dump(results, filehandle)
    
#linear ones
name = "linear1"
final_analysis_linear(name, name)
name = "linear2"
final_analysis_linear(name, name)
name = "linear3"
final_analysis_linear(name, name)

#non-linear ones
name = "nonlinear1"
final_analysis_linear(name, name)
name = "nonlinear2"
final_analysis_linear(name, name)
name = "nonlinear3"
final_analysis_linear(name, name)

with open('results/final_linear/linear1.data', 'rb') as filehandle:
    linear1 = pickle.load(filehandle)
with open('results/final_linear/linear2.data', 'rb') as filehandle:
    linear2 = pickle.load(filehandle)
with open('results/final_linear/linear3.data', 'rb') as filehandle:
    linear3 = pickle.load(filehandle)

with open('results/final_linear/nonlinear1.data', 'rb') as filehandle:
    nonlinear1 = pickle.load(filehandle)
with open('results/final_linear/nonlinear2.data', 'rb') as filehandle:
    nonlinear2 = pickle.load(filehandle)
with open('results/final_linear/nonlinear3.data', 'rb') as filehandle:
    nonlinear3 = pickle.load(filehandle)

#####
#3) L2:
def final_analysis_l2(dataset, name, window_preprocessing=10, iterations=100, pen=410):

    result = functions.analysis_l2_final(penalization = pen, iterations = iterations, size_concepts=200, 
                            dataset = dataset,  obs_amount_beyond_window=0,
                             windowsize_preprocessing = window_preprocessing)
    identified_bkps_total = result[0]
    not_detected_bkps_total =  result[1]
    miss_detected_bkps_total = result[2]
    delays_score_total = result[3]

    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
        
    #F1 = 2 * (precision * recall) / (precision + recall)    
    f1 = 2*(precision * recall) / (precision + recall) 

    results = [precision, recall, average_delay, f1]
    with open("results/final_l2/" + name + ".data", 'wb') as filehandle:
        pickle.dump(results, filehandle)
    
#linear ones
name = "linear1"
final_analysis_l2(name, name)
name = "linear2"
final_analysis_l2(name, name)
name = "linear3"
final_analysis_l2(name, name)

#non-linear ones
name = "nonlinear1"
final_analysis_l2(name, name)
name = "nonlinear2"
final_analysis_l2(name, name)
name = "nonlinear3"
final_analysis_l2(name, name)


with open('results/final_l2/linear1.data', 'rb') as filehandle:
    linear1 = pickle.load(filehandle)
with open('results/final_l2/linear2.data', 'rb') as filehandle:
    linear2 = pickle.load(filehandle)
with open('results/final_l2/linear3.data', 'rb') as filehandle:
    linear3 = pickle.load(filehandle)

with open('results/final_l2/nonlinear1.data', 'rb') as filehandle:
    nonlinear1 = pickle.load(filehandle)
with open('results/final_l2/nonlinear2.data', 'rb') as filehandle:
    nonlinear2 = pickle.load(filehandle)
with open('results/final_l2/nonlinear3.data', 'rb') as filehandle:
    nonlinear3 = pickle.load(filehandle)

















#-------------------------------------------------------
#Stability Analysis - Experimental!!!
#--
#Short term
linear1 = functions.stability_analysis_short_term(
    penalization = 12, iterations = 10, dataset = "linear1", size_concepts=200 ,
    windowsize_preprocessing = 10)

futures = [functions.stability_analysis_short_term.remote(
    penalization = 12, iterations = 20, dataset = i,
    size_concepts = 200, windowsize_preprocessing = 10) 
    for i in ["linear1", "linear2", "linear3", "nonlinear1", "nonlinear2", "nonlinear3"]
           ]  
result = ray.get(futures)
with open('results/stability_analysis/short_term/result.data', 'wb') as filehandle:
    pickle.dump(result, filehandle)
    
with open('results/stability_analysis/short_term/result.data', 'rb') as filehandle:
    result = pickle.load(filehandle)
    
linear1=result[0]
print(sum(linear1[0])/len(linear1[0]))
print(statistics.median(linear1[0]))
linear2=result[1]
print(sum(linear2[0])/len(linear2[0]))
print(statistics.median(linear2[0]))
linear3=result[2]
print(sum(linear3[0])/len(linear3[0]))
print(statistics.median(linear3[0]))
nonlinear1=result[3]
print(sum(nonlinear1[0])/len(nonlinear1[0]))
print(statistics.median(nonlinear1[0]))
nonlinear2=result[4]
print(sum(nonlinear2[0])/len(nonlinear2[0]))
print(statistics.median(nonlinear2[0]))
nonlinear3=result[5]
print(sum(nonlinear3[0])/len(nonlinear3[0]))
print(statistics.median(nonlinear3[0]))
#--
#Long term
penalization = 12
iterations = 20
size_concepts = 200

#linear1_abrupt
data_creation_function = create_simdata.linear1_abrupt

sds_linear1_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/long_term/sds_linear1_abrupt.data', 'wb') as filehandle:
    pickle.dump(sds_linear1_abrupt, filehandle)
# with open('results/stability_analysis/long_term/sds_linear1_abrupt.data', 'rb') as filehandle:
#     sds_linear1_abrupt = pickle.load(filehandle)
    
result = functions.stability_analysis_short_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/short_term/linear1_abrupt.data', 'wb') as filehandle:
    pickle.dump(result, filehandle)
# with open('results/stability_analysis/short_term/linear1_abrupt.data', 'rb') as filehandle:
#     result = pickle.load(filehandle)
    
#Parallelize this shit:   
futures1 = [functions.stability_analysis_long_term.remote(penalization = penalization, iterations = iterations, 
                                                         data_creation_function = i, size_concepts = size_concepts) for i in 
           [create_simdata.linear2_abrupt, create_simdata.linear3_abrupt, create_simdata.linear1_inc,
            create_simdata.linear2_inc, create_simdata.linear3_inc, create_simdata.nonlinear1_abrupt,
            create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt, create_simdata.nonlinear1_inc,
            create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]]  

result1 = ray.get(futures1)
with open('results/stability_analysis/long_term/result1.data', 'wb') as filehandle:
    pickle.dump(result1, filehandle)

futures2 = [functions.stability_analysis_short_term.remote(penalization = penalization, iterations = iterations, 
                                                         data_creation_function = i, size_concepts = size_concepts) for i in 
           [create_simdata.linear2_abrupt, create_simdata.linear3_abrupt, create_simdata.linear1_inc,
            create_simdata.linear2_inc, create_simdata.linear3_inc, create_simdata.nonlinear1_abrupt,
            create_simdata.nonlinear2_abrupt, create_simdata.nonlinear3_abrupt, create_simdata.nonlinear1_inc,
            create_simdata.nonlinear2_inc, create_simdata.nonlinear3_inc]]
    
result2 = ray.get(futures2)
with open('results/stability_analysis/short_term/result2.data', 'wb') as filehandle:
    pickle.dump(result2, filehandle)

# with open('results/stability_analysis/short_term/linear1_abrupt.data', 'rb') as filehandle:
#     result = pickle.load(filehandle)

ray.get(futures1)

#linear2_abrupt
data_creation_function = create_simdata.linear2_abrupt

sds_linear2_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/long_term/sds_linear2_abrupt.data', 'wb') as filehandle:
    pickle.dump(sds_linear2_abrupt, filehandle)
    
result = functions.stability_analysis_short_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/short_term/linear2_abrupt.data', 'wb') as filehandle:
    pickle.dump(result, filehandle)
  
#linear3_abrupt
data_creation_function = create_simdata.linear3_abrupt

sds_linear3_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/long_term/sds_linear3_abrupt.data', 'wb') as filehandle:
    pickle.dump(sds_linear3_abrupt, filehandle)

result = functions.stability_analysis_short_term(penalization, iterations, data_creation_function, size_concepts)
with open('results/stability_analysis/short_term/linear3_abrupt.data', 'wb') as filehandle:
    pickle.dump(result, filehandle)


data_creation_function = create_simdata.linear1_inc
sds_linear1_inc = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.linear2_inc
sds_linear2_inc = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.linear3_inc
sds_linear3_inc = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

#nonlinear
data_creation_function = create_simdata.nonlinear1_abrupt
sds_nonlinear1_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.nonlinear2_abrupt
sds_nonlinear2_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.nonlinear3_abrupt
sds_nonlinear3_abrupt = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.nonlinear1_inc
sds_nonlinear1_inc = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.nonlinear2_inc
sds_nonlinear2_inc= functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)

data_creation_function = create_simdata.nonlinear3_inc
sds_nonlinear3_inc = functions.stability_analysis_long_term(penalization, iterations, data_creation_function, size_concepts)





#-------------------------------------------------------------------------
#testing area for random stuff
#left it in there, because it allows to run our drift detection on a single time series as well :)
lin1_abrupt = create_simdata.linear1_abrupt()
lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)

series = pd.DataFrame({"t":lin1_abrupt})
series = functions.autocorrelations_in_window(10, series)
series = functions.partial_autocorrelations_in_window(10, series)
series = functions.features_in_window(10, series)
series = functions.oscillation_behaviour_in_window(10, series)


timeseries = create_simdata.linear1_abrupt()
timeseries = functions.ada_preprocessing(timeseries, delay_correction=2)

nonlinear2_abrupt_raw = create_simdata.linear2_abrupt()
nonlinear2_abrupt_raw_2 = nonlinear2_abrupt_raw[0:620]
nonlinear2_abrupt_raw_3 =  nonlinear2_abrupt_raw[1:650]
nonlinear2_abrupt_raw_4 =  nonlinear2_abrupt_raw[1:700]
nonlinear2_abrupt_raw_5 =  nonlinear2_abrupt_raw[1:750]
nonlinear2_abrupt = functions.preprocess_timeseries(nonlinear2_abrupt_raw, windowsize=10)
nonlinear2_abrupt_raw_2 = functions.preprocess_timeseries(nonlinear2_abrupt_raw_2, windowsize=10)
nonlinear2_abrupt_raw_3 = functions.preprocess_timeseries(nonlinear2_abrupt_raw_3, windowsize=10)
nonlinear2_abrupt_raw_4 = functions.preprocess_timeseries(nonlinear2_abrupt_raw_4, windowsize=10)
nonlinear2_abrupt_raw_5= functions.preprocess_timeseries(nonlinear2_abrupt_raw_5, windowsize=10)

#plt.plot(nonlinear2_abrupt)

#lin1_abrupt = functions.preprocess_timeseries(lin1_abrupt)


signal = nonlinear2_abrupt.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps1 = algo.predict(pen=12)
signal = nonlinear2_abrupt_raw_2.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps2 = algo.predict(pen=12)
signal = nonlinear2_abrupt_raw_3.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps3 = algo.predict(pen=12)
signal = nonlinear2_abrupt_raw_4.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps4 = algo.predict(pen=12)
signal = nonlinear2_abrupt_raw_5.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                  'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
my_bkps5 = algo.predict(pen=12)




#fig, (ax,) = rpt.display(signal[:,0], my_bkps, figsize=(10, 6))
#plt.show()

