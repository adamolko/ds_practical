# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:41:05 2020

@author: Daniel
"""

import statsmodels.tsa.stattools as arma_stats
from scipy.stats import kurtosis, skew
from statistics import variance
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
import math
import create_simdata
import ruptures as rpt
#import ray
#ray.init()

def transform_bkps_to_features(bkps, timeseries, delay_correction = 0):
    series = timeseries.copy()
    list_concepts = []
    list_transition = []
    count_rows = series.shape[0] 
    current_concept = 1
    
    bkps = bkps.copy()
    bkps = [x-delay_correction for x in bkps]
  
   # for x in range(1, count_rows+1):
    transition_count = 11
    
    for x in range(0, count_rows):
        if (x in bkps): 
            current_concept+=1
            transition_count = 1
        list_concepts.append(current_concept)
        if transition_count < 11:
            list_transition.append(True)
        if transition_count > 10:
            list_transition.append(False)
        transition_count += 1    

    
    series["concept"] = list_concepts
    series["concept"] = series["concept"].astype("category")
    series["transition"] = list_transition
    return series

def ada_preprocessing(timeseries, delay_correction = 0):
    series = timeseries.copy()
    
    
    #series = create_simdata.linear1_abrupt()
    series = preprocess_timeseries(series) #cuts out the first 10 observations
    signal = series.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
    algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
    bkps = algo.predict(pen=12)
    #print(bkps)
    bkps = bkps[:-1]
    
    series = series.reset_index(drop=True)
    series = transform_bkps_to_features(bkps, series, delay_correction)
    series = series.loc[:,["t","t-1","t-2","t-3","t-4","t-5","concept", "transition"]]
    
    return series


def standardize(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result

def mutual_info(windowsize, timeseries):
    series = timeseries.copy()
    series['mi_lag1'], series['mi_lag2'], series['mi_lag3'] = [0, 0, 0]
    number_rows = series.shape[0] 
    starting_point = windowsize
    for i in range(starting_point, number_rows+1, 1):
        window = series.iloc[i-windowsize:i].dropna() #drop rows if some lags are missing
        mi = mutual_info_regression(y = window.loc[:,['t']].values.ravel(), X = window.loc[:,['t-1','t-2','t-3']])
        series.loc[i-1,['mi_lag1','mi_lag2','mi_lag3']] = mi     
    return series

def autocorrelations_in_window(windowsize, timeseries):
    series = timeseries.copy()
    series['acf0'], series['acf1'], series['acf2'], series['acf3'], series['acf4'], series['acf5'] = [0, 0, 0, 0, 0, 0]
    number_rows = series.shape[0] 
    for i in range(windowsize, number_rows+1, 1):
        window = series.iloc[i-windowsize:i]
        acfs = arma_stats.acf(window["t"], nlags=5, fft=False)
        series.loc[i-1,['acf0','acf1','acf2', 'acf3', 'acf4', 'acf5' ]] = acfs     
    return series
 

def partial_autocorrelations_in_window(windowsize, timeseries):
    #there is still an error: RuntimeWarning: invalid value encountered in sqrt return rho, np.sqrt(sigmasq)
    series = timeseries.copy()
    series['pacf0'], series['pacf1'], series['pacf2'], series['pacf3'] = [0, 0, 0, 0]
    number_rows = series.shape[0] 
    for i in range(windowsize, number_rows+1, 1):
        window = series.iloc[i-windowsize:i]
        pacfs = arma_stats.pacf(window["t"], nlags=3, method = "ywmle")
        series.loc[i-1,['pacf0','pacf1','pacf2', 'pacf3']] = pacfs
    return series

def features_in_window(windowsize, timeseries):
    series = timeseries.copy()
    series['var'], series['kurt'], series['skew'] = [0, 0, 0]
    number_rows = series.shape[0]
    for i in range(windowsize, number_rows+1, 1):
        window = series.iloc[i-windowsize:i]
        kurt = kurtosis(window["t"])
        skewness = skew(window["t"])
        var = variance(window["t"])
        series.loc[i-1,['var','kurt','skew']] = var, kurt, skewness
    return series

def turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING: 
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max
def oscillation_behaviour_in_window(windowsize, timeseries):
    series = timeseries.copy()
    series['osc']= 0
    number_rows = series.shape[0]
    for i in range(windowsize, number_rows+1, 1):
        window = series.iloc[i-windowsize:i]
        points = turning_points(window["t"].values)
        sum_points = sum(len(x) for x in points)
        oscillation = sum_points/windowsize
        series.loc[i-1,['osc']] = oscillation
    return series

def simulate_ar(n_obs, sigma, list_alphas, starting_values):
    list_y = starting_values.copy()
    alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7 = list_alphas[0],list_alphas[1],list_alphas[2],list_alphas[3],list_alphas[4],list_alphas[5],list_alphas[6]
    for x in range(7, n_obs + 7, 1):
        error = np.random.normal(0, sigma, 1)
        new_y = alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  alpha_3 * list_y[x-3] +  alpha_4 * list_y[x-4]
        new_y +=  alpha_5 * list_y[x-5] +  alpha_6 * list_y[x-6] +  alpha_7 * list_y[x-7]  + error[0]
        list_y.append(new_y)
    del list_y[0:7]    
    return list_y

def simulate_ar_incremental(n_obs, sigma_new, sigma_old, speed, list_alphas, list_old_alphas, starting_values):
    list_y = starting_values.copy()
    concept_count = 1
    for x in range(7, n_obs + 7, 1):
        if concept_count <=speed:
            weight_old = (speed-concept_count)/speed
            weight_new = concept_count/speed
    
        alpha_1 = weight_new*list_alphas[0] + weight_old*list_old_alphas[0]
        alpha_2 = weight_new*list_alphas[1] + weight_old*list_old_alphas[1]
        alpha_3 = weight_new*list_alphas[2] + weight_old*list_old_alphas[2]
        alpha_4 = weight_new*list_alphas[3] + weight_old*list_old_alphas[3]
        alpha_5 = weight_new*list_alphas[4] + weight_old*list_old_alphas[4]
        alpha_6 = weight_new*list_alphas[5] + weight_old*list_old_alphas[5]
        alpha_7 = weight_new*list_alphas[6] + weight_old*list_old_alphas[6]       
    
        sigma = weight_new*sigma_new + weight_old*sigma_old
        error = np.random.normal(0, sigma, 1) 
        new_y = alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  alpha_3 * list_y[x-3] +  alpha_4 * list_y[x-4]
        new_y +=  alpha_5 * list_y[x-5] +  alpha_6 * list_y[x-6] +  alpha_7 * list_y[x-7]  + error[0]
        list_y.append(new_y)
        concept_count+=1
    del list_y[0:7]    
    return list_y
def simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values):
    list_error = starting_values.copy()
    list_y = []
    alpha_1, alpha_2, alpha_3, alpha_4 = list_alphas[0],list_alphas[1],list_alphas[2],list_alphas[3]
    for x in range(2, n_obs + 2, 1):
        error = np.random.normal(0, sigma, 1)
        new_y = error[0] + alpha_1 * list_error[x-1] +  alpha_2 * list_error[x-2] +  alpha_3 * list_error[x-1]*list_error[x-2]
        new_y += alpha_4 * list_error[x-2] * list_error[x-2]
        list_y.append(new_y)
        list_error.append(error[0])   
    return list_y, list_error
def simulate_non_linear_moving_average_incremental(n_obs, sigma_new, sigma_old, speed,list_alphas, list_old_alphas, starting_values):
    list_error = starting_values.copy()
    list_y = []
    concept_count = 1
    for x in range(2, n_obs + 2, 1):
        if concept_count <=speed:
            weight_old = (speed-concept_count)/speed
            weight_new = concept_count/speed
            
        alpha_1 = weight_new*list_alphas[0] + weight_old*list_old_alphas[0]
        alpha_2 = weight_new*list_alphas[1] + weight_old*list_old_alphas[1]
        alpha_3 = weight_new*list_alphas[2] + weight_old*list_old_alphas[2]
        alpha_4 = weight_new*list_alphas[3] + weight_old*list_old_alphas[3]
        sigma = weight_new*sigma_new + weight_old*sigma_old

        error = np.random.normal(0, sigma, 1)
        new_y = error[0] + alpha_1 * list_error[x-1] +  alpha_2 * list_error[x-2] +  alpha_3 * list_error[x-1]*list_error[x-2]
        new_y += alpha_4 * list_error[x-2] * list_error[x-2]
        list_y.append(new_y)
        list_error.append(error[0])   
    return list_y, list_error
def simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values): 
    list_y = starting_values.copy()
    alpha_1, alpha_2, alpha_3, alpha_4 = list_alphas[0],list_alphas[1],list_alphas[2],list_alphas[3]
    for x in range(4, n_obs + 4, 1):
        error = np.random.normal(0, sigma, 1)
        new_y = (alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  alpha_3 * list_y[x-3] +  alpha_4 * list_y[x-4])*math.pow(1-math.exp(-10*list_y[x-1]), -1)
        new_y +=   error[0]
        list_y.append(new_y)
    del list_y[0:4]    
    return list_y
def simulate_smooth_transitition_ar_incremental(n_obs, sigma_new, sigma_old, speed, list_alphas, list_old_alphas, starting_values): 
    list_y = starting_values.copy()
    concept_count = 1
    for x in range(4, n_obs + 4, 1):
        if concept_count <=speed:
            weight_old = (speed-concept_count)/speed
            weight_new = concept_count/speed
        
        alpha_1 = weight_new*list_alphas[0] + weight_old*list_old_alphas[0]
        alpha_2 = weight_new*list_alphas[1] + weight_old*list_old_alphas[1]
        alpha_3 = weight_new*list_alphas[2] + weight_old*list_old_alphas[2]
        alpha_4 = weight_new*list_alphas[3] + weight_old*list_old_alphas[3]
        sigma = weight_new*sigma_new + weight_old*sigma_old
        
        error = np.random.normal(0, sigma, 1)
        new_y = (alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  alpha_3 * list_y[x-3] +  alpha_4 * list_y[x-4])*math.pow(1-math.exp(-10*list_y[x-1]), -1)
        new_y +=   error[0]
        list_y.append(new_y)
    del list_y[0:4]    
    return list_y
def simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values): 
    list_y = starting_values.copy()
    alpha_1, alpha_2, alpha_3, alpha_4 = list_alphas[0],list_alphas[1],list_alphas[2],list_alphas[3]
    for x in range(2, n_obs + 2, 1):
        error = np.random.normal(0, sigma, 1)
        new_y = alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  (alpha_3 * list_y[x-1] +  alpha_4 * list_y[x-2])*math.pow(1-math.exp(-10*list_y[x-1]), -1)
        new_y += error[0]
        #print(new_y)
        list_y.append(new_y)
    del list_y[0:2]    
    return list_y
def simulate_smooth_transitition_ar_2_incremental(n_obs, sigma_new, sigma_old, speed, list_alphas, list_old_alphas, starting_values): 
    list_y = starting_values.copy()
    concept_count = 1
    for x in range(2, n_obs + 2, 1):
        if concept_count <=speed:
            weight_old = (speed-concept_count)/speed
            weight_new = concept_count/speed
        
        alpha_1 = weight_new*list_alphas[0] + weight_old*list_old_alphas[0]
        alpha_2 = weight_new*list_alphas[1] + weight_old*list_old_alphas[1]
        alpha_3 = weight_new*list_alphas[2] + weight_old*list_old_alphas[2]
        alpha_4 = weight_new*list_alphas[3] + weight_old*list_old_alphas[3]
        sigma = weight_new*sigma_new + weight_old*sigma_old

        error = np.random.normal(0, sigma, 1)
        new_y = alpha_1 * list_y[x-1] +  alpha_2 * list_y[x-2] +  (alpha_3 * list_y[x-1] +  alpha_4 * list_y[x-2])*math.pow(1-math.exp(-10*list_y[x-1]), -1)
        new_y += error[0]
        #print(new_y)
        list_y.append(new_y)
    del list_y[0:2]    
    return list_y
def preprocess_timeseries(timeseries, windowsize=10):
    series = timeseries.copy()
    series = pd.DataFrame({"t":series})
    series = autocorrelations_in_window(windowsize, series)
    series = partial_autocorrelations_in_window(windowsize, series)
    series = features_in_window(windowsize, series)
    series = oscillation_behaviour_in_window(windowsize, series)
    lags = pd.concat([series["t"].shift(1), series["t"].shift(2), 
                      series["t"].shift(3),series["t"].shift(4),series["t"].shift(5)], axis=1)
    series["t-1"]= lags.iloc[:,0]
    series["t-2"]= lags.iloc[:,1]
    series["t-3"]= lags.iloc[:,2]
    series["t-4"]= lags.iloc[:,3]
    series["t-5"]= lags.iloc[:,4]
    series = mutual_info(windowsize, series)
    series = series[windowsize:]
    #series = series.reset_index(drop=True)
    stand = standardize(series.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']])
    series.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']] = stand
    return series
def bkps_stats(bkps_signal, signal, size_concepts, obs_amount_beyond_window, windowsize_preproc = 10):
    #TODO: test indices for bkp, since it gives out a breakpoint for 790?
    bkps = bkps_signal.copy()
    bkps = bkps[:-1]
    total_number_bkps = len(bkps)
    
    identified_bkps = 0
    list_delays = []
# =============================================================================
#     range1_result = list(x for x in bkps if 489 <= x <= 504)
#     range2_result = list(x for x in bkps if 989 <= x <= 1004)
#     range3_result = list(x for x in bkps if 1489 <= x <= 1504)
# =============================================================================
    range1_result = list(x for x in bkps if (size_concepts-windowsize_preproc) <= x <= (size_concepts - windowsize_preproc + 15 + obs_amount_beyond_window ))
    range2_result = list(x for x in bkps if (2*size_concepts-windowsize_preproc) <= x <= (2*size_concepts - windowsize_preproc + 15 + obs_amount_beyond_window))
    range3_result = list(x for x in bkps if (3*size_concepts-windowsize_preproc) <= x <= (3*size_concepts - windowsize_preproc + 15 + obs_amount_beyond_window))
    
    if len(range1_result)>=1:
        identified_bkps+=1;
        delay = range1_result[0] - (size_concepts-windowsize_preproc)
        list_delays.append(delay)
    if len(range2_result)>=1:
        identified_bkps+=1;
        delay = range2_result[0] - (2*size_concepts-windowsize_preproc)
        list_delays.append(delay)
    if len(range3_result)>=1:
        identified_bkps+=1;
        delay = range3_result[0] - (3*size_concepts-windowsize_preproc)
        list_delays.append(delay)
    
    miss_detected_bkps = total_number_bkps - identified_bkps
    not_detected_bkps =  3 - identified_bkps
    
    return [identified_bkps, not_detected_bkps, miss_detected_bkps, list_delays]

#@ray.remote
def analysis_rbf(penalization, iterations, data_creation_function, size_concepts, obs_amount_beyond_window, windowsize_preprocessing = 10):
    identified_bkps_total = 0
    not_detected_bkps_total = 0
    miss_detected_bkps_total = 0
    delays_score_total = 0
    
    for i in range(0, iterations, 1):
        print(i)
        data = data_creation_function()
        data = preprocess_timeseries(data, windowsize_preprocessing) #cuts out the first "windowsize_preprocessing" observations
        signal = data.loc[:,["t", 'pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']].to_numpy()
        algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal[:,1:])
        bkps = algo.predict(pen=penalization)
    
        
        result = bkps_stats(bkps, signal, size_concepts, obs_amount_beyond_window, windowsize_preproc = windowsize_preprocessing)
        identified_bkps = result[0]
        not_detected_bkps = result[1]
        miss_detected_bkps = result[2]
        list_delays = result[3]
        
        identified_bkps_total += identified_bkps
        not_detected_bkps_total += not_detected_bkps
        miss_detected_bkps_total += miss_detected_bkps
        delays_score_total += sum(list_delays)
        
    
    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
    
    return [precision, recall, average_delay]

def analysis_linear(penalization, iterations, data_creation_function, size_concepts, obs_amount_beyond_window):
    identified_bkps_total = 0
    not_detected_bkps_total = 0
    miss_detected_bkps_total = 0
    delays_score_total = 0
    
    for i in range(0, iterations, 1):
        print(i)
        data = data_creation_function()
        data = pd.DataFrame({"t":data})
        
        #data = preprocess_timeseries(data) #cuts out the first 10 observations
        
        lags = pd.concat([data["t"].shift(1), data["t"].shift(2), 
                      data["t"].shift(3),data["t"].shift(4),data["t"].shift(5)], axis=1)
        data["t-1"]= lags.iloc[:,0]
        data["t-2"]= lags.iloc[:,1]
        data["t-3"]= lags.iloc[:,2]
        data["t-4"]= lags.iloc[:,3]
        data["t-5"]= lags.iloc[:,4]
        data = mutual_info(10, data)
        data = data[10:]

        signal = data.loc[:,["t", 't-1','t-2', 't-3','t-4','t-5']].to_numpy()
        algo = rpt.Pelt(model="linear", min_size=2, jump=1).fit(signal)
        bkps = algo.predict(pen=penalization)
    
        
        result = bkps_stats(bkps, signal, size_concepts, obs_amount_beyond_window)
        identified_bkps = result[0]
        not_detected_bkps = result[1]
        miss_detected_bkps = result[2]
        list_delays = result[3]
        
        identified_bkps_total += identified_bkps
        not_detected_bkps_total += not_detected_bkps
        miss_detected_bkps_total += miss_detected_bkps
        delays_score_total += sum(list_delays)
        
    
    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
    
    return [precision, recall, average_delay]

def analysis_l2(penalization, iterations, data_creation_function, size_concepts, obs_amount_beyond_window):
    identified_bkps_total = 0
    not_detected_bkps_total = 0
    miss_detected_bkps_total = 0
    delays_score_total = 0
    
    for i in range(0, iterations, 1):
        print(i)
        data = data_creation_function()
        data = pd.DataFrame({"t":data})
        
        #data = preprocess_timeseries(data) #cuts out the first 10 observations
        
        data = data[10:]

        signal = data.loc[:,["t"]].to_numpy()
        algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)
        bkps = algo.predict(pen=penalization)
    
        result = bkps_stats(bkps, signal, size_concepts, obs_amount_beyond_window)
        identified_bkps = result[0]
        not_detected_bkps = result[1]
        miss_detected_bkps = result[2]
        list_delays = result[3]
        
        identified_bkps_total += identified_bkps
        not_detected_bkps_total += not_detected_bkps
        miss_detected_bkps_total += miss_detected_bkps
        delays_score_total += sum(list_delays)
        
    
    if  (identified_bkps_total + miss_detected_bkps_total)!=0:
        precision = identified_bkps_total/(identified_bkps_total + miss_detected_bkps_total)
    else:
        precision = 0
    recall = identified_bkps_total/(iterations*3)
    if identified_bkps_total!=0:
        average_delay = delays_score_total/identified_bkps_total
    else:
        average_delay = 0
    
    return [precision, recall, average_delay]


