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

def transform_bkps_to_features(bkps, timeseries):
    series = timeseries.copy()
    list_concepts = []
    count_rows = series.shape[0] 
    current_concept = 1
  
   # for x in range(1, count_rows+1):
    for x in range(0, count_rows):
        if (x in bkps): 
            current_concept+=1
        list_concepts.append(current_concept)

    
    series["concept"] = list_concepts
    series["concept"] = series["concept"].astype("category")
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
        acfs = arma_stats.acf(window["t"], nlags=5)
        series.loc[i-1,['acf0','acf1','acf2', 'acf3', 'acf4', 'acf5' ]] = acfs     
    return series
 

def partial_autocorrelations_in_window(windowsize, timeseries):
    series = timeseries.copy()
    series['pacf0'], series['pacf1'], series['pacf2'], series['pacf3'] = [0, 0, 0, 0]
    number_rows = series.shape[0] 
    for i in range(windowsize, number_rows+1, 1):
        window = series.iloc[i-windowsize:i]
        pacfs = arma_stats.pacf(window["t"], nlags=3)
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
        print(new_y)
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
        print(new_y)
        list_y.append(new_y)
    del list_y[0:2]    
    return list_y
def preprocess_timeseries(timeseries):
    series = timeseries.copy()
    series = pd.DataFrame({"t":series})
    series = autocorrelations_in_window(10, series)
    series = partial_autocorrelations_in_window(10, series)
    series = features_in_window(10, series)
    series = oscillation_behaviour_in_window(10, series)
    lags = pd.concat([series["t"].shift(1), series["t"].shift(2), 
                      series["t"].shift(3),series["t"].shift(4),series["t"].shift(5)], axis=1)
    series["t-1"]= lags.iloc[:,0]
    series["t-2"]= lags.iloc[:,1]
    series["t-3"]= lags.iloc[:,2]
    series["t-4"]= lags.iloc[:,3]
    series["t-5"]= lags.iloc[:,4]
    series = mutual_info(10, series)
    series = series[10:]
    series = series.reset_index(drop=True)
    stand = standardize(series.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']])
    series.loc[:,['pacf1','pacf2', 'pacf3','acf1','acf2', 'acf3', 'acf4', 'acf5',
                                      'var','kurt','skew', 'osc', 'mi_lag1', 'mi_lag2', 'mi_lag3']] = stand
    return series