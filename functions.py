# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:41:05 2020

@author: Daniel
"""

import statsmodels.tsa.stattools as arma_stats
from scipy.stats import kurtosis, skew
from statistics import variance


def transform_bkps_to_features(bkps, timeseries):
    list_concepts = []
    count_rows = timeseries.shape[0] 
    current_concept = 1
  
   # for x in range(1, count_rows+1):
    for x in range(0, count_rows):
        if (x in bkps): 
            current_concept+=1
        list_concepts.append(current_concept)

    
    timeseries["concept"] = list_concepts
    timeseries["concept"] = timeseries["concept"].astype("category")
    return timeseries


def standardize(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result

def autocorrelations_in_window(windowsize, series):
    series['acf0'], series['acf1'], series['acf2'], series['acf3'], series['acf4'], series['acf5'] = [0, 0, 0, 0, 0, 0]
    number_rows = series.shape[0] 
    for i in range(windowsize, number_rows, 1):
        window = series.iloc[i-windowsize:i+1]
        acfs = arma_stats.acf(window["t"], nlags=5)
        series.loc[i,['acf0','acf1','acf2', 'acf3', 'acf4', 'acf5' ]] = acfs     
    return series
 

def partial_autocorrelations_in_window(windowsize, series):
    series['pacf0'], series['pacf1'], series['pacf2'], series['pacf3'] = [0, 0, 0, 0]
    number_rows = series.shape[0] 
    for i in range(windowsize, number_rows, 1):
        window = series.iloc[i-windowsize:i+1]
        pacfs = arma_stats.pacf(window["t"], nlags=3)
        series.loc[i,['pacf0','pacf1','pacf2', 'pacf3']] = pacfs
    return series

def features_in_window(windowsize, series):
    series['var'], series['kurt'], series['skew'] = [0, 0, 0]
    number_rows = series.shape[0]
    for i in range(windowsize, number_rows, 1):
        window = series.iloc[i-windowsize:i+1]
        kurt = kurtosis(window["t"])
        skewness = skew(window["t"])
        var = variance(window["t"])
        series.loc[i,['var','kurt','skew']] = var, kurt, skewness
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
def oscillation_behaviour_in_window(windowsize, series):
    series['osc']= 0
    number_rows = series.shape[0]
    for i in range(windowsize, number_rows, 1):
        window = series.iloc[i-windowsize:i+1]
        points = turning_points(window["t"].values)
        sum_points = sum(len(x) for x in points)
        oscillation = sum_points/windowsize
        series.loc[i,['osc']] = oscillation
    return series
#TODO: some function to find oscillation of time series in window