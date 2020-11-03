# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 08:48:02 2020

@author: Daniel
"""


import numpy as np
import pandas as pd
import functions

def linear1_abrupt():
    n_obs = 250
    #Linear 1 - Abrupt
    #Concept 1
    lin1_abrupt = []
    #n_obs=500
    starting_values = [0, 0, 0, 1, 0.5, 1.5, 1.2]
    list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]
    sigma = 0.5
    lin1_abrupt = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
    #n_obs=500
    starting_values = lin1_abrupt[n_obs-7:]
    list_alphas = [-0.3, 1.4, 0.4, -0.5, 0, 0, 0]   
    sigma = 1.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin1_abrupt = [*lin1_abrupt, *new_data]
    ##########
    #Concept 3
    #n_obs=500
    starting_values = lin1_abrupt[(2*n_obs)-7:]
    list_alphas = [1.5, -0.4, -0.3, 0.2, 0, 0, 0]   
    sigma = 2.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin1_abrupt = [*lin1_abrupt, *new_data]
    ##########
    #Concept 4
    #n_obs=500
    starting_values = lin1_abrupt[(3*n_obs)-7:]
    list_alphas = [-0.1, 1.4, 0.4, -0.7, 0, 0, 0]   
    sigma = 3.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin1_abrupt = [*lin1_abrupt, *new_data]
    
    return lin1_abrupt
    
def linear2_abrupt():
    n_obs = 250
    ##########
    #Linear 2 - Abrupt
    #Concept 1
    lin2_abrupt = []
    #n_obs=500
    starting_values = [0, 0.8, 0.2, 1, 0.5, 1.5, 1.2]
    list_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]
    sigma = 0.5
    lin2_abrupt = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
    #n_obs=500
    starting_values = lin2_abrupt[n_obs-7:]
    list_alphas = [-0.1, 1.2, 0.4, 0.3, -0.2, -0.6, 0]   
    sigma = 1.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin2_abrupt = [*lin2_abrupt, *new_data]
    ##########
    #Concept 3
    #n_obs=500
    starting_values = lin2_abrupt[(2*n_obs)-7:]
    list_alphas = [1.2, -0.4, -0.3, 0.7, -0.6, 0.4, 0]   
    sigma = 2.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin2_abrupt = [*lin2_abrupt, *new_data]
    ##########
    #Concept 4
    #n_obs=500
    starting_values = lin2_abrupt[(3*n_obs)-7:]
    list_alphas = [-0.1, 1.1, 0.5, 0.2, -0.2, -0.5, 0]   
    sigma = 3.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin2_abrupt = [*lin2_abrupt, *new_data]
    
    return lin2_abrupt

def linear3_abrupt():
    n_obs = 250
    ##########
    #Linear 3 - Abrupt
    #Concept 1
    lin3_abrupt = []
    #n_obs=500
    starting_values = [0, 0, 0, 0, 0, 0.5, 1]
    list_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
    sigma = 0.5
    lin3_abrupt = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
   # n_obs=500
    starting_values = lin3_abrupt[n_obs-7:]
    list_alphas = [1.5, -0.5, 0, 0, 0, 0, 0]   
    sigma = 1.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin3_abrupt = [*lin3_abrupt, *new_data]
    ##########
    #Concept 3
   # n_obs=500
    starting_values = lin3_abrupt[(2*n_obs)-7:]
    list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]   
    sigma = 2.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin3_abrupt = [*lin3_abrupt, *new_data]
    ##########
    #Concept 4
    #n_obs=500
    starting_values = lin3_abrupt[(3*n_obs)-7:]
    list_alphas = [0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4]   
    sigma = 3.5
    new_data = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    lin3_abrupt = [*lin3_abrupt, *new_data]
    
    return lin3_abrupt

def linear1_inc():
    ##########
    #Linear 1 - Incremental
    #Concept 1
    lin1_inc = []
    n_obs=500
    starting_values = [0, 0, 0, 1, 0.5, 1.5, 1.2]
    list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]
    sigma = 0.5
    lin1_inc = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
    n_obs=500
    starting_values = lin1_inc[493:]
    list_alphas = [-0.3, 1.4, 0.4, -0.5, 0, 0, 0]  
    list_old_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0] 
    sigma = 1.5
    sigma_old = 0.5
    speed = 30
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
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
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
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
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
    lin1_inc = [*lin1_inc, *new_data]
    
    return lin1_inc

def linear2_inc():
    lin2_inc = []
    n_obs=500
    starting_values = [0, 0.8, 0.2, 1, 0.5, 1.5, 1.2]
    list_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]
    sigma = 0.5
    lin2_inc = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
    n_obs=500
    starting_values = lin2_inc[493:]
    list_alphas = [-0.1, 1.2, 0.4, 0.3, -0.2, -0.6, 0]
    list_old_alphas = [1, -0.6, 0.8, -0.5, -0.1, 0.3, 0]   
    sigma = 1.5
    sigma_old = 0.5
    speed = 30
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
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
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
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
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
    lin2_inc = [*lin2_inc, *new_data]
    
    return lin2_inc

def linear3_inc():
    ##########
    #Linear 3 - Incremental
    #Concept 1
    lin3_inc = []
    n_obs=500
    starting_values = [0, 0, 0, 0, 0, 0.5, 1]
    list_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
    sigma = 0.5
    lin3_inc = functions.simulate_ar(n_obs, sigma, list_alphas, starting_values)
    #Concept 2
    n_obs=500
    starting_values = lin3_inc[493:]
    list_alphas = [1.5, -0.5, 0, 0, 0, 0, 0]   
    list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
    sigma = 1.5
    sigma_old = 0.5
    speed = 30
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
    lin3_inc = [*lin3_inc, *new_data]
    ##########
    #Concept 3
    n_obs=500
    starting_values = lin3_inc[993:]
    list_alphas = [0.9, -0.2, 0.8, -0.5, 0, 0, 0]   
    list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
    sigma = 2.5
    sigma_old = 1.5
    speed = 30
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
    lin3_inc = [*lin3_inc, *new_data]
    ##########
    #Concept 4
    n_obs=500
    starting_values = lin3_inc[1493:]
    list_alphas = [0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4]   
    list_old_alphas = [0.5, 0.5, 0, 0, 0, 0, 0]
    sigma = 3.5
    sigma_old = 2.5
    speed = 30
    new_data = functions.simulate_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
    lin3_inc = [*lin3_inc, *new_data]
    
    return lin3_inc

def nonlinear1_abrupt():
    ##########
    #Nonlinear 1 - Abrupt
    #Concept 1
    nonlin1_abrupt = []
    n_obs=500
    starting_values = [-0.5, 0.5] #here the errors are the starting values
    list_alphas = [0.9, -0.2, 0.8, -0.5]
    sigma = 0.5
    result = functions.simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values)
    nonlin1_abrupt = result[0]
    last_errors = result[1][500:]
    #Concept 2
    n_obs=500
    starting_values = last_errors
    list_alphas = [-0.3, 1.4, 0.4, -0.5]   
    sigma = 1.5
    result = functions.simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values)
    nonlin1_abrupt = [*nonlin1_abrupt, *result[0]]
    last_errors = result[1][500:]
    ##########
    #Concept 3
    n_obs=500
    starting_values = last_errors
    list_alphas = [1.5, -0.4, -0.3, 0.2]   
    sigma = 2.5
    result = functions.simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values)
    nonlin1_abrupt = [*nonlin1_abrupt, *result[0]]
    last_errors = result[1][500:]
    ##########
    #Concept 4
    n_obs=500
    starting_values = last_errors
    list_alphas = [-0.1, 1.4, 0.4, -0.7]   
    sigma = 3.5
    result = functions.simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values)
    nonlin1_abrupt = [*nonlin1_abrupt, *result[0]]
    last_errors = result[1][500:]
    
    return nonlin1_abrupt
    
def nonlinear2_abrupt():
    try:
        ##########
        #Nonlinear 2 - Abrupt
        #Concept 1
        nonlin2_abrupt = []
        n_obs=500
        starting_values = [1, 0.5, 1, 1.2]
        list_alphas = [0.9, -0.2, 0.8, -0.5]
        sigma = 0.5
        nonlin2_abrupt = functions.simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values)
        #Concept 2
        n_obs=500
        starting_values = nonlin2_abrupt[496:]
        list_alphas = [-0.3, 1.4, 0.4, -0.5]   
        sigma = 1.5
        new_data = functions.simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values)
        nonlin2_abrupt = [*nonlin2_abrupt, *new_data]
        ##########
        #Concept 3
        n_obs=500
        starting_values = nonlin2_abrupt[996:]
        list_alphas = [1.5, -0.4, -0.3, 0.2]   
        sigma = 2.5
        new_data = functions.simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values)
        nonlin2_abrupt = [*nonlin2_abrupt, *new_data]
        ##########
        #Concept 4
        n_obs=500
        starting_values = nonlin2_abrupt[1496:]
        list_alphas = [-0.1, 1.4, 0.4, -0.7]   
        sigma = 3.5
        new_data = functions.simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values)
        nonlin2_abrupt = [*nonlin2_abrupt, *new_data]
        
        return nonlin2_abrupt
    except:
        return nonlinear2_abrupt()
    
def nonlinear3_abrupt():
    try:
        ##########
        #Nonlinear 3 - Abrupt
        #Concept 1
        nonlin3_abrupt= []
        n_obs=500
        starting_values = [0.2, 0.1]
        list_alphas = [0.9, -0.2, 0.8, -0.5]
        sigma = 0.5
        nonlin3_abrupt = functions.simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values)
        #Concept 2
        n_obs=500
        starting_values = nonlin3_abrupt[498:]
        list_alphas = [-0.5, 0.4, 1.4, -0.3]   
        sigma = 1.5
        new_data = functions.simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values)
        nonlin3_abrupt = [*nonlin3_abrupt, *new_data]
        ##########
        #Concept 3
        n_obs=500
        starting_values = nonlin3_abrupt[998:]
        list_alphas = [1.5, -0.4, -0.3, 0.2]  
        sigma = 2.5
        new_data = functions.simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values)
        nonlin3_abrupt = [*nonlin3_abrupt, *new_data]
        ##########
        #Concept 4
        n_obs=500
        starting_values = nonlin3_abrupt[1498:]
        list_alphas = [-0.7, 0.4, 1.4, -0.1]   
        sigma = 3.5
        new_data = functions.simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values)
        nonlin3_abrupt = [*nonlin3_abrupt, *new_data]
        
        return nonlin3_abrupt
    except:
        return nonlinear3_abrupt()

def nonlinear1_inc():
    ##########
    #Nonlinear 1 - Incremental
    #Concept 1
    nonlin1_inc = []
    n_obs=500
    starting_values = [-0.5, 0.5] #here the errors are the starting values
    list_alphas = [0.9, -0.2, 0.8, -0.5]
    sigma = 0.5
    result = functions.simulate_non_linear_moving_average(n_obs, sigma, list_alphas, starting_values)
    nonlin1_inc = result[0]
    last_errors = result[1][500:]
    #Concept 2
    n_obs=500
    starting_values = last_errors
    list_alphas = [-0.3, 1.4, 0.4, -0.5]   
    list_old_alphas =  [0.9, -0.2, 0.8, -0.5]
    sigma = 1.5
    sigma_old = 0.5
    speed = 20
    result = functions.simulate_non_linear_moving_average_incremental(n_obs, sigma, sigma_old, speed,list_alphas, list_old_alphas, starting_values)
    nonlin1_inc = [*nonlin1_inc, *result[0]]
    last_errors = result[1][500:]
    ##########
    #Concept 3
    n_obs=500
    starting_values = last_errors
    list_alphas = [1.5, -0.4, -0.3, 0.2]   
    list_old_alphas = [-0.3, 1.4, 0.4, -0.5]   
    sigma = 2.5
    sigma_old = 1.5
    result = functions.simulate_non_linear_moving_average_incremental(n_obs, sigma, sigma_old, speed,list_alphas, list_old_alphas, starting_values)
    nonlin1_inc = [*nonlin1_inc, *result[0]]
    last_errors = result[1][500:]
    ##########
    #Concept 4
    n_obs=500
    starting_values = last_errors
    list_alphas = [-0.1, 1.4, 0.4, -0.7]   
    list_old_alphas = [1.5, -0.4, -0.3, 0.2]   
    sigma = 3.5
    sigma_old = 2.5
    result = functions.simulate_non_linear_moving_average_incremental(n_obs, sigma, sigma_old, speed,list_alphas, list_old_alphas, starting_values)
    nonlin1_inc = [*nonlin1_inc, *result[0]]
    last_errors = result[1][500:]
    
    return nonlin1_inc
def nonlinear2_inc():
    try:
        ##########
        #Nonlinear 2 - Incremental
        #Concept 1
        nonlin2_inc = []
        n_obs=500
        starting_values = [1, 0.5, 1, 1.2]
        list_alphas = [0.9, -0.2, 0.8, -0.5]
        sigma = 0.5
        nonlin2_inc = functions.simulate_smooth_transitition_ar(n_obs, sigma, list_alphas, starting_values)
        #Concept 2
        n_obs=500
        starting_values = nonlin2_inc[496:]
        list_alphas = [-0.3, 1.4, 0.4, -0.5]   
        list_old_alphas = [0.9, -0.2, 0.8, -0.5]
        sigma = 1.5
        sigma_old = 0.5
        speed = 20
        new_data = functions.simulate_smooth_transitition_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin2_inc = [*nonlin2_inc, *new_data]
        ##########
        #Concept 3
        n_obs=500
        starting_values = nonlin2_inc[996:]
        list_alphas = [1.5, -0.4, -0.3, 0.2]   
        list_old_alphas = [-0.3, 1.4, 0.4, -0.5]   
        sigma = 2.5
        sigma_old = 1.5
        new_data = functions.simulate_smooth_transitition_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin2_inc = [*nonlin2_inc, *new_data]
        ##########
        #Concept 4
        n_obs=500
        starting_values = nonlin2_inc[1496:]
        list_alphas = [-0.1, 1.4, 0.4, -0.7]   
        list_old_alphas = [1.5, -0.4, -0.3, 0.2]  
        sigma = 3.5
        sigma_old = 2.5
        new_data = functions.simulate_smooth_transitition_ar_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin2_inc = [*nonlin2_inc, *new_data]
        
        return nonlin2_inc
    except:
        return nonlinear2_inc()
    
def nonlinear3_inc():
    try:
        ##########
        #Nonlinear 3 - Incremental
        #Concept 1
        nonlin3_inc = []
        n_obs=500
        starting_values = [0.2, 0.1]
        list_alphas = [0.9, -0.2, 0.8, -0.5]
        sigma = 0.5
        nonlin3_inc = functions.simulate_smooth_transitition_ar_2(n_obs, sigma, list_alphas, starting_values)
        #Concept 2
        n_obs=500
        starting_values = nonlin3_inc[498:]
        list_alphas = [-0.5, 0.4, 1.4, -0.3] 
        list_old_alphas =  [0.9, -0.2, 0.8, -0.5]
        sigma = 1.5
        sigma_old = 0.5
        speed = 20
        new_data = functions.simulate_smooth_transitition_ar_2_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin3_inc = [*nonlin3_inc, *new_data]
        ##########
        #Concept 3
        n_obs=500
        starting_values = nonlin3_inc[998:]
        list_alphas = [1.5, -0.4, -0.3, 0.2]  
        list_old_alphas = [-0.5, 0.4, 1.4, -0.3] 
        sigma = 2.5
        sigma_old = 1.5
        new_data = functions.simulate_smooth_transitition_ar_2_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin3_inc = [*nonlin3_inc, *new_data]
        ##########
        #Concept 4
        n_obs=500
        starting_values = nonlin3_inc[1498:]
        list_alphas = [-0.7, 0.4, 1.4, -0.1]   
        list_old_alphas = [1.5, -0.4, -0.3, 0.2] 
        sigma = 3.5
        sigma_old = 2.5
        new_data = functions.simulate_smooth_transitition_ar_2_incremental(n_obs, sigma, sigma_old, speed, list_alphas, list_old_alphas, starting_values)
        nonlin3_inc = [*nonlin3_inc, *new_data]
        
        return nonlin3_inc
    
    except:
        return nonlinear3_inc
    