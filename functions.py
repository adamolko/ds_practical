# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:41:05 2020

@author: Daniel
"""


def transform_bkps_to_features(bkps, timeseries):
    list_concepts = []
    count_rows = timeseries.shape[0] 
    current_concept = 1
  
    for x in range(1, count_rows+1):
        if (x in bkps): 
            current_concept+=1
        list_concepts.append(current_concept)

    
    timeseries["concept"] = list_concepts
    timeseries["concept"] = timeseries["concept"].astype("category")
    return timeseries