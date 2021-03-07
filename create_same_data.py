#!/usr/bin/env python3

import functions
import pandas as pd
import numpy as np
import create_simdata
import random

random.seed(49)

list_of_functions = [("linear1_abrupt",create_simdata.linear1_abrupt),
             ("linear2_abrupt",create_simdata.linear2_abrupt),
             ("linear3_abrupt",create_simdata.linear3_abrupt),
             ("nonlinear1_abrupt",create_simdata.nonlinear1_abrupt),
             ("nonlinear2_abrupt",create_simdata.nonlinear2_abrupt),
             ("nonlinear3_abrupt",create_simdata.nonlinear3_abrupt)]

def create_data():

	for name, fun in list_of_functions:
		# create 10 variations of one series and store in dictionary
		series_dict = {}
		path = "global_run/data/"+name
		for i in range(1, 11):
			series = pd.Series(fun())
			series_dict["t{}".format(i)] = series.values
		data = pd.DataFrame.from_dict(series_dict)
		data.to_csv(path, index = False, columns = data.columns)

create_data()
