#!/usr/bin/env python3

import functions
import pandas as pd
import numpy as np
import create_simdata

list_of_functions = [("linear1_abrupt",create_simdata.linear1_abrupt),
             ("linear2_abrupt",create_simdata.linear2_abrupt),
             ("linear3_abrupt",create_simdata.linear3_abrupt),
             ("linear1_inc", create_simdata.linear1_inc),
             ("linear2_inc", create_simdata.linear2_inc),
             ("linear3_inc", create_simdata.linear3_inc),
             ("nonlinear1_abrupt",create_simdata.nonlinear1_abrupt),
             ("nonlinear2_abrupt",create_simdata.nonlinear2_abrupt),
             ("nonlinear3_abrupt",create_simdata.nonlinear3_abrupt),
             ("nonlinear1_inc", create_simdata.nonlinear1_inc),
             ("nonlinear2_inc", create_simdata.nonlinear2_inc),
             ("nonlinear3_inc", create_simdata.nonlinear3_inc)]

for name, fun in list_of_functions:
	data = functions.ada_preprocessing(fun())
	path = "data/"+name
	data.to_csv(path, index = False)