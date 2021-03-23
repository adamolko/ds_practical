#!/usr/bin/env python3

import multiprocessing

# import code to run
import baseline
import xgboost_retrain
import xgboost_redefine
import xgboost_discard

if __name__ == '__main__':
	processes = []
	list_of_functions = [baseline.main, xgboost_retrain.main, xgboost_redefine.main, xgboost_discard.main]
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt", "nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]

	for iteration in range(1,6):
		for fun in list_of_functions:
			for name in list_of_names:
				p = multiprocessing.Process(target = fun, args = (iteration, name,))
				processes.append(p)
				p.start()

		for process in processes:
			process.join()
