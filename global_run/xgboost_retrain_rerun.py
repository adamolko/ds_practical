#!/usr/bin/env python3

import multiprocessing

# import code to run
import xgboost_retrain

if __name__ == '__main__':
	processes = []
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt", "nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]

	for iteration in range(1,4):
		print("iteration ", str(iteration))
		for name in list_of_names:
			p = multiprocessing.Process(target = xgboost_retrain.main, args = (iteration, name,))
			processes.append(p)
			p.start()

	for process in processes:
		process.join()
