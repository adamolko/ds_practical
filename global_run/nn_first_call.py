#!/usr/bin/env python3

import multiprocessing

# import code to run
import lstm_retrain
import lstm_discard
import condrnn

if __name__ == '__main__':
	iteration = 0
	processes = []
	list_of_functions = [lstm_discard.main]
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt", "nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]
	#list_of_names = ["linear1_abrupt"]
	for fun in list_of_functions:
		for name in list_of_names:
			p = multiprocessing.Process(target=fun, args=(iteration,name, ))
			processes.append(p)
			p.start()

	for process in processes:
		process.join()
