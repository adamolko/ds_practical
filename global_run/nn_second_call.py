#!/usr/bin/env python3

import multiprocessing

import lstm_retrain
import lstm_discard
import condrnn

if __name__ == "__main__":
	processes = []
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt", "nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]
	for iteration in range(1, 3):
		for name in list_of_names:
			p = multiprocessing.Process(target = condrnn.main, args = (iteration, name, ))
			processes.append(p)
			p.start()

	for process in processes:
		process.join()
