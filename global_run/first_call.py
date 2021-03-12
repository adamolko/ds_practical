#!/usr/bin/env python3

import multiprocessing

# import code to run
import baseline
import xgboost_retrain
import xgboost_redefine
import xgboost_discard
import lstm_tuning


if __name__ == '__main__':
    iteration = 0
    processes = []
    list_of_functions = [xgboost_redefine.main, xgboost_discard.main, lstm_tuning.main]
    list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt", "nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]

    for fun in list_of_functions[:-1]:
        for name in list_of_names:
            p = multiprocessing.Process(target=fun, args=(iteration,name, ))
            processes.append(p)
            p.start()
    
    p = multiprocessing.Process(target = lstm_tuning.main, args = (iteration,))
    processes.append(p)
    p.start()

    for process in processes:
        process.join()
