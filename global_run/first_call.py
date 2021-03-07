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
    # starttime = time.time()
    processes = []
    list_of_functions = [baseline.main, xgboost_retrain.main, 
        xgboost_redefine.main, xgboost_discard.main, lstm_tuning.main]

    for fun in list_of_functions:
        p = multiprocessing.Process(target=fun, args=(iteration,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()