#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def smape(predictions, actual):
    difference = np.abs(predictions-actual)
    summation = np.abs(actual)+np.abs(predictions)
    error = np.mean(difference/summation)
    return error


def baseline_predict(train, test):
    train_array = np.asarray(train.iloc[:,0].tail(1))
    test_array = np.asarray(test.iloc[:-1, 0])
    
    predictions = np.concatenate([train_array, test_array])

    return predictions

def main(iteration, name):
    smape_dict = {}

    data = pd.read_csv("data/"+name, usecols = [iteration])
    split = int(0.7*len(data))
    train, test = data.loc[:split, :], data.loc[split:, :]
    ground_truth = data.iloc[int(0.7*len(data)):, 0].reset_index(drop = True)

    predictions = baseline_predict(train, test)


    error = smape(predictions, ground_truth.values.reshape([-1,1]))
    smape_dict[name] = error
    print("SMAPE: {:.4f}".format(error))

    plt.plot(ground_truth, label = "expected", color = "black")
    plt.plot(predictions, label = "predicted", color = "red")
    plt.title(name)
    plt.legend()    

    #saving the plots
    image_path = "results/baseline/"+name+".png"
    plt.savefig(image_path)
    plt.close()

    #saving the dictionary containing errors
    dict_path = "results/baseline/errors/error"+str(iteration)+name+".txt"
    with open(dict_path, 'w') as file:
        for key in smape_dict.keys():
            file.write("%s,%s\n"%(key,smape_dict[key]))

