#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
import json
import csv

def smape(predictions, actual):
	difference = np.abs(predictions-actual)
	summation = np.abs(actual)+np.abs(predictions)
	error = np.mean(difference/summation)
	return error

def plot_save(predictions, actual, bkp, name, setback):
	plt.plot(actual, label = "Expected", color = "black")
	plt.plot(predictions, label = "Predicted", color = "red")
	plt.legend()

	#saving the plots
	image_path = name+".png"
	plt.savefig(image_path)


	#retrieve rows with breakpoints
	bkps = []
	for i in bkp.unique()[1:]:
		bkps.append(np.where(bkp == i)[0][0])
	#     print(bkps)
	plot_bkps = [i-setback for i in bkps if i-setback>0]
	plt.vlines(x = plot_bkps, ymin = min(actual), ymax = max(actual), 
		linestyles = "dashed", color = "deepskyblue", label = "Breakpoints")
	plt.legend()
	image_path = name+"_breakpoints.png"
	plt.savefig(image_path)
	plt.clf()
	bkp_path = name+"_breakpoints.txt"
	with open(bkp_path, 'w') as file:
		file.write(json.dumps(",".join([str(j) for j in bkps])))