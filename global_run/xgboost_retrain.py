#!/usr/bin/env python3

from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time
import functions
import json

def smape(predictions, actual):
	difference = np.abs(predictions-actual)
	summation = np.abs(actual)+np.abs(predictions)
	error = np.mean(difference/summation)
	return error

# walk-forward validation for univariate data
def walk_forward_validation(train, test):
	predictions = []
	history = train.copy()

	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		#target y is the first column in the dataset
		test_X, test_y = test.iloc[i, 1:], test.iloc[i, 0]

		#test_X is a series, but our model was trained on dataframe
		test_X = test_X.to_frame().T.reset_index(drop = True)

		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, test_X)
		predictions.append(yhat)

		#appending test observation to training requires transforming it to dataframe
		new_data = test.iloc[i].to_frame().T.reset_index(drop = True)
		# add actual observation to training data for the next loop
		history = history.append(new_data, ignore_index = True)

	# estimate prediction error
	error = smape(predictions, test.iloc[:, 0], )

	return error, test.iloc[:, 0].reset_index(drop = True), predictions


def xgboost_forecast(train, test_X):
	train_X, train_y = train.iloc[:,1:], train.iloc[:,0]
	print("xgboost with retrain is alive")
	model = XGBRegressor(objective = 'reg:squarederror', n_estimators = 100, random_state = 40)
	model.fit(train_X, train_y)
	yhat = model.predict(test_X)

	return yhat[0]

def main(iteration):
	print("xgboost with retrain is running")
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt",
	"nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]

	smape_dict = {}

	for name in list_of_names:
		#loading the data
		#print("xgboost with retrain is alive")
		data = pd.read_csv("data/"+name, usecols = [iteration]).iloc[:,0].to_list()

		#note: i only use this to get the lagged values, the concepts and others are dropped subsequently
		data = functions.ada_preprocessing(data)
		data = data.loc[:, "t":"t-5"]

		#train/test split
		n = len(data)
		train, test = data[:int(0.7*n)], data[int(0.7*n):]    

		#fitting and plotting with concept
		start = time.perf_counter()
		error, y, yhat = walk_forward_validation(train, test)
		end = time.perf_counter()
		print("Time wasted on xgboost with retrain: {:.2f}s".format((end-start)))

		smape_dict[name] = error
		# print("SMAPE: {:.4f}".format(error))
		plt.plot(y, label = "Expected", color = "black")
		plt.plot(yhat, label = "Predicted", color = "red")
		plt.legend()
		plt.title(name)

		#saving the plots
		image_path = "results/xgboost/retrain/"+name+".png"
		plt.savefig(image_path)
		plt.clf()
		# plt.show()

	#saving the dictionary containing errors
	dict_path = "results/xgboost/retrain/errors/error"+str(iteration)+".txt"
	with open(dict_path, 'w') as file:
		for key in smape_dict.keys():
			file.write("%s,%s\n"%(key,smape_dict[key]))
