#!/usr/bin/env python3

from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time
import json
import functions

def smape(predictions, actual):
	difference = np.abs(predictions-actual)
	summation = np.abs(actual)+np.abs(predictions)
	error = np.mean(difference/summation)
	return error

def xgboost_forecast(train, test_X):
	train_X, train_y = train.iloc[:,1:], train.iloc[:,0]

	model = XGBRegressor(objective = 'reg:squarederror', n_estimators = 100, random_state = 40)
	model.fit(train_X, train_y)
	yhat = model.predict(test_X)

	return yhat[0]

def manual_preprocessing(values):
	#receives the list of values up until and including the test point

	columns = ["t", "t-1", "t-2", "t-3", "t-4", "t-5"]

	#retrieve lagged values
	data = [values[-1], values[-2], values[-3], values[-4], values[-5], values[-6]]
	df = pd.DataFrame(columns=columns, data=[data])
	return df

def is_enough(data):
	new_concept = max(list(data["concept"]))
	number_of_points = sum(data["concept"]==new_concept)
	# print(number_of_points)
	return number_of_points

def plot_save(predictions, actual, bkp, name):
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
	plt.vlines(x = bkps, ymin = actual.min(), ymax = actual.max(), 
		linestyles = "dashed", color = "deepskyblue", label = "Breakpoints")
	plt.legend()
	image_path = name+"_breakpoints.png"
	plt.savefig(image_path)
	plt.clf()
	bkp_path = name+"_breakpoints.txt"
	with open(bkp_path, 'w') as file:
		file.write(json.dumps("".join([str(j) for j in bkps])))

def main(iteration):
	print("xgboost with retrain is running")
	list_of_names = ["linear1_abrupt", "linear2_abrupt", "linear3_abrupt",
	"nonlinear1_abrupt", "nonlinear2_abrupt", "nonlinear3_abrupt"]

	smape_dict = {}

	for name in list_of_names:
		print("hey there, i'm running xgboost with discard")
		start = time.perf_counter()

		#loading the data
		data = pd.read_csv("data/"+name, usecols = [iteration]).iloc[:,0].to_list()

		#70/30 train/test split
		split = int(0.7*len(data))
		train, test = data[:split], data[split:]

		#get breakpoints for train set
		history = functions.ada_preprocessing(train)

		#note the last concept that appeared
		last_num_concepts = max(list(history["concept"]))

		predictions = []
		ground_truth = []
		points = 0
		for i in range(len(test)):
			#add new test observation to train series
			train.append(test[i])

			#pass all the values available in series up to and including the new test point
			test_df = manual_preprocessing(train)

			ground_truth.append(train[-1])

			#training data = history
			prediction = xgboost_forecast(history.loc[:,"t":"t-5"], test_df.loc[:,"t-1":"t-5"])
			predictions.append(prediction)

			#new dataframe with the predicted test observation already appended
			history = functions.ada_preprocessing(train)

			#note the real concept for the test observation
			new_num_concepts = max(list(history["concept"]))

			#if the number of concepts change, check if we have enough datapoints for new concept
			if new_num_concepts>last_num_concepts:
				#if we have more than 20 points for new concept, keep them and drop the rest of the data
				points = is_enough(history)
			if points>=20:
				history = history.tail(points)
				last_num_concepts = new_num_concepts
				points = 0
				#otherwise just keep using the same dataset

	        
	end = time.perf_counter()
	print("Time wasted on xgboost with discard: {:.2f}m".format((end-start)/60))

	error = smape(np.asarray(predictions), np.asarray(ground_truth))
	smape_dict[name] = error
	# print("SMAPE: {:.4f}".format(error))
	plot_save(predictions, ground_truth, bkp, "results/xgboost/discard/"+name)

    
	dict_path = "results/xgboost/discard/errors/error"+str(iteration)+".txt"
	with open(dict_path, 'w') as file:
		for key in smape_dict.keys():
			file.write("%s,%s\n"%(key,smape_dict[key]))