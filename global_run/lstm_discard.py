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


def manual_preprocessing(values):
	#receives the list of values up until and including the test point

	columns = ["t", "t-1", "t-2", "t-3", "t-4", "t-5"]
	data = [values[-1], values[-2], values[-3], values[-4], values[-5], values[-6]]

	df = pd.DataFrame(columns=columns, data=[data])
	return df


def fit_lstm(train):
	# reshape training into [samples, timesteps, features]
	X, y = train.iloc[:, 1:], train.iloc[:, 0]

	#cannot reshape datafame
	X_arrays = np.asarray(X)
	X = np.hstack(X_arrays).reshape(X.shape[0], 1, X.shape[1])

	#build model
	model = Sequential()
	model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')

	es = tf.keras.callbacks.EarlyStopping(monitor='loss',  patience=5, verbose=0, mode='auto')

	# fit network
	model.fit(X, y, epochs = 710, batch_size = 80, verbose = 0, callbacks=[es], shuffle = False)

	return model

def is_enough(data):
	new_concept = max(list(data["concept"]))
	number_of_points = sum(data["concept"]==new_concept)
	# print(number_of_points)
	return number_of_points

def main(iteration, name):
	print("lstm with discard is running")
	smape_dict = {}
	# load the data
	data = pd.read_csv("data/"+name, usecols = [iteration]).iloc[:,0].to_list()

	#70/30 train/test split
	split = int(0.7*len(data))
	train, test = data[:split], data[split:]

	#get breakpoints for train set
	history = functions.ada_preprocessing(train)

	#note the last concept that appeared
	last_num_concepts = max(list(history["concept"]))

	model = fit_lstm(history.loc[:, "t":"t-5"])

	predictions = []
	points = 0
	bkp = None

	start = time.perf_counter()

	for i in range(0, len(test)):
		print("lstm with discard is alive")
		#get test observation into necessary shape
		train.append(test[i])
		test_row = manual_preprocessing(train)

		X = test_row.loc[:,"t-1":"t-5"]
		X_arrays = np.asarray(X)
		test_X = np.hstack(X_arrays).reshape(X.shape[0], 1, X.shape[1])

		#get predictions for new test observation
		prediction = model.predict(test_X)
		predictions.append(prediction)

		#new dataframe with the predicted test observation already appended
		history = functions.ada_preprocessing(train)
		if i == len(test)-1:
			bkp = history["concept"]

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
			# retrain the model
			model = fit_lstm(history.loc["t":"t-5"])
		#otherwise just keep using the same dataset


	end = time.perf_counter()
	print("Time wasted: {:.2f}h".format((end-start)/3600))

	#inverting predictions to original scale
	#     predictions = scaler.inverse_transform(np.asarray(predictions).reshape([-1,1]))

	error = smape(np.asarray(predictions), np.asarray(test))
	smape_dict[name] = error
	print("SMAPE: {:.4f}".format(error))

	plot_save(predictions, test, bkp, "results/lstm/discard/"+name, setback)

	#saving the dictionary containing errors
	dict_path = "results/lstm/discard/errors/error"+str(iteration)+name+".txt"
	with open(dict_path, 'w') as file:
		for key in smape_dict.keys():
			file.write("%s,%s\n"%(key,smape_dict[key]))