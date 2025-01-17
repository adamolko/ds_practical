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
import csv
import json
import functions

def smape(predictions, actual):
	difference = np.abs(predictions-actual)
	summation = np.abs(actual)+np.abs(predictions)
	error = np.mean(difference/summation)
	return error

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
	model.fit(X, y, epochs = 710, batch_size = 80, verbose = 1, callbacks=[es], shuffle = False)

	return model


def main(iteration, name):
	print("lstm with oneshot is running")
	smape_dict = {}

	#loading the data
	data = pd.read_csv("data/"+name, usecols = [iteration]).iloc[:,0].to_list()

	#70/30 train/test split
	split = int(0.7*len(data))
	train, test = data[:split], data[split:]

	predictions = []

	# train the model outside the for-loop

	history = functions.ada_preprocessing(train)
	history = history.loc[:, "t":"t-5"]

	model = fit_lstm(history)	

	start = time.perf_counter()
	for i in range(0, len(test)):
		print("lstm with oneshot is alive")
		#get test observation into necessary shape
		train.append(test[i])
		test_row = manual_preprocessing(train)

		X = test_row.loc[:,"t-1":"t-5"]
		X_arrays = np.asarray(X)
		test_X = np.hstack(X_arrays).reshape(X.shape[0], 1, X.shape[1])

		#get predictions for new test observation
		prediction = model.predict(test_X)
		predictions.append(prediction)

		#get breakpoints for train dataset
		history = functions.ada_preprocessing(train)
		history = history.loc[:, "t":"t-5"]


	end = time.perf_counter()
	print("Time spent: {:.2f}h".format((end-start)/3600))

	#inverting predictions to original scale
	#     predictions = scaler.inverse_transform(np.asarray(predictions).reshape([-1,1]))

	error = smape(np.asarray(predictions), np.asarray(test))
	smape_dict[name] = error
	print("SMAPE: {:.4f}".format(error))

	plt.plot(test, label = "expected", color = "black")
	plt.plot(np.asarray(predictions).reshape([-1,1]), label = "predicted", color = "red")
	plt.title(name)
	plt.legend()
	image_path = "results/lstm/oneshot/"+name+".png"
	plt.savefig(image_path)
	plt.clf()

	#saving the dictionary containing errors
	dict_path = "results/lstm/oneshot/errors/error"+str(iteration)+name+".txt"
	with open(dict_path, 'w') as file:
		for key in smape_dict.keys():
			file.write("%s,%s\n"%(key,smape_dict[key]))
