#!/usr/bin/env python3

import numpy as np
import pandas as pd
import functions
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
import kerastuner as kt

def smape(predictions, actual):
	difference = np.abs(predictions-actual)
	summation = np.abs(actual)+np.abs(predictions)
	error = np.mean(difference/summation)
	return error

def fit_lstm(train, n_neuron, n_epoch, n_batch, optimizer):
	# reshape training into [samples, timesteps, features]
	X, y = train.iloc[:, 1:], train.iloc[:, 0]
	#cannot reshape datafame
	X_arrays = np.asarray(X)
	X = np.hstack(X_arrays).reshape(X.shape[0], 1, X.shape[1])

	#build model
	model = Sequential()
	model.add(LSTM(n_neuron, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	es = tf.keras.callbacks.EarlyStopping(
	    monitor='loss',  patience=5, verbose=0, mode='auto',
	)

	# fit network
	model.fit(X, y, epochs = n_epoch, batch_size = n_batch, verbose = 0, callbacks=[es], shuffle = False)

	return model


def main(iteration):
	print("hey i'm tuning lstm")

	data = pd.read_csv("data/linear1_abrupt", usecols = [iteration]).iloc[:,0].to_list()
	#note: i only use this to get the lagged values, the concepts and others are dropped subsequently
	data = functions.ada_preprocessing(data)
	data = data.loc[:, "t":"t-5"]

	#train/test split
	n = len(data)
	train, test = data[:int(0.7*n)], data[int(0.7*n):]

	X_test, y_test = test.loc[:,"t-1":"t-5"], test.loc[:,"t"]
	X_arrays = np.asarray(X_test)
	test_X = np.hstack(X_arrays).reshape(X_test.shape[0], 1, X_test.shape[1])

	optimizers = ["adam", "adamax", "rmsprop"]
	n_epochs = np.arange(10,1020, 100)
	n_batches = np.arange(1, 110, 20)
	n_neurons = [4,8,16,32,64]

	min_error = 100
	params = {}
	start = time.perf_counter()

	for optimizer in optimizers:
		for n_batch in n_batches:
			for n_neuron in n_neurons:
				for n_epoch in n_epochs:
					#fit the model just once
					print("lstm tuning is alive")
					model = fit_lstm(train, n_neuron, n_epoch, n_batch, optimizer)

					#get predictions for new test observation
					predictions = model.predict(test_X)
					#                 print("Time wasted: {:.2f}h".format((end-start)/3600))
					error = smape(np.asarray(predictions), np.asarray(y_test))
					if error<min_error:
						min_error
						params["optimizer"] = optimizer
						params["n_batch"] = n_batch
						params["n_neuron"] = n_neuron
						params["n_epoch"] = n_epoch
						print("SMAPE for lstm tunings {:.2f}".format(error))
						print(params)
						path = "results/tuning_results.csv"
						with open(path, 'w') as file:
							for key in params.keys():
								file.write("%s,%s\n"%(key,params[key]))
	end = time.perf_counter()
	print("Time wasted on NN tuning: {:.2f}m".format((end-start)/60))
