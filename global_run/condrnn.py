#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from tqdm import tqdm
import time
from cond_rnn import ConditionalRNN
import csv
import tensorflow as tf
import functions
import json

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
	for i in bkp.unique():
		bkps.append(np.where(bkp == i)[0][0])
	plot_bkps = [i-setback for i in bkps if i-setback>0]
	plt.vlines(x = plot_bkps, ymin = min(actual), ymax = max(actual), 
		linestyles = "dashed", color = "deepskyblue", label = "Breakpoints")
	plt.legend()
	image_path = name+"_breakpoints.png"
	plt.savefig(image_path)
	plt.clf()
	
	bkp_path = name+"_breakpoints.txt"
	with open(bkp_path, 'w') as file:
		file.write(json.dumps("".join([str(j) for j in bkps])))

def manual_preprocessing(values, pattern):
	#receives the list of values up until and including the test point

	columns = list(pattern.columns)
	data = [values[-1], values[-2], values[-3], values[-4], values[-5], values[-6], pattern.loc[:,"concept"]]

	df = pd.DataFrame(columns=columns, data=[data])
	return df

def forecast_preprocessing(train, test):
	train_X, train_y = train.iloc[:,1:], train.iloc[:,0]
	test_X, test_y = test.iloc[:,1:], test.iloc[:,0]

	#separate both train and test sets into inputs and auxiliary variables
	train_X_input = train_X.loc[:,"t-1":"t-5"]
	train_X_aux = train_X.loc[:,"concept"]
	test_X_input = test_X.loc[:,"t-1":"t-5"]
	test_X_aux = test_X.loc[:,"concept"]

	#now also need to reshape X_input and X_aux
	X_arrays = np.asarray(train_X_input)
	train_X_input = np.hstack(X_arrays).reshape(train_X_input.shape[0], 1, train_X_input.shape[1])

	X_arrays = np.asarray(train_X_aux)
	train_X_aux = np.hstack(X_arrays).reshape(train_X_aux.shape[0], 1)

	#need to do the same for test set
	X_arrays = np.asarray(test_X_input)
	test_X_input = np.hstack(X_arrays).reshape(test_X_input.shape[0], 1, test_X_input.shape[1])

	X_arrays = np.asarray(test_X_aux)
	test_X_aux = np.hstack(X_arrays).reshape(test_X_aux.shape[0], 1)

	return train_X_input, train_X_aux, test_X_input, test_X_aux, train_y, test_y


def fit_cond_rnn(X_input, X_aux, train_y):
	inputs = Input(shape=(X_input.shape[1], X_input.shape[2]), dtype = tf.float32)
	cond1 = Input(shape = (X_aux.shape[1]), dtype = tf.float32)

	#building model steps
	A = ConditionalRNN(4, cell='LSTM')([inputs, cond1])
	out = Dense(1)(A)
	model = Model(inputs=[inputs, cond1], outputs=out)
	model.compile(loss = "mean_squared_error", optimizer = "adam")

	es = tf.keras.callbacks.EarlyStopping(monitor='loss',  patience=5, verbose=0, mode='auto')

	#fitting the model
	model.fit([X_input, X_aux], train_y, epochs = 50, batch_size = 1, callbacks = [es], verbose = 0, shuffle = False)

	return model

def preprocessing(data):
	#so far these columns didn't improve the analysis, so just drop them
	data.drop(["transition", "steps_to_bkp", "steps_since_bkp"], axis = 1, inplace = True)

	#scaling
	scaler_x = MinMaxScaler(feature_range = (0,1))
	data.loc[:,"t-1":"t-5"] = scaler_x.fit_transform(data.loc[:,"t-1":"t-5"])

	#need separate scaler only for target
	scaler_y = MinMaxScaler(feature_range = (0,1))
	data.loc[:,"t"] = scaler_y.fit_transform(np.asarray(data.loc[:,"t"]).reshape([-1,1]))

	return data, scaler_x, scaler_y


def main(iteration, name):
	smape_dict = {}

	data = pd.read_csv("data/"+name, usecols = [iteration]).iloc[:,0].to_list()

	#70/30 train/test split
	split = int(0.7*len(data))
	train, test = data[:split], data[split:]
	setback = len(train)

	predictions = []

	start = time.perf_counter()


	#we've got the first prediction, so now start from 1
	for i in range(0, len(test)):
		print("cond_rnn is alive")
		#get breakpoints for train dataset
		history = functions.ada_preprocessing(train)

		history.drop(["transition", "steps_to_bkp", "steps_since_bkp"], axis = 1, inplace = True)

		#get the dataframe for new test observation
		train.append(test[i])
		test_row = manual_preprocessing(train, history.tail(1))

		#change train and test into form appropriate for CondRNN
		train_X_input, train_X_aux, test_X_input, test_X_aux, train_y, test_y = forecast_preprocessing(history, test_row)

		model = fit_cond_rnn(train_X_input, train_X_aux, train_y)

		#get predictions for new test observation
		prediction = model.predict([test_X_input, test_X_aux])
		predictions.append(prediction)


	end = time.perf_counter()
	print("Time wasted on cond_rnn: {:.2f}h".format((end-start)/3600))

	#inverting predictions to original scale
	#     predictions = scaler.inverse_transform(np.asarray(predictions).reshape([-1,1]))

	error = smape(np.asarray(predictions), np.asarray(test))
	smape_dict[name] = error
	plot_save(predictions, ground_truth, bkp, "results/cond_rnn/"+name, setback)
    
dict_path = "results/cond_rnn/errors/error"+str(iteration)+name+".txt"
with open(dict_path, 'w') as file:
	for key in smape_dict.keys():
		file.write("%s,%s\n"%(key,smape_dict[key]))