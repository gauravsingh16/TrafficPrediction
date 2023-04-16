import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataLoader():

	def __init__(self, filename, split):
		self.dataframe = pd.read_csv(filename)
		self.yscaler = MinMaxScaler(feature_range=(0,1))
		self.i_split = int(len(self.dataframe) * split)
	
	def create_database(self, data_columns, y_columns):

		le = LabelEncoder()
		scaler = MinMaxScaler(feature_range=(0,1))
		dataset = self.dataframe
		sender_label = self.dataframe.get(["Sender's IP"])
		receiver_label = self.dataframe.get(["Receiver's IP"])
		protocol_label = self.dataframe.get(["Protocol Stack"])

		dataset["Sender's IP"] = le.fit_transform(sender_label)
		dataset["Receiver's IP"] = le.fit_transform(receiver_label)
		dataset["Protocol Stack"] = le.fit_transform(protocol_label)
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace('.','')
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace(':','')
		dataset["SrNo"] = scaler.fit_transform(self.dataframe.get(["SrNo"]))
		dataset["Timestamp"] = scaler.fit_transform(self.dataframe.get(["Timestamp"]))
		dataset["Packets"] = scaler.fit_transform(self.dataframe.get(["Packets"]))
		dataset["Packets"] = self.yscaler.fit_transform(self.dataframe.get(["Packets"]))
		
		data_train = dataset.get(data_columns).values[:self.i_split]
		data_test  = dataset.get(data_columns).values[self.i_split:]
		y_train 	= dataset.get(y_columns).values[:self.i_split]
		y_test 	= dataset.get(y_columns).values[self.i_split:]

		return data_train, data_test, y_train, y_test

	def get_train_data(self, data_train, y_train):

		data_train = np.reshape(data_train, (data_train.shape[0], data_train.shape[1], 1))
		y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

		return data_train, y_train
	
	def get_test_data(self, data_test, y_test):
        
		data_test = np.reshape(data_test, (data_test.shape[0], data_test.shape[1], 1))
		y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

		return data_test,y_test

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(test_prediction[:, 0, 0].reshape(-1, 1))

		return yTrainPredict, yTrain, yTest, yTestPredict

	def generate_train_batch(self, seq_len, batch_size, normalise):
		'''Yield a generator of training data from filename on given list of cols split for train/test'''
		i = 0
		while i < (self.len_train - seq_len):
			x_batch = []
			y_batch = []
			for b in range(batch_size):
				if i >= (self.len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't divide evenly
					yield np.array(x_batch), np.array(y_batch)
					i = 0
				x, y = self._next_window(i, seq_len, normalise)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
		yield np.array(x_batch), np.array(y_batch)
