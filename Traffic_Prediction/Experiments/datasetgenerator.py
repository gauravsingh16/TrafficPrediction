import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class DataLoader():

	def __init__(self, filename, split):
		self.dataframe = pd.read_csv(filename)
		self.yscaler = MinMaxScaler(feature_range=(0, 1))
		self.i_split = int(len(self.dataframe) * split)

	def create_database(self, data_columns, y_columns):

		le = LabelEncoder()
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = self.dataframe
		sender_label = self.dataframe.get(["Sender's IP"])
		receiver_label = self.dataframe.get(["Receiver's IP"])
		protocol_label = self.dataframe.get(["Protocol Stack"])

		dataset["Sender's IP"] = le.fit_transform(sender_label)
		dataset["Receiver's IP"] = le.fit_transform(receiver_label)
		dataset["Protocol Stack"] = le.fit_transform(protocol_label)
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace('.', '')
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace(':', '')
		dataset["SrNo"] = scaler.fit_transform(self.dataframe.get(["SrNo"]))
		dataset["Timestamp"] = scaler.fit_transform(
		    self.dataframe.get(["Timestamp"]))
		dataset["Packets"] = scaler.fit_transform(self.dataframe.get(["Packets"]))
		dataset["Packets"] = self.yscaler.fit_transform(
		    self.dataframe.get(["Packets"]))
  
		data_train = dataset.get(data_columns).values[:self.i_split]
		data_test = dataset.get(data_columns).values[self.i_split:]
		y_train = dataset.get(y_columns).values[:self.i_split]
		y_test = dataset.get(y_columns).values[self.i_split:]
        
		return data_train, data_test, y_train, y_test

	def get_train_data(self, data_train, y_train, seq_len, normalise):

		data_x = []
		data_y = []
		len_train = len(data_train)
		for i in range(len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise, data_train, train = True)
			data_x.append(x)
			data_y.append(y)
		
		#data_train = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))
		#y_train = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 1))

		return data_x, data_y

	def get_test_data(self, data_test, y_test, seq_len, normalise):

		data_x = []
		data_y = []
		len_test = len(data_test)
		for i in range(len_test - seq_len):
			x,y = self._next_window(i, seq_len, normalise, data_test, train = False)
			data_x.append(x)
			data_y.append(y)
   
		
		#data_test = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))
		#y_test = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 1))

		return data_x, data_y

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(
		    train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(
		    test_prediction[:, 0, 0].reshape(-1, 1))

		return yTrainPredict, yTrain, yTest, yTestPredict

	def generate_train_batch(self, data_train, seq_len, batch_size, normalise):
		'''Yield a generator of training data from filename on given list of cols split for train/test'''

		i = 0
		len_train = len(data_train)
		while i < (len_train - seq_len):
			x_batch = []
			y_batch = []
			for b in range(batch_size):
				print(len_train)
				print(seq_len)
				if i >= (len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't divide evenly
					print("inside if generate train batch")
					yield np.array(x_batch), np.array(y_batch)
					i = 0
				x, y = self._next_window(i, seq_len, normalise, data_train, train = True)
				print(x.shape, y.shape)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
		print("inside generate train batch")
		
		return np.array(x_batch), np.array(y_batch)

	def _next_window(self, i, seq_len, normalise, data_train, train=True):
		'''Generates the next data window from the given index location i'''

		window = data_train[i:i+seq_len+1]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		
		x = window[:-1]
		y = window[-1, [0]]
		
		return x, y

	def normalise_windows(self, window_data, single_window=False):
		'''Normalise window with a base value of zero'''
		normalised_data = []
		eps = 0.00001
		window_data = [window_data] if single_window else window_data
		for window in window_data:
			normalised_window = []
			for col_i in range(window.shape[1]):
				normalised_col = [((float(p) / (float(window[0, col_i])+ eps )) - 1) for p in window[:, col_i]]
				normalised_window.append(normalised_col)
			normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
			normalised_data.append(normalised_window)
		return np.array(normalised_data)
