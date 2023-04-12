import numpy as np
import pandas as pd
import datetime as dt
from Models.LSTM.model import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class ModelTrainer:

	def __init__(self, filename, split, cols, cols1):
		le = LabelEncoder()
		scaler = MinMaxScaler(feature_range=(0,1))
		self.yscaler = MinMaxScaler(feature_range=(0,1))
		dataframe 		= pd.read_csv(filename)
		i_split 		= int(len(dataframe) * split)
		sender_label = dataframe.get(["Sender's IP"])
		receiver_label = dataframe.get(["Receiver's IP"])
		protocol_label = dataframe.get(["Protocol Stack"])

		dataframe["Sender's IP"] = le.fit_transform(sender_label)
		dataframe["Receiver's IP"] = le.fit_transform(receiver_label)
		dataframe["Protocol Stack"] = le.fit_transform(protocol_label)
		dataframe['TimeStamp'] = dataframe['TimeStamp'].str.replace('.','')
		dataframe['TimeStamp'] = dataframe['TimeStamp'].str.replace(':','')
		dataframe[["Sr.No.", "TimeStamp", "Packets"]] = scaler.fit_transform(dataframe.get(["Sr.No.", "TimeStamp", "Packets"]))
		dataframe[["Packets"]] = self.yscaler.fit_transform(dataframe.get(["Packets"]))
		self.data_train = dataframe.get(["Sr.No.", "TimeStamp", "Sender's IP", "Receiver's IP", "Protocol Stack"]).values[:i_split]
		self.data_test  = dataframe.get(["Sr.No.", "TimeStamp", "Sender's IP", "Receiver's IP", "Protocol Stack"]).values[i_split:]
		self.y_train 	= dataframe.get(["Packets"]).values[:i_split]
		self.y_test 	= dataframe.get(["Packets"]).values[i_split:]
		
		self.len_train  = len(self.data_train)
		self.len_test   = len(self.data_test)

		self.len_train_windows = None
		self.model = Model()

	def get_train_data(self):

		data_train = np.reshape(self.data_train, (self.data_train.shape[0], self.data_train.shape[1], 1))
		y_train = np.reshape(self.y_train, (self.y_train.shape[0], self.y_train.shape[1], 1))

		return data_train, y_train
	
	def get_test_data(self):
        
		data_test = np.reshape(self.data_test, (self.data_test.shape[0], self.data_test.shape[1], 1))
		y_test = np.reshape(self.y_test, (self.y_test.shape[0], self.y_test.shape[1], 1))

		return data_test,y_test

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(test_prediction[:, 0, 0].reshape(-1, 1))
		print(yTrain.shape)
		print(yTrainPredict.shape)
		print(yTest.shape)
		print(yTestPredict.shape)
		plt.plot(yTrain, color='blue')
		plt.plot(self.y_train, color='red')
		plt.show()
		
		return yTrainPredict, yTrain

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
