import numpy as np
import pandas as pd
import random
import csv
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
		
		dataset["Sender's IP"] = scaler.fit_transform(dataset.get(["Sender's IP"]))
		dataset["Receiver's IP"] = scaler.fit_transform(dataset.get(["Receiver's IP"]))
		dataset['Protocol Stack'] = scaler.fit_transform(dataset.get(['Protocol Stack']))
		dataset["Timestamp"] = scaler.fit_transform(dataset.get(["Timestamp"]))
		dataset["Packets"] = scaler.fit_transform(dataset.get(["Packets"]))
  
		dataset["Sender's IP"] = self.yscaler.fit_transform(self.dataframe.get(["Sender's IP"]))
		dataset["Receiver's IP"] = self.yscaler.fit_transform(self.dataframe.get(["Receiver's IP"]))
		dataset['Protocol Stack'] = self.yscaler.fit_transform(self.dataframe.get(["Protocol Stack"]))
		dataset["Timestamp"] = self.yscaler.fit_transform(self.dataframe.get(["Timestamp"]))
		dataset["Packets"] = self.yscaler.fit_transform(self.dataframe.get(["Packets"]))		
  
		dataset.drop_duplicates()

		self.feature_len = len(data_columns)
		self.data_train = dataset.get(data_columns).values[:self.i_split]
		self.data_test  = dataset.get(data_columns).values[self.i_split:]
		self.len_train = len(self.data_train)
  
		return self.data_train, self.data_test

	def get_train_data(self, data_train, sequence_length):
		
		data_x, data_y = self.split_sequences(self.data_train, sequence_length )
		
		x_train = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], self.feature_len))
		y_train = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 5))

		return x_train, y_train
	
	def get_test_data(self, data_test, sequence_length):
        
		data_x, data_y = self.split_sequences(self.data_test, sequence_length ) 
  
		x_test = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], self.feature_len))
		y_test = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 5))

		return x_test,y_test

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(test_prediction[:, 0, 0].reshape(-1, 1))

		return yTrainPredict, yTrain, yTest, yTestPredict

  
	def split_sequences(self, sequences, sequence_length):
		X, y = list(), list()
		for i in range(len(sequences)):
 		# find the end of this pattern
			end_ix = i + sequence_length
			out_ix = end_ix + sequence_length -1
 			# check if we are beyond the dataset
			if out_ix > len(sequences):
				break
 		# gather input and output parts of the pattern
			seq_x = sequences[i:end_ix, :]
			#print("teri behn ki chur", end_ix)
			seq_y = sequences[end_ix-1:out_ix, : ]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)	
  
  
