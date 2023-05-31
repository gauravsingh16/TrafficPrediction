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
	
	def create_database(self, data_columns, predict_cols):

		le = LabelEncoder()
		scaler = MinMaxScaler(feature_range=(0,1))
		dataset = self.dataframe
		self.dataframe.drop('SrNo', axis=1, inplace=True)
		sender_label = self.dataframe.get(["Sender's IP"])
		receiver_label = self.dataframe.get(["Receiver's IP"])
		protocol_label = self.dataframe.get(["Protocol Stack"])
		print(dataset)
		dataset["Sender's IP"] = le.fit_transform(sender_label)
		dataset["Receiver's IP"] = le.fit_transform(receiver_label)
		dataset["Protocol Stack"] = le.fit_transform(protocol_label)
		
		dataset.set_index('Timestamp', inplace = True, drop = False)
  
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace('.','')
		dataset['Timestamp'] = self.dataframe['Timestamp'].str.replace(':','')

		dataset["Timestamp"] = scaler.fit_transform(dataset.get(["Timestamp"]))
		dataset["Packets"] = scaler.fit_transform(dataset.get(["Packets"]))

		
		print(dataset)
		self.feature_len = len(data_columns)
		self.data_train = dataset.get(data_columns).values[:self.i_split]
		self.data_test  = dataset.get(data_columns).values[self.i_split:]
  
		dataset["Packets"] = self.yscaler.fit_transform(self.dataframe.get(["Packets"]))	
  	
		self.yTrain = dataset.get(predict_cols).values[:self.i_split]
		self.yTest = dataset.get(predict_cols).values[self.i_split:]
		print(dataset)

		self.len_train = len(self.data_train)
		self.len_y_train = len(self.yTrain)
		return self.data_train, self.data_test

	def get_train_data(self, data_train, sequence_length):
		
		data_x, data_y = self.split_sequences(self.data_train, self.yTrain, sequence_length )
		
		x_train = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 4))
  
		y_train = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 1))
	
		return x_train, y_train
	
	def get_test_data(self, data_test, sequence_length):
        
		data_x, data_y = self.split_sequences(self.data_test, self.yTest, sequence_length ) 
  
		x_test = np.reshape(data_x, (data_x.shape[0], data_x.shape[1],4))
  
		#_, data_y = self.split_sequences(self.yTest, sequence_length ) 

		y_test = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 1))
		
		return x_test,y_test

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(test_prediction[:, 0, 0].reshape(-1, 1))

		return yTrainPredict, yTrain, yTest, yTestPredict

  
	def split_sequences(self, sequences, y_sequence, sequence_length):
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
			seq_y = y_sequence[end_ix-1:out_ix, : ]
	
			X.append(seq_x)
			y.append(seq_y)
			
		return np.array(X), np.array(y)	
  
  
