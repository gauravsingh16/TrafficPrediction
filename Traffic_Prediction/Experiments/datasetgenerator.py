import numpy as np
import pandas as pd
import random
import csv
from matplotlib import pyplot
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
		minValues1 = []
		secValues = []
		randLink = []
		dataset.drop("SrNo", axis = 1, inplace = True)
		sender_label = self.dataframe.get(["Sender's IP"])
		receiver_label = self.dataframe.get(["Receiver's IP"])
		protocol_label = self.dataframe.get(["Protocol Stack"])
		#print(dataset)
		dataset["Sender's IP"] = le.fit_transform(sender_label)
		dataset["Receiver's IP"] = le.fit_transform(receiver_label)
		dataset["Protocol Stack"] = le.fit_transform(protocol_label)
		
		connectivity_type = [0, 1]
		for i in range(len(dataset)):
			randValue = random.choice(connectivity_type)
			randLink.append(randValue)
          
		dataset['Connectivity'] = randLink
		#dataset.drop_duplicates(subset=['Timestamp'], keep = 'first', inplace = True)
		
		#dataset.drop_duplicates(subset=['Timestamp'], keep = 'first', inplace = True)
		#self.actual_traffic(dataset)
		#dataset.set_index('Timestamp', inplace = True)
		for i in self.dataframe['Timestamp']:
			minValues = i.split(':')
			minValues1.append(minValues[0])
			#print(len(minValues1))
			#secValues = self.dataframe['Timestamp'].str.split(':')[1]
			secValues.append(minValues[1])
  		
		dataset['Seconds'] = secValues
		dataset["Minutes"] = minValues1
		dataset.set_index('Timestamp', inplace = True)
		dataset["Seconds"] = scaler.fit_transform(dataset.get(["Seconds"]))
		dataset["Minutes"] = scaler.fit_transform(dataset.get(["Minutes"]))
		dataset["Packets"] = scaler.fit_transform(self.dataframe.get(["Packets"]))
		dataset["Packets"] = self.yscaler.fit_transform(self.dataframe.get(["Packets"]))	
		#print(dataset)
		self.feature_len = len(data_columns)
		self.data_train = dataset.get(data_columns).values[:self.i_split]
		self.data_test  = dataset.get(data_columns).values[self.i_split:]		
		#del dataset['SrNo']
		#dataset.drop(["SrNo"], axis = 1, inplace = True)
		self.yTrain = dataset.get(predict_cols).values[:self.i_split]
		self.yTest = dataset.get(predict_cols).values[self.i_split:]
		print(dataset)

		self.len_train = len(self.data_train)
		self.len_y_train = len(self.yTrain)
		return self.data_train, self.data_test

	def get_train_data(self, data_train, sequence_length):
		data_x, data_y = self.split_sequences(self.data_train, self.yTrain, sequence_length )
		#print(data_x.shape)
		x_train = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 6))
  
		y_train = np.reshape(data_y, (data_y.shape[0], data_y.shape[1], 1))
		#print(x_train.shape)
		return x_train, y_train
	
	def get_test_data(self, data_test, sequence_length):
		#print(self.data_test)
		data_x, data_y = self.split_sequences(self.data_test, self.yTest, sequence_length ) 
		#print(data_y)
		x_test = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 6))
  
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
			
			seq_y = y_sequence[end_ix-1:out_ix, : ]
	
			X.append(seq_x)
			y.append(seq_y)
			
		return np.array(X), np.array(y)	
  
	def actual_traffic(self, dataset):
		values = dataset.values
		
		
		groups = [1,2,3,4]
    
		i = 1
		pyplot.figure()
		for group in groups:
			
			pyplot.subplot(len(groups), 1, i)
			pyplot.plot(values[:, 0], values[:, group])
			#pyplot.plot(values[:, 0], values[:,1])
			#x = value[0]
			#y = value[4]
			#pyplot.plot(scalex = x,scaley = y)
			pyplot.title(dataset.columns[group], y =0.5, loc= 'right')
			pyplot.xlabel('Timestamp',  fontsize = 16, fontdict=dict(weight='bold'))
			pyplot.ylabel('Packets',  fontsize = 16, fontdict=dict(weight='bold'))
			i += 1
		pyplot.show()
  
