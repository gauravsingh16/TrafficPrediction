import numpy as np
import pandas as pd
import random
import csv
from matplotlib import pyplot
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataLoader():

	def __init__(self, filename, split):
		self.dataframe = pd.read_csv(filename)
		self.yscaler = MinMaxScaler(feature_range=(0,1))
		#self.i_split = int(len(self.dataframe) * split)
	
	def create_database(self, data_columns, predict_cols, sequence_length):

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
		data_x = self.split_sequences(dataset, sequence_length, sequence_length)
		i_split = int(len(data_x) * 0.80)

		train = data_x.iloc[:i_split, :]
		test = data_x.iloc[i_split:, :]
		train.drop(["Packets"], axis=1, inplace=True)
		test.drop(["Packets"], axis=1, inplace=True)
		self.data_train, self.yTrain  = train.iloc[:, :], train.iloc[:, -1]
		self.data_test, self.yTest  = test.iloc[:, :], test.iloc[:, -1]		

		#self.len_train = len(self.data_train)
		#self.len_y_train = len(self.yTrain)
		return self.data_train, self.data_test

	def get_train_data(self):

		x_train = np.reshape(self.data_train, (self.data_train.shape[0], 1, self.data_train.shape[1]))
		y_train = np.reshape(self.yTrain, (self.yTrain.shape[0], 1, 1))
		return x_train, y_train
	
	def get_test_data(self):
		
		x_test = np.reshape(self.data_test, (self.data_test.shape[0], 1, self.data_test.shape[1]))
		y_test = np.reshape(self.yTest, (self.yTest.shape[0], 1, 1))
		#x_test = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 6))
		#print(x_test.shape)
		#_, data_y = self.split_sequences(self.yTest, sequence_length ) 

		#y_test = np.reshape(self.yTest, (self.yTest.shape[0], self.yTest.shape[1], 1))
		
		return x_test,y_test

	def transform(self, y_train, train_predictions, y_test, test_prediction):
		yTrain = self.yscaler.inverse_transform(y_train.reshape(-1, 1))
		yTrainPredict = self.yscaler.inverse_transform(train_predictions[:, 0, 0].reshape(-1, 1))
		yTest = self.yscaler.inverse_transform(y_test.reshape(-1, 1))
		yTestPredict = self.yscaler.inverse_transform(test_prediction[:, 0, 0].reshape(-1, 1))

		return yTrainPredict, yTrain, yTest, yTestPredict

  
	def split_sequences(self, data_sequences, input_sequence, output_sequence, dropnan = True):
		n_vars = 1 if type(data_sequences) is list else data_sequences.shape[1]
		x, y = list(), list()
		for i in range(input_sequence, 0, -1):
			x.append(self.dataframe.shift(i))
			y += [('var%d(t-%d)'% (j+1, i)) for j in range(n_vars)]
 		# find the end of this pattern
		for i in range(0, output_sequence):
			x.append(self.dataframe.shift(-i))
			if i == 0:
				y += [('var%d(t)'% (j+1)) for j in range(n_vars)]
			else:
				y += [('var%d(t-%d)'% (j+1, i)) for j in range(n_vars)]
		
		agg = concat(x, axis=0)
		agg.x = y
 		# gather input and output parts of the pattern
		if dropnan:
			agg.dropna(inplace=True)
		return agg
  
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
  
