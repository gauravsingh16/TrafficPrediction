import os
import datetime as dt
from keras.datasets import mnist
from keras.optimizers import Adam
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:

	def __init__(self, filename, split, cols, cols1):
		dataframe = pd.read_csv(filename)
		i_split = int(len(dataframe) * split)
		self.data_train = dataframe.get(cols).values[:i_split]
		self.data_test  = dataframe.get(cols).values[i_split:]
		self.y_train = dataframe.get(cols1).values[:i_split]
		self.y_test = dataframe.get(cols1).values[i_split:]
		self.len_train  = len(self.data_train)
		self.len_test   = len(self.data_test)
		self.len_train_windows = None

	def get_train_data(self, seq_len, normalise):
		data_x = []
		data_y = []

		data_train = np.reshape(self.data_train, (self.data_train.shape[0], self.data_train.shape[1], 1))
		data_test = np.reshape(self.data_test, (self.data_test.shape[0], self.data_test.shape[1], 1))
		y_train = np.reshape(self.y_train, (self.y_train.shape[0], y_train.shape[1], 1))
		y_test = np.reshape(self.y_test, (self.y_test.shape[0], self.y_test.shape[1], 1))

		print(data_train.shape)

		return np.array(data_train), np.array(y_train)
	
	def get_test_data(self, seq_len, normalise):
		data_windows = []
		for i in range(self.len_test - seq_len):
			data_windows.append(self.data_test[i:i+seq_len])

		data_windows = np.array(data_windows).astype(float)
		data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

		x = data_windows[:, :-1]
		y = data_windows[:, -1, [0]]
		return x,y

	def train(self, data_train, y, epochs, batch_size, save_dir):
			
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]  
		self.model.fit(
			data_train,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):

		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)	
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
	 
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


	def _next_window(self, i, seq_len, normalise):
		'''Generates the next data window from the given index location i'''
		window = self.data_train[i:i+seq_len]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		x = window[:-1]
		y = window[-1, [0]]
		return x, y

	def normalise_windows(self, window_data, single_window=False):
		'''Normalise window with a base value of zero'''
		normalised_data = []
		window_data = [window_data] if single_window else window_data
		for window in window_data:
			normalised_window = []
			for col_i in range(window.shape[1]):
				normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
				normalised_window.append(normalised_col)
			normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
			normalised_data.append(normalised_window)
		return np.array(normalised_data)