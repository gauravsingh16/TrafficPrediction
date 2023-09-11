import os
import time
import datetime as dt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, RepeatVector, CuDNNLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers


class MLP_AELSTM_Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs, data_train):

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			repeatvector_rate = layer['repeatvector_rate'] if 'repeatvector_rate' in layer else None

			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, return_sequences = return_seq,  input_shape=(input_timesteps, input_dim), activation = activation))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))
			if layer['type'] == 'Dense':
				self.model.add((Dense(neurons, activation=activation)))
			if layer['type'] == 'repeatVector':
				self.model.add(RepeatVector(repeatvector_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=optimizers.Adam(learning_rate = 0.001), metrics=['accuracy'])
	
		print('[Model] Model Compiled')

	def train_generator(self, data_train, y_train, data_test, y_test, epochs, batch_size,  save_dir):

		start_time = time.time()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=10),
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]

		history = self.model.fit(
			data_train,
			y_train,
			batch_size,
			epochs=epochs,
			validation_data=(data_test, y_test),
			callbacks=callbacks,
		)
		self.model.summary()
		time_elapsed = time.time()-start_time
		print('Time Taken for training %s' % time_elapsed)
		print('[Model] Training Completed. Model saved as %s' % save_fname)

		return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], time_elapsed

	def predict_point_by_point(self, train_data, test_data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		train_predict = self.model.predict(train_data)
		test_predict = self.model.predict(test_data)
		print(train_predict.shape, test_predict.shape)
		
		return train_predict, test_predict
	
