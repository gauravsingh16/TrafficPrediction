import numpy as np
from numpy import newaxis


def predict_point_by_point(self, data):
	#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	print('[Model] Predicting Point-by-Point...')
	predicted = self.model.predict(data)
	predicted = np.reshape(predicted, (predicted.size,))
	return predicted

def predict_sequences_multiple(self, data, window_size, prediction_len):
	#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
	print('[Model] Predicting Sequences Multiple...')
	prediction_seqs = []
	for i in range(int(len(data)/prediction_len)):
		curr_frame = data[i*prediction_len]
		predicted = []
		for j in range(prediction_len):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		prediction_seqs.append(predicted)
	return prediction_seqs

def predict_sequence_full(self, data, window_size):
	#Shift the window by 1 new prediction each time, re-run predictions on new window
	print('[Model] Predicting Sequences Full...')
	curr_frame = data[0]
	predicted = []
	for i in range(len(data)):
		predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
		curr_frame = curr_frame[1:]
		curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
	return predicted