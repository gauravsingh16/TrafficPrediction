import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from Models.LSTM.model import Model
from Experiments.training import ModelTrainer
from sklearn.metrics import mean_squared_error

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Train Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():

    configs = json.load(open('/home/gaurav/TrafficPrediction/Traffic_Prediction/Models/Configs.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = ModelTrainer(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        configs['data']['cols']
    )

    model = Model()
    model_predict = Model()

    x_train, y_train = data.get_train_data()
    x_test, y_test = data.get_test_data()
    model.build_model(configs, x_train)

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    

    #predictions = model_predict.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    #predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    #plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    
    train_predictions, test_prediction = model_predict.predict_point_by_point(x_train, x_test)
    trainScore = np.sqrt(mean_squared_error(y_train[:, 0, 0], train_predictions[:, 0, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(y_test[:, 0, 0], test_prediction[:, 0, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    yTrainPredict, yTrain = data.transform(y_train, train_predictions, y_test, test_prediction)
    
    plot_results(yTrainPredict, yTrain)

if __name__ == '__main__':
    main()