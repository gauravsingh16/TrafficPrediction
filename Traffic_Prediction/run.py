import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from Models.LSTM.model import Model
from Utils.InputSampler import InputSampler
from Experiments.datasetgenerator import DataLoader
from sklearn.metrics import mean_squared_error

def plot_training_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Train Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_validation_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Test Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_accuracy(accuracy, val_accuracy):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Val Accuracy')
    plt.legend()
    plt.show()

def plot_loss(loss, val_loss):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.show()

def main():

    configs = json.load(open('/home/gaurav/TrafficPrediction/Traffic_Prediction/Models/Configs.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    dataset = InputSampler()
    dataset.create_sample()
    model = Model()
    data = DataLoader(
        os.path.join('data',configs['data']['filename']),
        configs['data']['train_test_split']
    )
   
    data_train, data_test, y_train, y_test = data.create_database(configs['data']['columns'],
        configs['data']['cols'])
    x_train, y_train = data.get_train_data(data_train, y_train, seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'])
    x_test, y_tests = data.get_test_data(data_test, y_test, seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'])
    y_trains = np.array(y_train)
    x_trains = np.array(x_train)
    x_tests = np.array(x_test)
    y_tests = np.array(y_tests)   
    print(y_trains.shape, x_trains.shape, x_tests.shape, y_tests.shape)
    model.build_model(configs, data_train)
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    data_gen= data.generate_train_batch(
            data_train = data_train,
            seq_len = configs['data']['sequence_length'],
            batch_size= configs['training']['batch_size'],
            normalise = configs['data']['normalise'])
    len_train = len(data_train)
    print(data_gen)
    
    steps_per_epoch = math.ceil((len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    accuracy, val_accuracy, loss, val_loss = model.train_generator(
        x_train,
        y_train,
        x_test,
        y_test,
        data_gen,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch = steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    train_predictions, test_prediction = model.predict_point_by_point(x_train, x_test)

    trainScore = np.sqrt(mean_squared_error(y_train[:, 0, 0], train_predictions[:, 0, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(y_test[:, 0, 0], test_prediction[:, 0, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    yTrainPredict, yTrain, yTestPredict, yTest = data.transform(y_train[:, 0, 0], train_predictions, y_test, test_prediction)
    
    plot_training_results(yTrainPredict, yTrain)
    plot_validation_results(yTestPredict,yTest)
    plot_accuracy(accuracy, val_accuracy)
    plot_loss(loss, val_loss)

if __name__ == '__main__':
    main()