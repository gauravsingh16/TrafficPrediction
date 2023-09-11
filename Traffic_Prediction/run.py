import os
import json
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from Models.LSTM.model import LSTM_Model
from Models.AE_LSTM.model import AE_LSTM_Model
from Models.MLP_AELSTM.model import MLP_AELSTM_Model
from Utils.InputSampler import InputSampler
from Experiments.datasetgenerator import DataLoader
from sklearn.metrics import mean_squared_error

def plot_training_results(true_data, predicted_data ):
    plt.plot(true_data)
    plt.plot(predicted_data)
    plt.title('Predicted Data vs Train Data')
    plt.ylabel('MSE_value')
    plt.xlabel('Time(Minutes)')
    plt.legend([' Train Data', ' Predicted Data'], loc = 'upper right')
    plt.show()
    
def plot_validation_results(true_data, predicted_data ):
    plt.plot(true_data)
    plt.plot(predicted_data)
    plt.title('Predicted Data vs Test Data')
    plt.ylabel('MSE_value')
    plt.xlabel('Time(Minutes)')
    plt.legend([' Test Data', 'Predicted Data'], loc = 'upper right')
    plt.show()


def plot_accuracy(accuracy, val_accuracy):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Val Accuracy')
    plt.legend()
    plt.show()

def plot_loss(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model train vs Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper right')
    plt.show()
    


def main():
    
    dataset = InputSampler()
    input_size = int(input("Specify the sample size : "))
    default_size = 20000
    
    if input_size <= 0 or None:
        dataset.create_sample(default_size)
    else:
        dataset.create_sample(input_size)
        
    model_input = input("Specificy the model you want to try. Please choose LSTM, AE-LSTM, MLP : ")
    
    if model_input == "LSTM":
        configs_path = os.path.join(
            os.path.dirname(__file__), "Experiments/LSTM", "Configs.json"
        )
        model = LSTM_Model()
        
    elif model_input == "AELSTM":
        configs_path = os.path.join(
            os.path.dirname(__file__), "Experiments/AE-LSTM", "Configs.json"
        )
        model = AE_LSTM_Model()
    
    elif model_input == "MLP":
        configs_path = os.path.join(
            os.path.dirname(__file__), "Experiments/MLP_AELSTM", "Configs.json"
        )
        model = MLP_AELSTM_Model()

    configs = json.load(open(configs_path, 'r'))
    if not os.path.exists(configs['model']['save_dir']): 
        os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data',configs['data']['filename']),
        configs['data']['train_test_split']
    )
   
    data_train, data_test = data.create_database(configs['data']['columns'],configs['data']['cols'], 1, configs['data']['sequence_length'])
    
    x_train, y_train = data.get_train_data()
    x_test, y_test = data.get_test_data()
    
    size_of_training = len(x_train)
    
    value = input('Do you want to create new model??? Respond with Yes/No. : ')
    
    if value == "Yes":
        model.build_model(configs, x_train)
    elif value == "No":
        print(configs['model']['save_dir'])
        
        model.load_model(os.path.join(configs['model']['save_dir'], '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y'), configs['training']['epochs'])))
    else:
        print('Try Again')
        
    accuracy, val_accuracy, loss, val_loss , time_elapsed = model.train_generator(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    
    train_predictions, test_prediction = model.predict_point_by_point(x_train, x_test)

    yTrainPredict, yTrain, yTest, yTestPredict = data.transform(y_train[:,0,0], train_predictions, x_test[:, 0, 0], y_test[:,0,0], test_prediction)

    trainScore = np.sqrt(mean_squared_error(yTrain, yTrainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    testScore = np.sqrt(mean_squared_error(yTest, yTestPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    plot_training_results(yTrain, yTrainPredict)
    plot_validation_results(yTest, yTestPredict)
    plot_accuracy(accuracy, val_accuracy)
    plot_loss(loss, val_loss)


if __name__ == '__main__':
    main()