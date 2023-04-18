import os
import json
import numpy as np
import datetime as dt
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

def plot_train_data_time_taken(train_size, time_elapsed):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(train_size, label='Train size')
    plt.plot(time_elapsed, label='Time elapsed')
    plt.legend()
    plt.show()
    
    
def main():
        
    dataset = InputSampler()
    input_size = int(input("Specify the sample size : "))
    default_size = 20000
    
    if input_size <= 0 or None:
        dataset.create_sample(default_size)
    else:
        dataset.create_sample(input_size)

    configs = json.load(open('/home/gaurav/TrafficPrediction/Traffic_Prediction/Models/Configs.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): 
        os.makedirs(configs['model']['save_dir'])

    model = Model()
    data = DataLoader(
        os.path.join('data',configs['data']['filename']),
        configs['data']['train_test_split']
    )
   
    data_train, data_test, y_train, y_test = data.create_database(configs['data']['columns'],
        configs['data']['cols'])
    x_train, y_train = data.get_train_data(data_train, y_train)
    x_test, y_test = data.get_test_data(data_test, y_test)
    
    size_of_training = len(x_train)
    
    value = input('Do you want to create new model??? Respond with Yes/No. : ')
    
    if value == "Yes":
        
        model.build_model(configs, x_train)
    elif value == "No":
        
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

    trainScore = np.sqrt(mean_squared_error(y_train[:, 0, 0], train_predictions[:, 0, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(y_test[:, 0, 0], test_prediction[:, 0, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    yTrainPredict, yTrain, yTestPredict, yTest = data.transform(y_train[:, 0, 0], train_predictions, y_test, test_prediction)
    
    plot_training_results(yTrainPredict, yTrain)
    plot_validation_results(yTestPredict,yTest)
    plot_accuracy(accuracy, val_accuracy)
    plot_loss(loss, val_loss)
    #plot_train_data_time_taken(size_of_training, time_elapsed)

if __name__ == '__main__':
    main()