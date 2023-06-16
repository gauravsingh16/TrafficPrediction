import csv
import random
import datetime
from pandas import read_csv
import matplotlib.pyplot as plt


class InputSampler():
  
  def __init__(self):
    self.database = '/home/gaurav/TrafficPrediction/Traffic_Prediction/dataset1.csv'

  def create_sample(self, dataset_size):
    '''
      Fetches sample_size continuous values from self.networkdata,
      Stores as csv file of the format 'dataNetwork_20000_Sun Apr 16 20:11:07 2023.csv'
    '''
    data = []
    sample_data = []
    nd_size = 0
    csv_file = open('dataNetwork.csv', 'w') 
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["SrNo", "Timestamp", "Sender's IP", "Receiver's IP", "Protocol Stack", "Packets"])
    
    network_data = open(self.database, "r")
    myline = network_data.readline()
    while myline:
        header = myline.strip("\n").strip('"').split()
        data.append(header)                                                                                                                                        
        myline = network_data.readline()
    
    self.nd_size = len(data) 
    start_index = random.randint(0, self.nd_size - dataset_size)
    
    sample_data = data[start_index : start_index + dataset_size + 1]

    for i_data in sample_data:
      csv_writer.writerow(i_data)
    csv_file.close()
    network_data.close()
    
    self.update_dataset()
    return True
  
  def parse(self, x):
    return datetime.strptime(x, '%Timestamp')
  
  def update_dataset(self):
    randLink = []
    dataset = read_csv('dataNetwork.csv')
    dataset.drop('SrNo', axis=1, inplace=True)
    
    # manually specify column names
    dataset.columns = ["Timestamp","Sender's IP", "Receiver's IP", 'Protocol Stack', 'Packets']
    # mark all NA values with 0
    # summarize first 5 rows
    
    connectivity_type = [0, 1]
    for i in range(len(dataset)):
      randValue = random.choice(connectivity_type)
      randLink.append(randValue)
          
    dataset['Connectivity'] = randLink
    print((dataset))
    # save to file
    dataset.to_csv('dataNetwork.csv')
    