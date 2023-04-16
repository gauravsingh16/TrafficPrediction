import csv
import time
import random

class InputSampler():

  def __init__(self):
    self.networkdata = []
    self.nd_size = 0
    network_data = open("/Users/shrey_98/Desktop/TrafficPrediction/Traffic_Prediction/dataset1.csv", "r")
    myline = network_data.readline()
    while myline:
        header = myline.strip("\n").strip('"').split()
        self.networkdata.append(header)
        myline = network_data.readline()
    self.nd_size = len(self.networkdata)
    network_data.close()

  def create_sample(self, sample_size = 20000):
    '''
      Fetches sample_size continuous values from self.networkdata,
      Stores as csv file of the format 'dataNetwork_20000_Sun Apr 16 20:11:07 2023.csv'
    '''
    if sample_size > self.nd_size:
       return False
    csv_file = open('dataNetwork'+ str(sample_size) + time.ctime() + '.csv', 'w') 
    csv_writer = csv.writer(csv_file, delimiter='\t')
    csv_writer.writerow(["SrNo", "Timestamp", "Sender's IP", "Receiver's IP", "Protocol Stack", "Packets"])
    start_index = random.randint(0, self.nd_size - sample_size)
    sample_data = self.networkdata[start_index : start_index + sample_size + 1]
    for data in sample_data:
       csv_writer.writerow(data)
    csv_file.close()
    return True

if __name__ == '__main__':
    size = int(input("Specify the sample size : "))
    dataset = InputSampler()
    dataset.create_sample(size)
