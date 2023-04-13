import csv

class InputSampler():

  def create_sample(self):
    csv_file = open('dataNetwork.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["SrNo", "Timestamp", "Sender's IP", "Receiver's IP", "Protocol Stack", "Packets"])
    network_data = open("/home/gaurav/TrafficPrediction/Traffic_Prediction/dataset1.csv", "r")
    myline = network_data.readline()
    count = 0

    while myline:
        if myline == None:
          continue
        header = myline.strip("\n").strip('"').split()
        csv_writer.writerow(header)
        count+=1
        myline = network_data.readline()
        if count == 2000:
          print("check", network_data)
          break

    csv_file.close()
    network_data.close() 
    print(count)

  