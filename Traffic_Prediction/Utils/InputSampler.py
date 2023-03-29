import json
import csv

with open("/home/gaurav/TrafficPrediction/Traffic_Prediction/dataset Network.json") as json_file:
  data = json.load(json_file)

network_data = data["Sheet1"]

csv_file = open('dataNetwork.csv', 'w')
csv_writer = csv.writer(csv_file)
count = 0

for data in network_data:
  if data == None:
    continue
  if count == 0:
    header = data.keys()
    csv_writer.writerow(header)
    count+=1
  csv_writer.writerow(data.values())

csv_file.close()