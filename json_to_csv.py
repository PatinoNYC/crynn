import time
import datetime
import json
import csv
import sys

fileInput = "data\input.json"
fileOutput = "data\output.csv"

if sys.argv[1] is not None and sys.argv[2] is not None:
    fileInput = sys.argv[1]
    fileOutput = sys.argv[2]

f = open(fileInput)
data = json.load(f)
f.close()

f = csv.writer(open(fileOutput, "wb+"))

for x in data['Data']:
    tv = x.get('time')
    print ( datetime.datetime.fromtimestamp( tv ).strftime('%Y-%m-%d %H:%M:%S'))
    f.writerow ( x.values() )

    
