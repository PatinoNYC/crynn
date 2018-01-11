import json
import csv

fileInput = "data\input.json"
fileOutput = "data\output.csv"

if sys.argv[1] is not None and sys.argv[2] is not None:
    fileInput = sys.argv[1]
    fileOutput = sys.argv[2]

f = open(fileInput)
data = json.load(f)
f.close()

f = csv.writer(open(fileOutput))

for x in data['Data']:
    f.writerow ( x.values() )

    
