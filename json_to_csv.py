import time
import datetime
import json
import csv
import sys
import glob

currs = [ "BTC", "ETH", "LTC", "XRP" ]

#fileLocs = {}
currData = {}
for curr in currs:
    currData[curr] = {}

for curr in currs:
    print "Loading and dictionarizing data for " + curr
    files = glob.glob("data/cc_json/*" + curr + "*.json")

    for fileName in files:        
        f = open(fileName)
        json_data = json.load(f)
        f.close()

        data = json_data['Data']

        for dp in data:
            thisKey = dp["time"]
            currData[curr][thisKey] = {}
            currData[curr][thisKey]["volume"] = dp["volumeto"]
            currData[curr][thisKey]["closePx"] = dp["close"]

existingTSs = set() #not necessarily all TSs, we'll check that later

for curr in currs:
    for ts in currData[curr]:
        existingTSs.add(ts)

sortedExistingTSs = sorted( existingTSs )

prevVal = 0
count = 0
print "Checking for any gaps in the hourly data"
for ts in sortedExistingTSs:
    if ts - prevVal != 3600 and prevVal != 0:
         print str(ts) + " | " + str(prevVal) + " : " + str(ts - prevVal)
         count += 1
    prevVal = ts
if count == 0:
    print "None found!"
else:
    print "Some hourly data points are missing..."

with open('consolidated_data.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)



    prevRow = 0
    headerRow = ["Unix Date"]
    for curr in currs:
        headerRow.append( curr + " Px")
        headerRow.append( curr + " Volume")
        
    csvwriter.writerow( headerRow )
    for ts in sortedExistingTSs:
        thisRow = [ts]
        for curr in currs:
            if ( currData[curr][ts]["closePx"] > 0 ) or ( prevRow == 0 ):
                thisRow.append( currData[curr][ts]["closePx"] )
                thisRow.append( currData[curr][ts]["volume"] )
            else:
                print "Found 0 value for px data for " + curr
                thisRow.append( currData[curr][prevRow]["closePx"] )
                thisRow.append( currData[curr][prevRow]["volume"] )
                
        csvwriter.writerow( thisRow )
        prevrow = [ts]
    
