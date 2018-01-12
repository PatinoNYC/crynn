import urllib
import json
import sys

startVal = 1450936800

for a in range (10):
    url = "https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD&limit=2000&aggregate=1&e=CCCAGG&extraParams=apnyc&toTs=" + str(startVal)
    response = urllib.urlopen(url)
    data = json.loads(response.read())

    fn = "data/cc_btc_hourly_" + str(a) + ".json"
    with open(fn, 'w') as outfile:
        json.dump(data, outfile)

    startVal += 7196400
