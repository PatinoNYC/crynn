import time
import urllib
import json
import sys
from optparse import OptionParser

usage = "usage: %prog [options] arg1 arg2"
parser = OptionParser(usage=usage)
parser.add_option("-s", "--startval", default=1450936800, dest="startVal",
                  help="Unix Starting Timestamp")
parser.add_option("-c", "--cointype", default = "BTC", dest="coinType",
                  help="Cointypes: BTC, ETH, LTC, XRP")
#https://www.cryptocompare.com/api/data/coinlist/ for more CTs
(options, args) = parser.parse_args()

currentUnixTS = time.time()

while options.startVal < currentUnixTS:
    url = "https://min-api.cryptocompare.com/data/histohour?fsym=" + options.coinType + "&tsym=USD&limit=2000&aggregate=1&e=CCCAGG&extraParams=apnyc&toTs=" + str(options.startVal)
    
    response = urllib.urlopen(url)
    data = json.loads(response.read())

    fn = "data/cc_" + options.coinType + "_hourly_" + str(options.startVal) + ".json"
    with open(fn, 'w') as outfile:
        json.dump(data, outfile)

    options.startVal += 7196400
