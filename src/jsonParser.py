import sys

sys.path.append('urllib1')
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import json
import csv

def parse(alt, mach, zxn):
    url = 'http://192.168.1.32:3030/v1/api/perf/'
    input_obj = {'alt': alt, 
                 'mach': mach,
                 'zxn': zxn}
    postdata = urlencode(input_obj).encode()

    httprequest = Request(url, data=postdata, method='POST')

    with urlopen(httprequest) as response:
        jsonbody = json.load(response)
        json_ = jsonbody[0]
    # print(json_)
    return json_

out = parse(0,0,1)
data_file = open('output.csv', 'w')
csv_writer = csv.writer(data_file)
header = out.keys()
csv_writer.writerow(header)
csv_writer.writerow(out.values())
data_file.close()