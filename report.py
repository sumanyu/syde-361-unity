import os
import time
import json



def openJson():
    filename = 'output_' + time.strftime("%Y-%m-%d")
    if not os.path.isfile(filename):
        print filename + "file not found"
    
    with open(filename,'r') as content_file:
        for line in content_file:
            print json.loads(line)


        # content = content_file.read()

    # print content



openJson()


        
