import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))

inputfile = sys.argv[1]
outputfile = sys.argv[2]

lines = open(inputfile, "r").readlines()
output = open(outputfile, "w")

current_file = ""
for line in lines:
    line = line.rstrip()
    if line.endswith(".tiff"):
        current_file = line
        if current_file.startswith("./"):
            current_file = current_file[1:]
    else:
        split = line.split(" ")
        if len(split) == 4:
            x1 = float(split[0])
            y2 = float(split[1])
            x2 = x1+float(split[2])
            y1 = y2-float(split[3])
            classname = "text"

            string = current_path+current_file+","+str(int(x1))+","+str(int(y1))+","+str(int(x2))+","+str(int(y2))+","+classname
            output.write(string+"\n")
output.close()
