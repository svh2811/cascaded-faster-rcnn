import sys

image_file = sys.argv[1]
lines = open(image_file, "r").readlines()

files = []
for line in lines:
    if line.endswith(".tiff\n"):
        splitline = line.split("/")
        if len(splitline) > 1:
	    splitline = splitline[1].split("_")[0]
	    if splitline not in files:
	    	files.append( splitline )

print(files)
        
