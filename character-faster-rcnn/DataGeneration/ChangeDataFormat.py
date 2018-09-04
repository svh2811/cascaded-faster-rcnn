import sys

filename = sys.argv[1]
file = open(filename, "r")

lines = file.readlines()

output = open("clean_test.txt", "w")

prev_name = ""

for i, line in enumerate(lines):
    if line.endswith(".tiff\n"):
	dirname = line.split("/")[0]
        imgname = line.split("/")[1]
        split_line = imgname.split("_")
        angle = float(split_line[-1][:-5])
        imgname = dirname + "/" + split_line[0] + ".tiff"

        if imgname != prev_name:
            output.write(imgname + "\n")

        prev_name = imgname
        output.write("angle " + str(angle) + "\n")
    else:
        output.write(line)

output.close()
