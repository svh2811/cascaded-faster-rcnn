import sys
import glob
import numpy as np

target_dir = sys.argv[1]
source_1 = sys.argv[2]
source_2 = sys.argv[3]
source_3 = sys.argv[4]
source_4 = sys.argv[5]

train = open(target_dir+"train.txt", "w")
sources = [ source_1, source_2, source_3, source_4]
#sources = [ source_1, source_2, source_3]

for source in sources:
    lines = open(source, "r").readlines()
    for line in lines:
        train.write(line)

train.close()
'''
def create_trains(train_files):
	train = open(target_dir+"train.txt", "w")
	for i in range(len(train_files)):
		V = []
		filenpy = './annotations/current/'+train_files[i]+'.npy'
		fileimg = './img/'+train_files[i]+'.tiff'
		train.write(fileimg+'\n')
		data = np.load(filenpy).item()
		for key in data.keys():
			vertices = data[key]['vertices']
			V.append(vertices)
		print V
			
	train.close()
	

test_files = []
test_file_lines = open(target_dir+"test.txt", "r").readlines()
for line in test_file_lines:
	if line.endswith('.tiff\n'):
		h,s,t = line.partition('/')
		h,s,t = t.partition('.')
		h,s,t = h.partition('_')
		test_files.append(h)

train_files = []
for filename in glob.glob('./annotations/current/*'):
	h1,s1,t1 = filename.partition('current/')
	h,s,t = t1.partition('.')
	if h in test_files:
		pass
	else:
		train_files.append(h)

create_trains(train_files)
'''
