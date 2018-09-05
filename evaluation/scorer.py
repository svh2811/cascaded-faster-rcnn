import numpy as np
import glob
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os
import cv2
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from scipy.optimize import linear_sum_assignment as lsa
import matplotlib.pyplot as plt

train_dir = '../annotations/current/'
maps_dir = '../maps/'
test_dir = './results_CNF_NMS_NPYs/'
out_dir = './outputs/'
plot_dir = 'ROC/'

IoU_thresh = 0.5

if os.path.isdir(out_dir):
	pass
else:
	os.mkdir(out_dir)

if os.path.isdir(out_dir+plot_dir):
	pass
else:
	os.mkdir(out_dir+plot_dir)

trains_all = []
test_all = []
P = []
R = []
step = 0.1 
global_stats = []
global_IoU_vals = []
files = []

# compute the IoUs
for dir_name in glob.glob(test_dir+'*'):
	print dir_name
	stats = []
	for filename in glob.glob(dir_name+'/'+'*.npy'):
		print filename
		_dir, _, _file = filename.rpartition('/')
		_image, _, _ext = _file.rpartition('.')
		_annots, _, _im_ext = _image.rpartition('.')
	
		files.append(_image)
		# print _image
	
		# grab the train and test annotations
		trains = np.load(train_dir+_annots+_+_ext).item()
		tests = np.load(filename)
		#print filename, maps_dir+_image, train_dir+_annots+_+_ext
	
		# data structure for saving the IoU values
		IoU_vals = np.zeros((len(tests), len(trains.keys())))
	
		# save the train anots
		train_polys = []
		for i in trains.keys():
			#print trains[i]['vertices']
			train_polys.append(Polygon(trains[i]['vertices']))
			pass

		s = STRtree(train_polys)
	
		# save the test annots
		test_polys = []
		for i in range(len(tests)):
			poly = tests[i]
			poly = poly.tolist()
			#poly.append(poly[0])
			#print Polygon(poly).area
			test_polys.append(Polygon(poly))
			results = s.query(test_polys[i])
			for j in range(len(results)):
				_id = train_polys.index(results[j])
				_intersection = train_polys[_id].intersection(test_polys[i]).area
				_union = train_polys[_id].union(test_polys[i]).area
				IoU_vals[i, _id] = _intersection / _union
		
		# do the linear sum assignment
		_row, _col = lsa(1-IoU_vals)
		assignment_matrix = IoU_vals[_row, _col]
		
		# compute the numbers
		TP = (assignment_matrix >= IoU_thresh).sum()
		FP = (assignment_matrix < IoU_thresh).sum()
		FN = len(trains.keys()) - TP

		stats.append([TP, FP, FN])

	global_stats.append(stats)
	global_IoU_vals.append(assignment_matrix)
	# assemble stats
	stats = np.asarray(stats)
	print stats.shape
	avg_TP = float(stats[:,0].sum()) / float(stats.shape[0])
	avg_FP = float(stats[:,1].sum()) / float(stats.shape[0])
	avg_FN = float(stats[:,2].sum()) / float(stats.shape[0])

	# compute P, R
	precision = float(avg_TP) / float(avg_TP + avg_FP)
	recall = float(avg_TP) / float(avg_TP + avg_FN)

	P.append(precision)
	R.append(recall)

print 'precision: ', P
print 'recall: ', R


# save as a plot
plt.close()
_id = np.argsort(P)
plt.figure(figsize=[24,18])
plt.plot(np.asarray(P)[_id], np.asarray(R)[_id])
plt.xlabel('precision')
plt.ylabel('recall')
plt.title('PR-curve for IoU_thresh: 0.5 and nms_thresh: 0.1')
#plt.xlim([0,1])
#plt.ylim([0,1])
conf_thresh = 0.60
step = .02

for i, j in zip(P, R):
	plt.annotate(str(conf_thresh), xy = (i,j))
	conf_thresh = conf_thresh+step

plt.savefig(out_dir+'PR-report.png')

print "done average PR-report"
