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
test_dir = './vis_images/'
out_dir = './outputs/'
plot_dir = 'ROC/'

#IoU_thresh = 0.7

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
for filename in glob.glob(test_dir+'*.npy'):
	_dir, _, _file = filename.rpartition('/')
	_image, _, _ext = _file.rpartition('.')
	_annots, _, _im_ext = _image.rpartition('.')
	
	files.append(_image)
	# print image name
	#print _image
	
	# grabbing the image
	Img = Image.open(maps_dir+_image)
	draw = ImageDraw.Draw(Img)

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
		for j in range(len(poly)):
			draw.line((poly[j][0], poly[j][1], poly[(j+1)%len(poly)][0], poly[(j+1)%len(poly)][1]), width = 3, fill="red")
			# break

	global_IoU_vals.append(IoU_vals)
	trains_all.append(train_polys)
	test_all.append(test_polys)
	#break

print "done computing the IoUs"

#'''
for IoU_thresh in np.arange(0, 1+step, step):
	print IoU_thresh
	stats = []
	for i in range(len(global_IoU_vals)):
		IoU_vals = np.copy(global_IoU_vals[i])
		_id = IoU_vals < IoU_thresh
		IoU_vals[_id] = 0
		_row, _col = lsa(1-IoU_vals)
		
		TP = (IoU_vals[_row, _col] > 0).sum()
		FP = (IoU_vals[_row, _col] == 0).sum()
		# can't have true negative
		FN = len(trains_all[i]) - TP
		
		stats.append([TP, FP, FN])

	global_stats.append(stats)
	
	stats = np.asarray(stats)
	avg_TP = float(stats[:,0].sum()) / float(stats.shape[0])
	avg_FP = float(stats[:,1].sum()) / float(stats.shape[0])
	avg_FN = float(stats[:,2].sum()) / float(stats.shape[0])
	
	precision = float(avg_TP) / float(avg_TP + avg_FP)
	recall = float(avg_TP) / float(avg_TP + avg_FN)

	P.append(precision)
	R.append(recall)
#'''

# alternate IoU cals
'''
matched_IoUs = []
for i in range(len(global_IoU_vals)):
	IoU_vals = np.copy(global_IoU_vals[i])
	_row, _col = lsa(1-IoU_vals)
	matched_IoUs.append(IoU_vals[_row, _col])
print "done matching"

for IoU_thresh in np.arange(0, 1+step, step):
	print IoU_thresh
	stats = []
	for i in range(len(matched_IoUs)):
		TP = (matched_IoUs[i] > IoU_thresh).sum()
		FP = (matched_IoUs[i] <= IoU_thresh).sum()
		FN = len(trains_all[i])

		stats.append([TP, FP, FN])

	global_stats.append(stats)
	
	stats = np.asarray(stats)
	avg_TP = float(stats[:,0].sum()) / float(stats.shape[0])
	avg_FP = float(stats[:,1].sum()) / float(stats.shape[0])
	avg_FN = float(stats[:,2].sum()) / float(stats.shape[0])
	
	precision = float(avg_TP) / float(avg_TP + avg_FP)
	recall = float(avg_TP) / float(avg_TP + avg_FN)

	P.append(precision)
	R.append(recall)
'''

#print P, R

plt.plot(P, R)
plt.xlim([0,1])
plt.ylim([0,1])
IoU_thresh = 0

for i, j in zip(P, R):
	plt.annotate(str(IoU_thresh), xy = (i,j))
	IoU_thresh = IoU_thresh+step

plt.savefig(out_dir+'PR-report.png')

print "done average PR-report"

Q = np.copy(global_stats)
for i in range(len(files)):
	melt= np.copy(Q[:,i])
	melt = melt.astype(float)
	precision_melt = melt[:, 0] / (melt[:,0] + melt[:,1])
	recall_melt = melt[:, 0] / (melt[:,0] + melt[:,2])
	
	plt.close()
	plt.plot(precision_melt, recall_melt)
	plt.xlim([0,1])
	plt.ylim([0,1])
	IoU_thresh = 0
	
	for x, y in zip(precision_melt, recall_melt):
		plt.annotate(str(IoU_thresh), xy = (x,y))
		IoU_thresh = IoU_thresh+step
	name,_,_ = files[i].rpartition('.')
	plt.savefig(out_dir+'PR-'+name+'.png')

print "done individual PR-report"