from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys
import pickle
import math
import config

sys.path.append("../evaluation/")
from util import rotate_image, adjust_image_size

NETS = {'vgg16': ('VGG16','/mnt/nfs/work1/elm/ray/trained_models/vgg16_faster_rcnn_map_iter_'+config.CHARACTER+'_16159.caffemodel')}

#NETS = {'vgg16': ('VGG16','/media/ray/vgg16_faster_rcnn_map_iter_56159.caffemodel')}
#NETS = {'vgg16': ('VGG16','../models/vgg16_faster_rcnn_map_iter_m_16159.caffemodel')}
global undetected
global reduce_threshold
undetected = np.load(config.UNDETECTED)
reduce_threshold = config.REDUCE_THRESHOLD

def get_imdb_map(data_dir):
	imdb = []
	#image_names = ['images/D0042-1070010.tiff', 'images/D0042-1070005.tiff']
	#image_names = ['letmap00001.jpg', 'letmap00045.jpg', 'letmap00069.jpg', 'letmap00134.jpg']

	image_names = ['D0090-5242001.tiff']
	#image_names = ['D0041-5370006.tiff']
	#image_names = ['D0079-0019007.tiff']
	#image_names = ['D0117-5755036.tiff']
	#image_names = ['D5005-5028100.tiff']
	#image_names = ['D0006-0285025_0_36.tiff']
	'''
	image_names = []
	test_file_lines = open("./DataGeneration/"+"test.txt", "r").readlines()
	for line in test_file_lines:
        	if line.endswith('.tiff\n'):
			h,s,t = line.partition('/')
		        h,s,t = t.partition('.')
                	h,s,t = h.partition('_')
                	image_names.append(h+'.tiff')
	#print image_names
	'''

	imdb.append(image_names)
	print("Number of images: ", len(imdb))
	return imdb

def vis_detections(im, title, dets, thresh):
	# im = im[:, :, (2, 1, 0)]
	for i in xrange(dets.shape[0]):
		bbox = dets[i, :4]
		score = dets[i, -1]
		if score > thresh:
			cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


	cv2.imshow(title, im)
	cv2.waitKey(0)

def save_detections(im, im_name, dets, thresh):
	global undetected
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in xrange(dets.shape[0]):
		bbox = (dets[i, :4])
		score = dets[i, -1]
		if score > thresh:
			cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(2*bbox[1]-bbox[3])), (0, 255, 0), 4)
			#cv2.putText(im, str(score),(int(bbox[0]),int(bbox[1])), font, 0.6,(0,0,255),2,cv2.LINE_AA)
	cv2.imwrite(im_name, im)

'''
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))
    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
'''

def im_detect_sliding_crop(net, im, crop_h, crop_w, step):
	imh, imw, _ = im.shape
	global undetected
	global reduce_threshold
	cls_ind = 1

	boxes = np.zeros((0, 4), dtype=np.float32)
	scores = np.zeros((0, 1), dtype=np.float32)
	boxes2 = np.zeros((0, 4), dtype=np.float32)
	scores2 = np.zeros((0, 1), dtype=np.float32)
	
	y1 = 0
	while y1 < imh:
		y2 = min(y1 + crop_h, imh)
		if y2 - y1 < 25:
			y1 += step
			continue

		x1 = 0
		while x1 < imw:
			x2 = min(x1 + crop_w, imw)
			if x2 - x1 < 25:
				x1 += step
				continue

			crop_im = im[y1:y2, x1:x2, :]

			crop_scores, crop_boxes = im_detect(net, crop_im)
			crop_boxes = crop_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
			crop_scores = crop_scores[:, cls_ind] + (0.01 * np.random.random() - 0.005)

			crop_boxes[:,0] += x1	# change for flipped results
			crop_boxes[:,1] += y1	# change for flipped results
			crop_boxes[:,2] += x1	# change for flipped results
			#crop_boxes[:,2] = x1 - crop_boxes[:,2]
			crop_boxes[:,3] += y1	# change for flipped results
			#crop_boxes[:,3] = crop_boxes[:,1]
			#crop_boxes[:,3] = crop_boxes[:,1] - (crop_boxes[:,3] - crop_boxes[:,1])
			if config.DEBUG:
				print('this_crop',y1,y2+crop_h,imh,x1,x1+crop_w,imw)
			if reduce_threshold:
				for points in undetected.item()[config.DICT_KEY]:
					if (x1<=points[0][0] and x2>=points[0][0] and y1<=points[0][1] and y2>=points[0][1]) or (x1<=points[1][0] and x2>=points[1][0] and y1<=points[1][1] and y2>=points[1][1]):
						if config.DEBUG:
							print('FOUND CROP',y1,y2,x1,x2,points)
						boxes2 = np.vstack((boxes2,crop_boxes))
						scores2 = np.vstack((scores2,crop_scores[:,np.newaxis]))
					else:
						boxes = np.vstack((boxes, crop_boxes))
						scores = np.vstack((scores, crop_scores[:, np.newaxis]))
			else:
				boxes = np.vstack((boxes, crop_boxes))
				scores = np.vstack((scores, crop_scores[:, np.newaxis]))
			x1 += step

		y1 += step
	if reduce_threshold:
		return scores, boxes, scores2, boxes2
	return scores, boxes

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						choices=NETS.keys(), default='vgg16')

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	global reduce_threshold
	global undetected
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	# cfg.TEST.BBOX_REG = False

	args = parse_args()

	prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
	caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
							  NETS[args.demo_net][1])

	prototxt = 'models/map/VGG16/faster_rcnn_end2end/test.prototxt'
	caffemodel = NETS[args.demo_net][1]

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/script/'
					   'fetch_faster_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print '\n\nLoaded network {:s}'.format(caffemodel)

	#data_dir = '/media/ray/maps_project/mapOutput_3/'
	#data_dir = '/media/ray/maps_project/faster-rcnn/'
	#work_dir = '/media/ray/maps_project/faster-rcnn/detections/'
	data_dir = './DataGeneration/maps_red/'
	#data_dir = './'
	work_dir = './detections/'
	force_new = False

	CONF_THRESH = 0.9
	NMS_THRESH = 0.1

	'''
	crop_w = 300
	crop_h = 300
	step = 100
	'''

	crop_w = 500
	crop_h = 500
	step = 400
	#'''
	imdb = get_imdb_map(data_dir)

	# Warmup on a dummy image
	im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	nfold = len(imdb)
	for i in xrange(nfold):
		print("\n\n")
		print("#nfold: ", i)
		image_names = imdb[i]

		# detection file
		dets_file_name = 'map_res/words-det-D0090-5242001-%02d.txt' % (i + 1)
		dets_file_name2 = 'map_res/words-det'+config.FID_EXTENSION+'-D0090-5242001-%02d.txt' % (i + 1)
		fid = open(dets_file_name, 'w')
		fid2 = open(dets_file_name2, 'w')
		sys.stdout.write('%s ' % (i + 1))

		for idx, im_name in enumerate(image_names):
			# timer = Timer()
			# timer.tic()
			print("\n\n")
			print("idx: ", idx, "    im_name: ", im_name)

			# Load the demo image
			mat_name = im_name[:-4] + '.mat'

			rot_box_filename = 'map_res/rot_box_'+im_name.split("/")[-1]+'_'+str(i+1)+'.pkl'
			rot_box_filename2 = 'map_res/rot_box2_'+im_name.split("/")[-1]+'_'+str(i+1)+'.pkl'
			print("rot_box_filename: ", rot_box_filename)
			rot_file = open(rot_box_filename,"wb")
			if reduce_threshold:
				rot_file2 = open(rot_box_filename2, "wb")
			# print os.path.join(work_dir, mat_name)

			print("Reading Image: ", os.path.join(data_dir, im_name))

			im = cv2.imread(os.path.join(data_dir, im_name))
			#print os.path.join(data_dir, im_name)
			#im, translation = adjust_image_size(im, padding_amount=500)
			translation = (0, 0)
			#print(im.shape)

			all_boxes, all_scores, all_rotations = [], [], []
			all_boxes2, all_scores2, all_rotations2 = [], [], []
			for angle in range(0, 95, 10):
				print("--------------------------------------------------")
				#print("Running detection at angle: " + str(angle))
				image_center = tuple(np.array((im.shape[0],im.shape[1]))/2)
				R = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
				#rot_img = cv2.warpAffine(im, R, (im.shape[0], im.shape[1]), flags=cv2.INTER_LINEAR)
				rot_img, rot_mat, bounds = rotate_image(im, angle, im.shape)

				# # Detect all object classes and regress object bounds
				timer = Timer()
				timer.tic()
				# scores, boxes = im_detect(net, im)
				if reduce_threshold:
					scores, boxes, scores2, boxes2 = im_detect_sliding_crop(net, rot_img, crop_h, crop_w, step)
				else:
					scores, boxes = im_detect_sliding_crop(net, rot_img, crop_h, crop_w, step)

				print("Max-Score: ", np.max(scores))
				print("Boxes for angle " + str(angle) + ": " + str(boxes.shape[0]))
				Rinv = cv2.getRotationMatrix2D(image_center, -angle, scale=1.0)
				if reduce_threshold:
					print("Max-Score2: ", np.max(scores2))
					print("Boxes2 for angle " + str(angle) + ": " + str(boxes2.shape[0]))
					Rinv = cv2.getRotationMatrix2D(image_center, -angle, scale=1.0)

				if reduce_threshold:
					all_boxes2.append(boxes2)
					all_scores2.append(scores2)
					all_rotations2.append( {'angle':angle, 'center':image_center, 'R':R, 'Rinv': Rinv} )					

				all_boxes.append(boxes)
				all_scores.append(scores)
				all_rotations.append( {'angle':angle, 'center':image_center, 'R':R, 'Rinv': Rinv} )

				timer.toc()
				print ('Detection took {:.3f}s for ''{:d} object proposals').format(timer.total_time, boxes.shape[0])

			pickle.dump([all_boxes, all_scores, all_rotations], rot_file)
			if reduce_threshold:
				pickle.dump([all_boxes2, all_scores2, all_rotations2], rot_file2)

			dir_name, mat_name = os.path.split(im_name)
			if not os.path.exists(os.path.join(work_dir, dir_name)):
				print("Creting directory: ", os.path.join(work_dir, dir_name))
				os.makedirs(os.path.join(work_dir, dir_name))

			print("im_name: ", im_name)
			if reduce_threshold:
				fid2.write(im_name + "\n")	
			else:
				fid.write(im_name + "\n")
			#import pdb; pdb.set_trace()
			if reduce_threshold:
				for k in range(len(all_boxes2)):
					boxes2 = all_boxes2[k]
					scores = all_scores2[k]
					rot = all_rotations2[k]
					angle = rot['angle']

					fid2.write("angle " + str(angle) + "\n")
					#print("Boxes: " + str(len(boxes)))
					res = {'boxes': boxes2, 'scores': scores2}
					sio.savemat(os.path.join(work_dir, mat_name), res)

					dets = np.hstack((boxes2,scores2)).astype(np.float32)
					# save dets for checks
					np.save('dets2_before_NMS.npy', dets)
					if config.DEBUG:
						print('boxes',boxes2.shape,np.min(boxes2[:,0]),np.max(boxes2[:,0]),np.min(boxes2[:,2]),np.max(boxes2[:,2]))
						print('scores',scores2.shape,np.min(scores2),np.max(scores2))
					# if reduce_threshold:
					# 	for points in undetected['e']:
					# 		if boxes[]
					NMS_THRESH = config.NMS_THRESH
					keep = nms(dets, NMS_THRESH, force_cpu=False)
					#import pdb; pdb.set_trace()

					dets = dets[keep, :]
					# save dets for checks
					CONF_THRESH = config.CONF_THRESH
					np.save('dets2_after_NMS.npy', dets)
					keep = np.where(dets[:, 4] > CONF_THRESH)
					dets = dets[keep]
					# save dets for checks
					np.save('dets2_after_CONF.npy', dets)
					#print("Detections: " + str(dets.shape))
					#save_detections(im, im_name, dets, CONF_THRESH)

					dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
					#dets[:, 2] = -dets[:, 2] + dets[:, 0] - 1
					dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
					#dets[:, 3] = -dets[:, 3] + dets[:, 1] - 1

					# timer.toc()
					# print ('Detection took {:.3f}s for '
					#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

					fid2.write(str(dets.shape[0]) + '\n')
					for j in xrange(dets.shape[0]):
						fid2.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


				if ((idx + 1) % 10) == 0:
					sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names) * 100))
					sys.stdout.flush()
			else:
				for k in range(len(all_boxes)):
					boxes = all_boxes[k]
					scores = all_scores[k]
					rot = all_rotations[k]
					angle = rot['angle']

					fid.write("angle " + str(angle) + "\n")
					#print("Boxes: " + str(len(boxes)))
					res = {'boxes': boxes, 'scores': scores}
					sio.savemat(os.path.join(work_dir, mat_name), res)

					dets = np.hstack((boxes,scores)).astype(np.float32)
					# save dets for checks
					np.save('dets_before_NMS.npy', dets)
					if config.DEBUG:
						print('boxes',boxes.shape,np.min(boxes[:,0]),np.max(boxes[:,0]),np.min(boxes[:,2]),np.max(boxes[:,2]))
					# if reduce_threshold:
					# 	for points in undetected['e']:
					# 		if boxes[]
					keep = nms(dets, NMS_THRESH, force_cpu=False)
					#import pdb; pdb.set_trace()

					dets = dets[keep, :]
					# save dets for checks
					np.save('dets_after_NMS.npy', dets)
					keep = np.where(dets[:, 4] > CONF_THRESH)
					dets = dets[keep]
					# save dets for checks
					np.save('dets_after_CONF.npy', dets)
					#print("Detections: " + str(dets.shape))
					#save_detections(im, im_name, dets, CONF_THRESH)

					dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
					#dets[:, 2] = -dets[:, 2] + dets[:, 0] - 1
					dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
					#dets[:, 3] = -dets[:, 3] + dets[:, 1] - 1

					# timer.toc()
					# print ('Detection took {:.3f}s for '
					#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

					fid.write(str(dets.shape[0]) + '\n')
					for j in xrange(dets.shape[0]):
						fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


				if ((idx + 1) % 10) == 0:
					sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names) * 100))
					sys.stdout.flush()

		print 'Done!!!!'
		fid.close()

