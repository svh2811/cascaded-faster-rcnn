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

NETS = {'vgg16': ('VGG16',
				  'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_map_iter_10000.caffemodel')}

def get_imdb_map(data_dir):
	imdb = []
	file_name = 'test.txt'
	file_name = os.path.join(data_dir, file_name)

	image_names = []
	with open(file_name) as f:
		# print len(f.lines())
		lines = f.readlines()

		idx = 0
		while idx < len(lines):
			image_name = lines[idx].split('\n')[0]
			image_names.append(image_name)
			# print image_name
			image_ext = os.path.splitext(image_name)[1].lower()
			#print image_ext
			assert(image_ext == '.png' or image_ext == '.jpg' or image_ext == '.jpeg' or image_ext == '.tiff')

			idx += 1
			num_boxes = int(lines[idx])
			# print num_boxes

			idx += num_boxes + 1

		assert(idx == len(lines))

	imdb.append(image_names)

	return imdb

def vis_detections(im, title, dets, thresh, filename):
	outdir = 'output/vis_ims/'
	# im = im[:, :, (2, 1, 0)]
	for i in xrange(dets.shape[0]):
		bbox = dets[i, :4]
		score = dets[i, -1]
		if score > thresh:
			cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

	
	#print title	
	#cv2.imshow(title, im)
	#cv2.waitKey(1000)
	cv2.imwrite(outdir+filename, im)

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

	data_dir = '/media/ray/maps_project/faster-rcnn/DataGeneration'
	work_dir = '/media/ray/maps_project/faster-rcnn/detections'
	force_new = False

	CONF_THRESH = 0.65
	NMS_THRESH = 0.15

	imdb = get_imdb_map(data_dir)

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	nfold = len(imdb)
	for i in xrange(nfold):
		image_names = imdb[i]

		# detection file
		dets_file_name = 'map_res/words-det-fold-%02d.txt' % (i + 1)
		fid = open(dets_file_name, 'w')
		sys.stdout.write('%s ' % (i + 1))

		for idx, im_name in enumerate(image_names):
			# timer = Timer()
			# timer.tic()

			# Load the demo image
			folder,sep,filenom = im_name.partition('/')
			
			mat_name = im_name[:-4] + '.mat'

			# print os.path.join(work_dir, mat_name)

			if (not force_new) and os.path.exists(os.path.join(work_dir, mat_name)):
				res = sio.loadmat(os.path.join(work_dir, mat_name))
				boxes = res['boxes']
				scores = res['scores']
			else:
				# im_path = im_name + '.jpg'
				im = cv2.imread(os.path.join(data_dir, im_name))

				# # Detect all object classes and regress object bounds
				# timer = Timer()
				# timer.tic()
				scores, boxes = im_detect(net, im)
				# timer.toc()
				# print ('Detection took {:.3f}s for '
				#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

				dir_name, mat_name = os.path.split(im_name)
				if not os.path.exists(os.path.join(work_dir, dir_name)):
					os.makedirs(os.path.join(work_dir, dir_name))

				res = {'boxes': boxes, 'scores': scores}
				sio.savemat(os.path.join(work_dir, mat_name), res)

			cls_ind = 1
			cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
			cls_scores = scores[:, cls_ind]
			dets = np.hstack((cls_boxes,
							  cls_scores[:, np.newaxis])).astype(np.float32)
			keep = nms(dets, NMS_THRESH)
			dets = dets[keep, :]
			
			keep = np.where(dets[:, 4] > CONF_THRESH)
			dets = dets[keep]
			vis_detections(im, 'words', dets, CONF_THRESH, filenom)

			dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
			dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

			# timer.toc()
			# print ('Detection took {:.3f}s for '
			#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

			fid.write(im_name + '\n')
			fid.write(str(dets.shape[0]) + '\n')
			for j in xrange(dets.shape[0]):
				fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


			if ((idx + 1) % 10) == 0:
				sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names) * 100))
				sys.stdout.flush()

		print ''
		fid.close()

			
