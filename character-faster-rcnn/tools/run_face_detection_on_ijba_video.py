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
				  'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_70000.caffemodel')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    plt.pause(0.1)

def det_on_video(video_dir):
	video_dir = os.path.join(video_dir, 'all_frames')
	image_names = []
	for im_name in os.listdir(video_dir):
		if os.path.isfile(os.path.join(video_dir, im_name)) and im_name.endswith('.png'):
			image_names.append(im_name)

	print 'There are %d images in %s.'% (len(image_names), video_dir)

	CONF_THRESH = 0.85
	NMS_THRESH = 0.15
	res = np.zeros(len(image_names, ), dtype=np.object)
	for idx, im_name in enumerate(image_names):
		# print im_name
		im = cv2.imread(os.path.join(video_dir, im_name))

		# # Detect all object classes and regress object bounds
		# timer = Timer()
		# timer.tic()
		scores, boxes = im_detect(net, im)
		# timer.toc()
		# print ('Detection took {:.3f}s for '
		#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

		cls_ind = 1
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
					  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		keep = np.where(dets[:, 4] > CONF_THRESH)
		dets = dets[keep]

		# vis_detections(im, 'face', dets, thresh=CONF_THRESH)

		dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
		dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

		res[idx] = dets

		if (idx + 1) % 20 == 0:
			sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names)))
			sys.stdout.flush()

	print ''
	tmp_image_names = image_names
	image_names = np.zeros((len(image_names), ), dtype=np.object)
	for idx, im_name in enumerate(tmp_image_names):
		image_names[idx] = im_name

	return image_names, res

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
	parser.add_argument('--video', dest='video_name', help='video name',
						default='')

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals

	args = parse_args()

	prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
							'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
	caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
							  NETS[args.demo_net][1])

	prototxt = 'models/face/VGG16/faster_rcnn_end2end/test.prototxt'
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

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	# data_dir = '/data/hzjiang/Data/face/Data/ijba_frame';
	# dst_dir = '/data/hzjiang/Data/face/WorkingData/ijba_frame'

	data_dir = '/data2/hzjiang/Data/CS3';
	dst_dir = '/data2/hzjiang/WorkingData/CS3'

	# process all videos
	if args.video_name == '':
		for video_name in os.listdir(data_dir):
			if os.path.isdir(os.path.join(data_dir, video_name)) and (video_name != '.') and (video_name != '..'):
				if not os.path.exists(os.path.join(dst_dir, video_name + '.mat')):
					image_names, boxes = det_on_video(os.path.join(data_dir, video_name))
					sio.savemat(os.path.join(dst_dir, video_name + '.mat'), {'fnames': image_names, 'boxes': boxes})
					print '%s done.' % video_name
	else:
		image_names, boxes = det_on_video(os.path.join(data_dir, args.video_name))
		sio.savemat(os.path.join(dst_dir, args.video_name + '.mat'), {'fnames': image_names, 'boxes': boxes})