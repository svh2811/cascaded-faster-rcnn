import glob
import numpy as np
import cv2

all_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for name in glob.glob("/home/svh/cascaded-faster-rcnn/character-faster-rcnn/dicts_test/*.npy"):
	map = np.load(name)
	#imgName = name[name.rfind('/')+1 : -4]
	print("ImageName: " + imgName)
	img = cv2.imread("./DataGeneration/maps/" + imgName + ".tiff")
	H, W, C = img.shape
	img = None
	missed_chars = map.item().keys();
	for char in all_chars:
		char_mask = np.zeros((H, W), dtype=np.uint8)
		if char in missed_chars: 
			for bbox in map.item()[key]:
				# TODO: get those points
				cv2.fillPoly(char_mask, [np.int32([tl, tr, br, bl])], 255)
		np.save("./stage-01/" + imgName + "-" + key + ".npy", char_mask)
		print("File written: " + imgName + "-" + key)
