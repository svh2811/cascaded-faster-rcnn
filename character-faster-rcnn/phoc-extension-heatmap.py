import glob
import numpy as np
import cv2

for name in glob.glob("/home/supadhyay/supadhyay/dicts/D0090-5242001.npy"):
	map = np.load(name)
	imgName = name[name.rfind('/')+1 : -4]
	# print("ImageName: " + imgName)
	img = cv2.imread("./DataGeneration/maps/" + imgName + ".tiff")
	H, W, C = img.shape
	img = None
	for key in map.item().keys():
		if key != 'y':
			continue
		char_mask = np.zeros((H, W), dtype=np.uint8)
		for bbox in map.item()[key]:
			# TODO: get those points
			cv2.fillPoly(char_mask, [np.int32([tl, tr, br, bl])], 255)
		np.save("./phoc-extension-heatmap/" + imgName + "-" + key + ".npy", char_mask)
		print("File written: " + imgName + "-" + key)
