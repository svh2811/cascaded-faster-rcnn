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
			tl, br = bbox[0], bbox[1]
			char_mask[tl[1]:br[1]+1, tl[0]:br[0]+1] = 255
		np.save("./phoc-extension-heatmap/" + imgName + "-" + key + ".npy", char_mask)
		print("File written: " + imgName + "-" + key)
