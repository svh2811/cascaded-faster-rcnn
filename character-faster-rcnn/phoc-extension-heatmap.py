import glob
import numpy as np

for name in glob.glob("./*.npy"):
	print("File: ", name)
	map = np.load(name)