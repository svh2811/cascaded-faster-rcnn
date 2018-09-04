import numpy as np

scores = np.zeros((0, 2), dtype=float)
for split in xrange(1, 11):
	fname = '/home/hzjiang/Code/FDDB/evaluation/ijba_split%dDiscROC.txt' % split
	stat = np.loadtxt(fname)
	idx1 = np.argmin(abs(stat[:, 1] - 0.1))
	idx2 = np.argmin(abs(stat[:, 1] - 0.01))
	scores = np.vstack((scores, np.hstack((stat[idx1, 0], stat[idx2, 0]))))
	# print stat[idx1, :2]

print np.mean(scores, axis=0)
print np.std(scores, axis=0)