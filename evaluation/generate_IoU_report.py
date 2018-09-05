import numpy as np
import cv2
import os
import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree
import sys
from sys import stdout
from util import retrieve_regions, compute_union, compute_intersection, convert_bbox_format, adjust_image_size
from scipy.optimize import linear_sum_assignment
import math

sys.path.append("../../code/")

from optparse import OptionParser

sys.setrecursionlimit(10000)

parser = OptionParser()
parser.add_option("-t", "--testfile", help="file with groundtruth annotations")
parser.add_option("-r", "--results", help="file with prediction results")

(options, args) = parser.parse_args()

annotation_file = options.testfile
prediction_file = options.results

print("Reading annotations...")
annotated_regions = retrieve_regions(annotation_file)
print("\n\nReading predictions...")
predicted_regions = retrieve_regions(prediction_file)

print("\n")
base_directory = "/Users/fmgarcia/Desktop/MapDetection/maps_project/detection_code"

IoU_by_image = {}
print(predicted_regions.keys())
print(annotated_regions.keys())
for key in predicted_regions.keys():
    print("Reading image: " + key)
    if key in annotated_regions.keys():
        IoU = []
        annotations = annotated_regions[key]
        predictions = predicted_regions[key]


    ###### Using Hungarian algorithm for finding correspondances ###############
    # C = np.zeros( (len(annotations)+null_nodes, len(predictions)) )
    C = np.zeros( (len(predictions), len(annotations)+len(predictions)) )

    for i in range(len(predictions)):
        for j in range(len(annotations)):
            p = predictions[i]
            a = annotations[j]
            intersection = compute_intersection(annotations[j], predictions[i])
            union = compute_union(annotations[j], predictions[i])
            C[i,j] = (1.0 / ((intersection / union) + 1e-3))

    #Give the option of no assignment, which is a bit better than the worst possible assignment
    maxc = np.max(C, axis=1)
    for i in range(C.shape[0]):
        C[i,len(annotations):] = maxc[i] - 1.0

    rows, columns = linear_sum_assignment(C)

    avg_iou = 0.0
    for idx in range(len(rows)):
        row = rows[idx]
        col = columns[idx]
        if col < len(annotations):
            intersection = compute_intersection(predictions[row], annotations[col])
            union = compute_union(predictions[row], annotations[col])

            iou = intersection / union
            avg_iou += iou
            IoU.append( (row, col, iou))

    avg_iou /= len(IoU)
    IoU_by_image[key] = [avg_iou, IoU]

    #############################################################################

    ##### Using KD tree to find correspondences ################
    # annotated_points = [(x[0], x[1]) for x in annotations]
    # kdtree = KDTree(annotated_points, leafsize=10)
    #
    # prediction_pts = [[p[0], p[1]] for p in predictions]
    # matches = kdtree.query(prediction_pts, k=1, p=2)
    #
    # avg_iou = 0.0
    # for i, match in enumerate(matches[1]):
    #
    #     intersection = compute_intersection(annotations[match], predictions[i])
    #     union = compute_union(annotations[match], predictions[i])
    #
    #     iou = intersection / union
    #     avg_iou += iou
    #     IoU.append( (match, i, iou))
    #
    # avg_iou /= len(matches[1])
    # IoU_by_image[key] = [avg_iou, IoU]
    ##########################################################################

report_file = open("iou_report.txt", "w")
for image in IoU_by_image.keys():
    report_file.write(image + " --> " + str(IoU_by_image[image][0]) + "\n")
    print(image + " --> " + str(IoU_by_image[image][0]))


