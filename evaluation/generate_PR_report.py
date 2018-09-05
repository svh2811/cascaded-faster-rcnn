
import numpy as np
import cv2
import os
import numpy as np
import scipy.io as sio
from scipy.spatial import KDTree
import sys
from sys import stdout
from util import retrieve_regions, compute_union, compute_intersection, convert_bbox_format, adjust_image_size, filter_predictions
from scipy.optimize import linear_sum_assignment
import math
import pickle

sys.path.append("../code/")
from compute_scores import PR_compute

from optparse import OptionParser

sys.setrecursionlimit(10000)

parser = OptionParser()
parser.add_option("-t", "--testfile", help="file with groundtruth annotations")
parser.add_option("-r", "--results", help="file with prediction results")
parser.add_option("-d", "--dir_output", help="output directory", default="./")
parser.add_option("-i", "--base_directory", help="input directory", default="./")

(options, args) = parser.parse_args()

annotation_file = options.testfile
prediction_file = options.results
output_dir = options.dir_output
base_directory = options.base_directory


print("Reading annotations...")
annotated_regions = retrieve_regions(annotation_file)
print("\n\nReading predictions...")
predicted_regions = retrieve_regions(prediction_file)

print("\n")
#base_directory = ".."

IoU_by_image = {}
print(predicted_regions.keys())
print(annotated_regions.keys())
for key in predicted_regions.keys():
    if key in annotated_regions.keys():
        annotations = annotated_regions[key]
        predictions = predicted_regions[key]

        original_image = base_directory + "/maps/" + key
        img = cv2.imread(original_image)
        padding_amount = 500
        img, translate = adjust_image_size(img, padding_amount)
        pivot = (img.shape[1] // 2, img.shape[0] // 2)

        annotation_np = np.zeros( (len(annotations), 8) )
        for i, a in enumerate(annotations):
            #Minus angle because we are not rotating the image here
            transformed_box = convert_bbox_format((a[0], a[1], a[2], a[3]), angle=-a[4], pivot=pivot)
            annotation_np[i] = transformed_box

        PR_values = []
        #thresholds = [0.2, 0.4, 0.6, 0.8, 0.9]
        thresholds = [0.8]
        
	for e in thresholds:
	    filtered_predictions = []
	    for i, p in enumerate(predictions):
                if p[4] > e:
		    transformed_box = convert_bbox_format((p[0], p[1], p[2], p[3]), angle=-p[5], pivot=pivot)
                    filtered_predictions.append( transformed_box )

            filtered_predictions = filter_predictions(filtered_predictions)
            predictions_np = np.zeros( (len(filtered_predictions), 8) )
            for i, p in enumerate(filtered_predictions):
                predictions_np[i] = p


            print("Annotations: " + str(annotation_np.shape[0]))
            print("Predictions: " + str(predictions_np.shape[0]))
            
	    print("Computing PR curve for: " + str(e))
            PR = PR_compute(annotation_np, predictions_np, annotation_np.shape[0], predictions_np.shape[0], threshold=0.5)
            print("PR: " + str(PR))
            # np.save("./annotation.npy", annotation_np)
            # np.save("./prediction.npy", prediction_mat)
            
            PR_values.append( (PR, e) )

        pickle.dump( PR_values, open(output_dir+key+"_PR.pkl", "wb"))
