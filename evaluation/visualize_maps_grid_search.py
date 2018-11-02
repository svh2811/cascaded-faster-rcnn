import cv2
import os
import numpy as np
import scipy.io as sio
import sys

from optparse import OptionParser

from util import retrieve_regions, vis_detections_pts, compute_union, compute_intersection
from util import vis_detections_bbox, rotate_image, adjust_image_size, convert_bbox_format, filter_predictions

base_directory = ".."

parser = OptionParser()
parser.add_option("-t", "--testfile", help="file with groundtruth annotations")
parser.add_option("-r", "--results", help="folder with prediction results")

(options, args) = parser.parse_args()

annotation_file = options.testfile
prediction_folder = options.results

import sys
sys.path.append("../character-faster-rcnn/tools")
import config

for nms_thresh in config.NMS_THRESH_LIST:
    for conf_tresh in config.CONF_THRESH_LIST:
        hyper_params = str(nms_thresh) + "_" + str(conf_tresh)
        print(hyper_params)
        prediction_file =  prediction_folder + "words-det-" + hyper_params + "_" +  config.FID_EXTENSION + "-D0090-5242001-01.txt"
        print(prediction_file)

        predicted = retrieve_regions(prediction_file)
        #annotated = retrieve_regions(options.testfile)

        print("\nFinished reading...")

        merged_predictions = predicted
        merged = True

exit()

if os.path.isdir(base_directory+"/evaluation/vis_images") == False:
    os.mkdir(base_directory+"/evaluation/vis_images/")

for key in merged_predictions.keys():
    #if key in annotated.keys():
        print("\nCreating image for:  " + key)

        original_image = "./vis_images/"+key
	print original_image
	print('reading image',original_image)
        img = cv2.imread(original_image)

        predictions = merged_predictions[key]
        prediction_by_angle = {}
        annotation_by_angle = {}
        for p in predictions:
            if p[5] not in prediction_by_angle:
                prediction_by_angle[p[5]] = [p[:5]]
            else:
                prediction_by_angle[p[5]].append( p[:5] )

     #   annotations = annotated[key]
     #  for a in annotations:
     #       if a[4] not in annotation_by_angle:
     #           annotation_by_angle[a[4]] = [a[:4]]
     #       else:
     #           annotation_by_angle[a[4]].append( a[:4] )

        img_directory = base_directory+"/evaluation/vis_images"

        if os.path.isdir(img_directory) == False:
            os.mkdir(img_directory)


        #padding_amount = 500
        #img, translate = adjust_image_size(img, padding_amount)
	translate = (0,0)
        img_shape = img.shape
        pivot = (img_shape[1] // 2, img_shape[0] // 2)

        all_predictions = []
        all_annotations = []
        for angle in prediction_by_angle:
            for pred in prediction_by_angle[angle]:
                corners = convert_bbox_format(pred, -angle, pivot=pivot)
                all_predictions.append( corners )

        #print all_predictions

        filtered_predictions = all_predictions
        #filtered_predictions = filter_predictions(all_predictions)

      #  for angle in annotation_by_angle:
      #      for annotation in annotation_by_angle[angle]:
      #          corners = convert_bbox_format(annotation, -angle, pivot=pivot)
      #          all_annotations.append( corners )



        image, cnt = vis_detections_pts(img, predictions=filtered_predictions, threshold=0.6, annotations=all_annotations)
        cv2.imwrite(img_directory+"/"+key, image)
        #print cnt
        np.save(img_directory+"/"+key+'.npy', cnt)
