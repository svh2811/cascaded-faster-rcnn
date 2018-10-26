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
parser.add_option("-r", "--results", help="file with prediction results")

(options, args) = parser.parse_args()

annotation_file = options.testfile
prediction_file = options.results
print prediction_file

predicted = retrieve_regions(options.results)
#annotated = retrieve_regions(options.testfile)

print("\nFinished reading...")

merged_predictions = predicted
merged = True

# while merged is True:
#     merged = False
#     for img_key in merged_predictions.keys():
#         predictions = merged_predictions[img_key]
#         print("Predictions: " + str(len(predictions)))

#         used_regions = []
#         merged_regions = []
#         for p in predictions:
#             if p not in used_regions:
#                 best_candidate = None
#                 best_score = -1.0
#                 for candidate in predictions:
#                     if p != candidate and candidate not in used_regions:
#                         bbox_pt = (p[0], p[1], 
#                                     p[0]+p[2], p[1], 
#                                     p[0]+p[2], p[1]+p[3], 
#                                     p[0], p[1]+p[3])
#                         candidate_pt = (candidate[0], candidate[1], 
#                                         candidate[0]+candidate[2], candidate[1], 
#                                         candidate[0]+candidate[2], candidate[1]+candidate[3], 
#                                         candidate[0], candidate[1]+candidate[3])

#                         intersection = compute_intersection(bbox_pt, candidate_pt)
#                         union = compute_union(p, candidate, intersection)

#                         iou = intersection / union
#                         if iou > best_score:
#                             best_score = iou
#                             best_candidate = candidate

#                 used_regions.append( p )
#                 region = (p[0], p[1], p[2], p[3], p[4])
#                 if best_score > 0.5:
#                     used_regions.append( best_candidate )

#                     xmin = min(p[0], best_candidate[0])
#                     ymin = min(p[1], best_candidate[1])
#                     xmax = max(p[0]+p[2], best_candidate[0]+best_candidate[2])
#                     ymax = max(p[1]+p[3], best_candidate[1]+best_candidate[3])

#                     region = (xmin, ymin, xmax-xmin, ymax-ymin, (p[4]+best_candidate[4])/2.0)
#                     merged = True

#                 merged_regions.append( region )

#         merged_predictions[img_key] = merged_regions



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


        # full_vis_image = img.copy()
        #
        # for angle in prediction_by_angle:
        #     print("Rotating angle " + str(angle))
        #     predictions = prediction_by_angle[angle]
        #     annotations = None if angle not in annotation_by_angle else annotation_by_angle[angle]
        #     rot_img, rotation_mat, bounds = rotate_image(img, angle, img_shape)
        #
        #     vis_img = vis_detections_bbox(image=rot_img, annotations=annotations, predictions=predictions, threshold=0.1)
        #
        #     image, rot_mat, bounds = rotate_image(vis_img, -angle, img_shape)
        #     img_name=key.split(".")[0]+"_"+str(angle)
        #
        #     cv2.imwrite(img_directory+"/"+img_name+".png", image)
        #
        #     full_vis_image, rotation_mat, bounds = rotate_image(full_vis_image, angle, img_shape)
        #     full_vis_image = vis_detections_bbox(image=full_vis_image, annotations=annotations, predictions=predictions, threshold=0.1)
        #     full_vis_image, rot_mat, bounds = rotate_image(full_vis_image, -angle, img_shape)
        #
        #
        # cv2.imwrite(img_directory+"/"+key+".png", full_vis_image)



