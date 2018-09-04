import numpy as np
import cv2
import sys
import os
import math


crop_h = 500
crop_w = 500
step = 400

IGNORE_EMPTY_CROPS = True

aligned_filename = sys.argv[1]
aligned_annotations = open(aligned_filename, "r").readlines()

incline_filename = sys.argv[2]
incline_annotations = open(incline_filename, "r").readlines()

images_directory = sys.argv[3]
train_directory = sys.argv[4]

copped_aligned_annotations = open(train_directory+"simple_cropped_align.txt", "w")
copped_incline_annotations = open(train_directory+"simple_cropped_incline.txt", "w")
current_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir("./cropped_img") == False:
    os.mkdir("./cropped_img")

def create_crops(regions):

    for key in regions:
        img = cv2.imread(key)
        img_name = key.split("/")[-1]

        height = img.shape[0]; width = img.shape[1]
        current_x = 0; current_y = 0
        index = 0

        print(img.shape)
        while current_y + crop_h < height:
            while current_x + crop_w < width:
                crop_img = img[current_y:current_y+crop_h, current_x:current_x+crop_w]

                split_img = img_name.split(".")
                cropped_img_name = "cropped_img/"+split_img[0]+"_"+str(index)+"."+split_img[1]

                cropped_aligned, cropped_incline = [], []

                for i in range(len(regions[key]["aligned_bbox"])):
                    aligned_bbox = regions[key]["aligned_bbox"][i]
                    x1a, y1a, x2a, y2a = aligned_bbox
                    x = x1a
                    y = y1a
                    w = x2a - x1a
                    ha = y2a - y1a

                    incline_bbox = None
                    for box in regions[key]["incline_bbox"]:
                        x1, y1, x2, y2, hi = box
                        if x1 >= x and x2 <= x+w and y1 >= y and y1 <= y+ha and y2 >= y and y2 <= y+ha:
                            incline_bbox = box
                            break

                    x1, y1, x2, y2, hi = incline_bbox

                    #### Height (h) moves up in index.
                    if x > current_x and  x+w < (current_x+crop_w)-50 and y < (current_y+crop_h-50) and y-ha > current_y:
                        aligned_x = x - current_x
                        aligned_y = y - current_y

                        cropped_aligned.append( (aligned_x, aligned_y, aligned_x+w, aligned_y+ha) )

                        incline_x1 = x1 - current_x
                        incline_y1 = crop_h - (y1 - current_y)
                        incline_x2 = x2 - current_x
                        incline_y2 = crop_h - (y2 - current_y)

                        cropped_incline.append( (incline_x1, incline_y1, incline_x2, incline_y2, hi))

                        ############ Visualize aligned bbox ########################################
                        # cnt = np.array([[int(aligned_x), int(aligned_y)], [int(aligned_x+w), int(aligned_y)],
                        #                [int(aligned_x+w), int(aligned_y+ha)], [int(aligned_x), int(aligned_y+ha)]])
                        # cv2.drawContours(crop_img, [cnt], 0, (255, 0, 0), 7)
                        ###############################################################################

                        ############ Visualize incline bbox ##########################################
                        # angle = math.atan2(incline_y2 - incline_y1, incline_x2 - incline_x1)
                        #
                        # pt3x = incline_x2+hi*math.cos(angle+(math.pi/2.0))
                        # pt3y = incline_y2+hi*math.sin(angle+(math.pi/2.0))
                        #
                        # pt4x = incline_x1+hi*math.cos(angle+(math.pi/2.0))
                        # pt4y = incline_y1+hi*math.sin(angle+(math.pi/2.0))
                        #
                        # pt1inc = [int(incline_x1), int(crop_h-incline_y1)]
                        # pt2inc = [int(incline_x2), int(crop_h-incline_y2)]
                        # pt3inc = [int(pt3x), int(crop_h-pt3y)]
                        # pt4inc = [int(pt4x), int(crop_h-pt4y)]
                        # cnt = np.array([pt1inc, pt2inc, pt3inc, pt4inc])
                        # cv2.drawContours(crop_img, [cnt], 0, (0, 255, 0), 2)
                        #########################################################################

                cv2.imwrite(cropped_img_name, crop_img)


                if IGNORE_EMPTY_CROPS == False or (len(cropped_aligned) > 0 and IGNORE_EMPTY_CROPS):
                    for i in range(len(cropped_aligned)):
                        ra = cropped_aligned[i]
                        ri = cropped_incline[i]
                        img_path = current_path+"/"+cropped_img_name
                        copped_aligned_annotations.write( img_path + "," + str(ra[0]) + "," + str(ra[1]) + "," + str(ra[2]) + "," + str(ra[3]) + ",text\n")
                        copped_incline_annotations.write( img_path + "," + str(ri[0]) + "," + str(ri[1]) + "," + str(ri[2]) + "," + str(ri[3]) + "," + str(ri[4]) + ",text\n")

                index += 1
                current_x += step

            current_x = 0
            current_y += step



region_by_img = {}
for line in aligned_annotations:
    (filepath, x1, y1, w, h, class_type) = line.split(",")
    image_name = filepath #filepath.split("/")[-1]
    if image_name not in region_by_img:
        region_by_img[image_name] = {}
    else:
        if "aligned_bbox" not in region_by_img[image_name]:
            region_by_img[image_name]["aligned_bbox"] = [(int(x1), int(y1),int(w), int(h))]
        else:
            region_by_img[image_name]["aligned_bbox"].append( (int(x1), int(y1),int(w), int(h)))

for line in incline_annotations:
    (filepath, x1, y1, x2, y2, h, class_type) = line.split(",")
    image_name = filepath #filepath.split("/")[-1]
    if image_name not in region_by_img:
        region_by_img[image_name] = {}
    else:
        if "incline_bbox" not in region_by_img[image_name]:
            region_by_img[image_name]["incline_bbox"] = [(int(x1), int(y1), int(x2), int(y2), int(h))]
        else:
            region_by_img[image_name]["incline_bbox"].append( (int(x1), int(y1), int(x2), int(y2), int(h)))

create_crops(region_by_img)

