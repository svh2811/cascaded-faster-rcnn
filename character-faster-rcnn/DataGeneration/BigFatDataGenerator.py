import numpy as np 
import os, sys
import math
import cv2

sys.path.append("../evaluation/")
from util import rotate_image, adjust_image_size
import pickle

annotations_directory = "./annotations/current/"

files = os.listdir(annotations_directory)

fold = 4
total_maps = 0

maps = {}
for f in files:
    if f.endswith(".npy"):
        if f.startswith("D004"):
            total_maps += 1
            data = np.load(annotations_directory+f).item()
            map_name = f.split(".")[0]
            maps[map_name] = []
            for key in data.keys():
                vertices = data[key]['vertices']
                if len(vertices) == 4:
                    maps[map_name].append(vertices)




print("Generating fat annotations...")
boxes_by_map = {}
for map_name in maps.keys():
    vertices = maps[map_name]
    fat_bboxes = []

    for box in vertices:
        minx = min([box[0][0], box[1][0], box[2][0], box[3][0]])
        maxx = max([box[0][0], box[1][0], box[2][0], box[3][0]])
        miny = min([box[0][1], box[1][1], box[2][1], box[3][1]])
        maxy = max([box[0][1], box[1][1], box[2][1], box[3][1]])

        fat_bboxes.append( [minx, maxx, miny, maxy])

    boxes_by_map[map_name] = fat_bboxes


if os.path.isdir("./fat_img/") == False:
    os.mkdir("./fat_img/")

current_fold = 0
annotations = None
for k, mapname in enumerate(boxes_by_map.keys()):

    if k % (total_maps / fold) == 0:
        current_fold += 1
        fold_dir = "./fat_fold_"+str(current_fold)
        if os.path.isdir(fold_dir) == False:
            os.mkdir(fold_dir)

        if annotations is not None:
            annotations.close()
        annotations = open(fold_dir+"/test.txt", "w")

    print("Writing map " + mapname)
    map_img = cv2.imread("./maps/" + mapname + ".tiff")


    vertices = maps[mapname]

    fat_rec_imgname = mapname+".tiff"
    if len(vertices) > 0:
        annotations.write("fat_img/" + fat_rec_imgname + "\n")
        annotations.write(str(len(vertices))+"\n")

        for i, vertex in enumerate(vertices):

            pt1 = [int(vertex[0][0]), int(vertex[0][1])]
            pt2 = [int(vertex[1][0]), int(vertex[1][1])]
            pt3 = [int(vertex[2][0]), int(vertex[2][1])]
            pt4 = [int(vertex[3][0]), int(vertex[3][1])]

            ############# Visualize bounding boxes on original image ##################
            # cnt = np.array([pt1, pt2, pt3, pt4])
            # cv2.drawContours(map_img, [cnt], 0, (255, 0, 0), 5)
            ##########################################################################

            fat_bbox = boxes_by_map[mapname][i]

            ########## Visualize mapped bounding boxes after rotation ###############
            pt1 = [int(fat_bbox[0]), int(fat_bbox[2])]
            pt2 = [int(fat_bbox[1]), int(fat_bbox[2])]
            pt3 = [int(fat_bbox[1]), int(fat_bbox[3])]
            pt4 = [int(fat_bbox[0]), int(fat_bbox[3])]
            
            # cnt = np.array([pt1, pt2, pt3, pt4])
            # cv2.drawContours(map_img, [cnt], 0, (0, 255, 0), 7)
            #########################################################################

            width =  math.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )
            height = math.sqrt( (pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2 )
            string = str(pt1[0]) + " " + str(pt1[1]) + " " + str(width) + " " + str(height)
            annotations.write(string+"\n")

	print("Writing map")
        cv2.imwrite("./fat_img/" + fat_rec_imgname, map_img)
