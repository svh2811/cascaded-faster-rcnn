import numpy as np
import os, sys
import math
import cv2

current_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append("../evaluation/")

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



if os.path.isdir("./img/") == False:
    os.mkdir("./img/")

print("Generating r2cnn annotations...")

current_fold = 0
aligned_annotations, inclined_annotations = None, None
for k, map_name in enumerate(maps.keys()):
    vertices = maps[map_name]
    fat_bboxes, incline_bboxes = [], []

    for box in vertices:
        minx = min([box[0][0], box[1][0], box[2][0], box[3][0]])
        maxx = max([box[0][0], box[1][0], box[2][0], box[3][0]])
        miny = min([box[0][1], box[1][1], box[2][1], box[3][1]])
        maxy = max([box[0][1], box[1][1], box[2][1], box[3][1]])

        fat_bboxes.append( [minx, maxx, maxy, miny])
        incline_height = math.sqrt((box[2][0]-box[1][0])**2 + (box[2][1]-box[1][1])**2)
        incline_bboxes.append( [box[0][0], box[0][1], box[1][0], box[1][1], incline_height])

    if k % (total_maps / fold) == 0:
        current_fold += 1
        fold_dir = "./fold_"+str(current_fold)
        if os.path.isdir(fold_dir) == False:
            os.mkdir(fold_dir)

        if aligned_annotations is not None:
            aligned_annotations.close()
            inclined_annotations.close()
        aligned_annotations = open(fold_dir+"/simple_aligned_test.txt", "w")
        inclined_annotations = open(fold_dir+"/simple_incline_test.txt", "w")

    print("Writing map " + map_name)
    map_img = cv2.imread("./maps/" + map_name + ".tiff")

    fat_rec_imgname = map_name+".tiff"
    if len(vertices) > 0:

        for i, vertex in enumerate(vertices):

            # print("vertex " + str(i))
            pt1 = [int(vertex[0][0]), int(vertex[0][1])]
            pt2 = [int(vertex[1][0]), int(vertex[1][1])]
            pt3 = [int(vertex[2][0]), int(vertex[2][1])]
            pt4 = [int(vertex[3][0]), int(vertex[3][1])]

            ############# Visualize bounding boxes on original image ##################
            # cnt = np.array([pt1, pt2, pt3, pt4])
            # cv2.drawContours(map_img, [cnt], 0, (255, 0, 0), 5)
            ##########################################################################

            fat_bbox = fat_bboxes[i]
            incline_bbox = incline_bboxes[i]

            pt1 = [int(fat_bbox[0]), int(fat_bbox[2])]
            pt2 = [int(fat_bbox[1]), int(fat_bbox[2])]
            pt3 = [int(fat_bbox[1]), int(fat_bbox[3])]
            pt4 = [int(fat_bbox[0]), int(fat_bbox[3])]

            ############# Visualize fat bounding boxes ##################
            # cnt = np.array([pt1, pt2, pt3, pt4])
            # cv2.drawContours(map_img, [cnt], 0, (255, 0, 0), 5)
            ##########################################################################


            # angle = math.atan2(incline_bbox[1] - incline_bbox[3], incline_bbox[2] - incline_bbox[0])
            #
            # pt3x = incline_bbox[2]+incline_bbox[4]*math.cos(angle+(math.pi/2.0))
            # pt3y = incline_bbox[3]-incline_bbox[4]*math.sin(angle+(math.pi/2.0))
            #
            # pt4x = incline_bbox[0]+incline_bbox[4]*math.cos(angle+(math.pi/2.0))
            # pt4y = incline_bbox[1]-incline_bbox[4]*math.sin(angle+(math.pi/2.0))
            #
            # pt1inc = [int(incline_bbox[0]), int(incline_bbox[1])]
            # pt2inc = [int(incline_bbox[2]), int(incline_bbox[3])]
            # pt3inc = [int(pt3x), int(pt3y)]
            # pt4inc = [int(pt4x), int(pt4y)]
            # cnt = np.array([pt1inc, pt2inc, pt3inc, pt4inc])
            # cv2.drawContours(map_img, [cnt], 0, (0, 255, 0), 7)
            #########################################################################

            width =  math.sqrt( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )
            height = math.sqrt( (pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2 )
            string = current_path+"/img/"+fat_rec_imgname + "," +str(int(fat_bbox[0])) + "," + str(int(fat_bbox[3])) + "," + str(int(fat_bbox[1])) + "," + str(int(fat_bbox[2])) + ",text\n"
            aligned_annotations.write(string)

            incline_str = current_path+"/img/"+fat_rec_imgname + "," +str(int(incline_bbox[0])) + "," + str(int(incline_bbox[1])) + \
                          "," + str(int(incline_bbox[2])) + "," + str(int(incline_bbox[3])) + "," + str(int(incline_bbox[4])) + ",text\n"

            inclined_annotations.write(incline_str)

    print("Writing map")
    cv2.imwrite("./img/" + fat_rec_imgname, map_img)
