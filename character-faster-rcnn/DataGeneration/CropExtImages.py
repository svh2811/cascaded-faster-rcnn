import numpy as np 
import cv2
import sys
import os

crop_h = 500
crop_w = 500
step = 400

annotations_filename = sys.argv[1]
annotations = open(annotations_filename, "r").readlines()

images_directory = sys.argv[2]
train_directory = sys.argv[3]

cropped_annotations = open(train_directory+"cropped_annotations.txt", "w")

if os.path.isdir("./croppedext_img") == False:
    os.mkdir("./croppedext_img")

def create_crops(img, img_name, regions):
    
    height = img.shape[0]; width = img.shape[1]
    current_x = 0; current_y = 0
    index = 0

    print(img.shape)
    while current_y + crop_h < height:
        while current_x + crop_w < width:
            crop_img = img[current_y:current_y+crop_h, current_x:current_x+crop_w]

            split_img = img_name.split(".")
            cropped_img_name = "./croppedext_img/"+split_img[0]+"_"+str(index)+"."+split_img[1]

            cropped_regions = []

            for region in regions:
                x, y, w, h = region
                #### Height (h) moves up in index.
                if x > current_x and  x+w < (current_x+crop_w)-50 and y < (current_y+crop_h-50) and y-h > current_y:
                    crop_x = x - current_x
                    crop_y = y - current_y

                    cropped_regions.append( (crop_x, crop_y, w, h) )

#                    cnt = np.array([[int(crop_x), int(crop_y)], [int(crop_x+w), int(crop_y)],
#                                    [int(crop_x+w), int(crop_y-h)], [int(crop_x), int(crop_y-h)]])
#                    cv2.drawContours(crop_img, [cnt], 0, (255, 0, 0), 2)

            cv2.imwrite(cropped_img_name, crop_img)

            if len(cropped_regions) > 0:
                cropped_annotations.write(cropped_img_name+"\n")
                cropped_annotations.write(str(len(cropped_regions))+"\n")
                for r in cropped_regions:
                    cropped_annotations.write( str(r[0]) + " " + str(r[1]) + " " + str(r[2]) + " " + str(r[3]) + "\n")

            index += 1
            current_x += step

        current_x = 0
        current_y += step



image = None
image_name = "" 
regions = None
for line in annotations:
    if line.endswith(".tiff\n"):
        if regions is not None:
            create_crops(image, image_name, regions)

        image_name = line.split("/")[1][:-1]
        print("Reading image: " + image_name)
        image = cv2.imread(images_directory+"/"+image_name)
        regions = []
    elif len(line.split(" ")) == 4:
        split_line = line.split(" ")
        x = float(split_line[0]); y = float(split_line[1])
        r_w = float(split_line[2]); r_h = float(split_line[3])

        regions.append( (x, y, r_w, r_h) )

create_crops(image, image_name, regions)

