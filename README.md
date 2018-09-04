Instructions for running faster-rcnn in any computer

You should have the faster-rcnn folder and the file map_data_converter.py

You should also need: caffe, open-cv

Annotations_file: updated in Dido: at /media/archan/maps_project/annotations/current

1. Convert annotations to canonical form. Run `python map_data_converter.py`

2. go to Datageneration (directory)

3.a. clear contents of annotations/current/
3.b. copy all annotations to annotations/current/ (directory)

4.a. delete contents of maps/ (directory)
4.b. Copy all the annotated maps from mapOutput_3/ to maps/ (directory)

5. Delete all directories named: fold_*, cropped_img, img. Run: rm -r fold_* cropped_img img

6a. Change DataGenerator.py’s annotations file name and the maps directory

6. run: `python DataGenerator.py`: this creates 5 folders named img, fold_1, .., fold_5 with one file named test.txt in each of the fold* folders (if you have different annotations and maps directory you WILL need to fix them in the code).

7. run: `python GenerateFoldTrain.py fold_1/ fold_2/test.txt fold_3/test.txt fold_4/test.txt fold_5/test.txt`: this creates a file called train.txt in fold_1

8. run `python CropImages.py fold_1/train.txt img/ fold_1/`: this reads contents of train.txt in fold_1 and crops them for the required sizes and creates a file called cropped_annotations.txt in fold_1

9. copy: `cp fold_1/cropped_annotations.txt ./train.txt`

10.copy: ` cp fold_1/test.txt ./test.txt`

11.a. go to faster_rcnn folder

11.b. go to lib/datasets/factory.py. Change line 50 according to your path to the train.txt file


11.c. run `./train_img.sh` This starts training the model. The -iters switch contains the number of trainings iterations. remember this as you would require while testing. Lets say you put this number as 288.

Then this will save the trained model in: 
'./output/faster_rcnn_end2end/train/vgg16_faster_rcnn_map_iter_288.caffemodel'

11.d FOR GYPSUM ONLY: source ~/.fasterrcnn-bashrc

12. TESTING.
1. so change line 19 in `./tools/run_words_detection_sliding_crop.py` as per your training model.
2. go to faster-rcnn directory. run `./test_img.sh`: This creates all test files in the faster_rcnn directory with the extension `.tiff`. (WORKS FOR HORIZONTAL DETECTIONS ONLY)
3. (ALL NON HORIZONTAL DETECTIONS) run `./test_img.sh`. 

You can copy this files to wherever you want ands see the performance.


