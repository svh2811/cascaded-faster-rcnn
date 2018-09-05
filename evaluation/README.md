1. copy the files in `maps` directory to a directory called `vis_images` in the root folder
2. copy `txt` files in `map_res` directory to `map_res` directory here.
3. copy path to the `test.txt` file; without altering anything this will be `..\word-faster-rcnn\DataGeneration\test.txt`

4. python ./visualize_maps.py -r .map_res/words-det-fold_*.txt -t ..\word-faster-rcnn\DataGeneration\test.txt

5. mkdir ./results_CNF_NMS_NPYs
6. mkdir ./results_CNF_NMS_NPYs/CNF{i}_NMS${j}, replace {i} and {j} with your confidence and nms values
7. cp -v ./vis_images/*.npy ./results_CNF_NMS_NPYs/CNF{i}_NMS${j} same drill as point 6.
8. outputs will be in folder vis_images

if you have results for multiple values you can use the following line:
run PR_report_recursive_eval.sh


