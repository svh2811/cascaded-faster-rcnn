#!/bin/bash
#for i in $(seq 0.1 0.7 0.72);
for i in $(seq 1);
do
	echo ${i}
	rm ./vis_images/*
	rm -r ./vis_images_*
	cp -v ../character-faster-rcnn/DataGeneration/maps_red/* ./vis_images/
	python ./visualize_maps_grid_search.py -r ../character-faster-rcnn/map_res/ -t ../character-faster-rcnn/DataGeneration/test_d.txt
  #python ./generate_IoU_report.py -r ../character-faster-rcnn/map_res/words-det-fold-01.txt -t ../character-faster-rcnn/DataGeneration/test.txt
	#rm -r ./results_CNF_NMS_NPYS/
	#mkdir ./results_CNF_NMS_NPYs
	#mkdir ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
	#cp -v ./vis_images/*.npy ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
done
