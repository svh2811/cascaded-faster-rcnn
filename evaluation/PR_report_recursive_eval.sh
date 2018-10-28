#!/bin/bash
#for i in $(seq 0.1 0.7 0.72);
for i in $(seq 1);
do
	echo ${i}
	rm ./vis_images/*
	cp -v ../character-faster-rcnn/DataGeneration/maps_red/* ./vis_images/
#	python ./visualize_maps.py -r ../../map_res/words-det-fold_0.88_${i}.txt -t ../data_generation/fold_1/test.txt
	python ./visualize_maps.py -r ../character-faster-rcnn/map_res/words-det2-D0090-5242001-01.txt -t ../character-faster-rcnn/DataGeneration/test_d.txt
        #python ./generate_IoU_report.py -r ../character-faster-rcnn/map_res/words-det-fold-01.txt -t ../character-faster-rcnn/DataGeneration/test.txt
	rm -r ./results_CNF_NMS_NPYS/
	mkdir ./results_CNF_NMS_NPYs
	mkdir ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
	cp -v ./vis_images/*.npy ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
done
