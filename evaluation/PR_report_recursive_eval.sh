for i in $(seq 0.1 0.7 0.72);
do
	echo ${i}
	rm ./vis_images/*
	cp -v ../maps/* ./vis_images/
	python ./visualize_maps.py -r ../../map_res/words-det-fold_0.88_${i}.txt -t ../data_generation/fold_1/test.txt
	mkdir ./results_CNF_NMS_NPYs
	mkdir ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
	cp -v ./vis_images/*.npy ./results_CNF_NMS_NPYs/CNF0.88_NMS${i}
done
