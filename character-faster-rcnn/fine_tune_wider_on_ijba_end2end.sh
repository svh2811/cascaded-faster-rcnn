#!/bin/bash
GPU_ID=2
orig_solver_file='models/face-from-wider/VGG16/faster_rcnn_end2end/solver.prototxt'
new_solver_file='models/face-from-wider/VGG16/faster_rcnn_end2end/solver_ijba.prototxt'	
cur_dir=`pwd`

for split in `seq 10 10`; do
	# rm ./data/cache/*

	# # prepare solver prototxt (change snapshot prefix)
	# sp=`grep 'snapshot_prefix' ${orig_solver_file}`
	# # snapshot_prefix: "vgg16_faster_rcnn_imagenet_wider_ijba_split_1"
	# new_sp="snapshot_prefix: \"vgg16_faster_rcnn_imagenet_wider_ijba_split_${split}\""
	# cat ${orig_solver_file} | sed -e "s/${sp}/${new_sp}/" > 'tmp.txt'
	# cat 'tmp.txt' > ${new_solver_file}
	# rm 'tmp.txt'

	# imdb_split=ijba_train_split_${split}

	# # run fine tuning
	# time ./tools/train_net.py --gpu ${GPU_ID} \
	#   --solver ${new_solver_file} \
	#   --weights output/faster_rcnn_end2end/vgg16_faster_rcnn_iter_70000.caffemodel \
	#   --imdb ${imdb_split} \
	#   --iters 15000 \
	#   --cfg experiments/cfgs/faster_rcnn_end2end.yml

	# # run face detection on test split
	# weight_file=output/faster_rcnn_end2end/train/vgg16_faster_rcnn_imagenet_wider_ijba_split_${split}_iter_10000.caffemodel
	# python ./tools/run_face_detection_on_ijba.py --gpu $GPU_ID --split ${split} --weight ${weight_file}

	# do evaluation
	cd /home/hzjiang/Code/FDDB/evaluation
	sh ./runEvaluateIJBA.sh ${split}
	cd ${cur_dir}
done
