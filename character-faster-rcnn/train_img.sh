#!/bin/bash

rm -r ./data/cache/
python ./tools/train_net.py --gpu 0 --solver ./models/map/VGG16/faster_rcnn_end2end/solver.prototxt --weights ./data/imagenet_models/VGG16.v2.caffemodel --iters 10010 --cfg ./experiments/cfgs/faster_rcnn_end2end.yml --imdb "map_train" 2>&1 | tee ./out_file.txt
