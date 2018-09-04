rm -r img/
rm -r cropped_img/
rm -r fold_1/cropped_annotations.txt
python Datagenerator.py
python GenerateFoldTrain.py fold_1/ fold_2/test.txt fold_3/test.txt fold_4/test.txt fold_5/test.txt
python CropImages.py fold_1/train.txt img/ fold_1/
cp fold_1/cropped_annotations.txt ./train.txt
cp fold_1/test.txt ./test.txt
