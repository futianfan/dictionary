#! /bin/bash

awk -F "\t" '{print $2}' data/SNOW_vocabMAP.txt > data/heartfailure_code_map.txt

python src/train_tf.py
python src/interpret.py




