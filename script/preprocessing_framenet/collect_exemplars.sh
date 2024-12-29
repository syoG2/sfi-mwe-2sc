#!/bin/zsh

src_dir=../../source/preprocessing_framenet
data_dir=../../data/preprocessing_framenet

python ${src_dir}/collect_exemplars.py \
    --output_dir_path ${data_dir}/collect_exemplars
