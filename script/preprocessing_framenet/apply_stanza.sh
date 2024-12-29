#!/bin/zsh

src_dir=../../source/preprocessing_framenet
data_dir=../../data/preprocessing_framenet

CUDA=3

CUDA_VISIBLE_DEVICES=${CUDA} \
    python ${src_dir}/apply_stanza.py \
    --input_exemplars_file ${data_dir}/collect_exemplars/exemplars.jsonl \
    --output_dir ${data_dir}/apply_stanza
