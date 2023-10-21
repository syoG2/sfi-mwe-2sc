#!/bin/bash

source_dir=../../source/preprocessing
data_dir=../../data/preprocessing

cuda=3

CUDA_VISIBLE_DEVICES=${cuda} \
    python ${source_dir}/apply_stanza.py \
    --input_file ${data_dir}/exemplars/exemplars.jsonl \
    --output_dir ${data_dir}/stanza
