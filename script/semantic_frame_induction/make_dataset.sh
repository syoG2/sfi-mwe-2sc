#!/bin/bash

source_dir=../../source/semantic_frame_induction
data_dir=../../data/semantic_frame_induction
input_dir=../../data/preprocessing

python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/stanza/exemplars.jsonl \
    --output_dir ${data_dir}/dataset
