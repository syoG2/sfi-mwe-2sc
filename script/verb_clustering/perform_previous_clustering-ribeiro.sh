#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering
input_dir=../../data/raw

setting=20-100

clustering_name=ribeiro

device=cuda:1
d1=${setting}
d2=${clustering_name}
python ${source_dir}/perform_previous_clustering.py \
    --input_dev_file ${data_dir}/dataset/${d1}/exemplars_dev.jsonl \
    --input_test_file ${data_dir}/dataset/${d1}/exemplars_test.jsonl \
    --output_dir ${data_dir}/previous_clustering/${d1}/${d2} \
    --input_elmo_options_file ${input_dir}/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json \
    --input_elmo_weights_file ${input_dir}/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
    --clustering_name ${clustering_name} \
    --batch_size 32 \
    --device ${device}
