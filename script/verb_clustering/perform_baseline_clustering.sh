#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering
input_dir=../../data/raw

setting=20-100

clustering_name=1cpv

d1=${setting}
d2=${clustering_name}
python ${source_dir}/perform_baseline_clustering.py \
    --input_dev_file ${data_dir}/dataset/${d1}/exemplars_dev.jsonl \
    --input_test_file ${data_dir}/dataset/${d1}/exemplars_test.jsonl \
    --output_dir ${data_dir}/baseline_clustering/${d1}/${d2} \
    --clustering_name ${clustering_name}
