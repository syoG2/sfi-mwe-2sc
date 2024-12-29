#!/bin/bash

source_dir=../../source/verb_clustering_c4_framenet
data_dir=../../data/verb_clustering_c4_framenet

setting=20-100

clustering_names=(1cpv)

splits=(dev test)

for clustering_name in ${clustering_names[@]}; do
    for split in ${splits[@]}; do
        d1=${setting}
        d2=${clustering_name}
        python ${source_dir}/evaluate_clustering.py \
            --input_file ${data_dir}/baseline_clustering/${d1}/${d2}/exemplars_${split}.jsonl \
            --input_params_file ${data_dir}/baseline_clustering/${d1}/${d2}/params.json \
            --output_dir ${data_dir}/evaluate_clustering_baseline/${d1}/${d2} \
            --split ${split}
    done
done
