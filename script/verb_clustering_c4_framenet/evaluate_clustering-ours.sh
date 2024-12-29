#!/bin/bash

source_dir=../../source/verb_clustering_c4_framenet
data_dir=../../data/verb_clustering_c4_framenet

setting=20-100

pretrained_model_name=bert-base-uncased

vec_types=(word mask wm)
# vec_types=(word mask)
# vec_types=(wm)

# clustering_name_methods=(onestep-average twostep-xmeans-average)
clustering_name_methods=(onestep-average)

splits=(dev test)

for vec_type in ${vec_types[@]}; do
    for clustering_name_method in ${clustering_name_methods[@]}; do
        for split in ${splits[@]}; do
            d1=${setting}/${pretrained_model_name}
            d2=${vec_type}/${clustering_name_method}
            python ${source_dir}/evaluate_clustering.py \
                --input_file ${data_dir}/clustering/${d1}/${d2}/exemplars_${split}.jsonl \
                --input_params_file ${data_dir}/clustering/${d1}/${d2}/params.json \
                --output_dir ${data_dir}/evaluate_clustering_ours/${d1}/${d2} \
                --split ${split}
        done
    done
done
