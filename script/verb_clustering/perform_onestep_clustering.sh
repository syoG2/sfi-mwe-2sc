#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

setting=20-100

pretrained_model_name=bert-base-uncased

# vec_types=(word mask wm)
vec_types=(word)
# vec_types=(mask)
# vec_types=(wm)

clustering_method=average

for vec_type in ${vec_types[@]}; do
    d1=${setting}/${pretrained_model_name}
    d2=${vec_type}/onestep-${clustering_method}
    python ${source_dir}/perform_onestep_clustering.py \
        --input_dir ${data_dir}/embedding/${d1} \
        --output_dir ${data_dir}/onestep_clustering/${d1}/${d2} \
        --input_params_file ${data_dir}/best_params_onestep_clustering/${d1}/${d2}/best_params.json \
        --clustering_method ${clustering_method}
done
