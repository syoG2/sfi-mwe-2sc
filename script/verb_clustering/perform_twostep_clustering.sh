#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

setting=20-100

pretrained_model_name=bert-base-uncased

# vec_types=(word mask wm word-mask mask-word word-wm mask-wm wm-word wm-mask wm-wm)
# vec_types=(word)
# vec_types=(mask)
vec_types=(wm)

clustering_method1=xmeans
clustering_method2=average

for vec_type in ${vec_types[@]}; do
    d1=${setting}/${pretrained_model_name}
    d2=${vec_type}/twostep-${clustering_method1}-${clustering_method2}
    python ${source_dir}/perform_twostep_clustering.py \
        --input_dir ${data_dir}/embedding/${d1} \
        --output_dir ${data_dir}/clustering/${d1}/${d2} \
        --input_params_file ${data_dir}/best_params_twostep_clustering/${d1}/${d2}/best_params.json \
        --clustering_method1 ${clustering_method1} \
        --clustering_method2 ${clustering_method2}
done
