#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

setting=20-100

pretrained_model_name=bert-base-uncased

# vec_types=(word mask wm word-mask mask-word word-wm mask-wm wm-word wm-mask wm-wm)
# vec_types=(word)
# vec_types=(mask)
vec_types=(wm)

layers=(00 01 02 03 04 05 06 07 08 09 10 11 12)

clustering_name=twostep
clustering_method1=xmeans
clustering_method2=average

for vec_type in ${vec_types[@]}; do
    d1=${setting}/${pretrained_model_name}
    d2=${vec_type}/${clustering_name}-${clustering_method1}-${clustering_method2}
    python ${source_dir}/find_best_params_twostep_clustering.py \
        --input_dir ${data_dir}/embedding/${d1} \
        --output_dir ${data_dir}/best_params_twostep_clustering/${d1}/${d2} \
        --vec_type ${vec_type} \
        --layers ${layers[@]} \
        --clustering_name ${clustering_name} \
        --clustering_method1 ${clustering_method1} \
        --clustering_method2 ${clustering_method2}
done
