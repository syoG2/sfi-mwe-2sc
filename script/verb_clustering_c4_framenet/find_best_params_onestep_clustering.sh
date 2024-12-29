#!/bin/bash

source_dir=../../source/verb_clustering_c4_framenet
data_dir=../../data/verb_clustering_c4_framenet

setting=20-100

pretrained_model_name=bert-base-uncased

vec_types=(word mask wm)
# vec_types=(word)
# vec_types=(mask)
# vec_types=(wm)

layers=(00 01 02 03 04 05 06 07 08 09 10 11 12)

clustering_method=average

for vec_type in ${vec_types[@]}; do
    d1=${setting}/${pretrained_model_name}
    d2=${vec_type}/onestep-${clustering_method}
    python ${source_dir}/find_best_params_onestep_clustering.py \
        --input_dir ${data_dir}/embedding/${d1} \
        --output_dir ${data_dir}/best_params_onestep_clustering/${d1}/${d2} \
        --vec_type ${vec_type} \
        --layers ${layers[@]} \
        --clustering_method ${clustering_method}
done
