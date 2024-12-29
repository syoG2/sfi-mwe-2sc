#!/bin/bash

source_dir=../../source/verb_clustering_c4_framenet
data_dir=../../data/verb_clustering_c4_framenet

setting=20-100

pretrained_model_name=bert-base-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-large

vec_types=(word mask wm word-mask mask-word word-wm mask-wm wm-word wm-mask wm-wm)
# vec_types=(word-wm mask-wm word mask word-mask mask-word wm-wm)
# vec_types=(word-wm word word-mask)
# vec_types=(mask-wm mask mask-word)
# vec_types=(wm-wm)

layers=(00 01 02 03 04 05 06 07 08 09 10 11 12)
# layers=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

clustering_methods1=(xmeans average 1cpv)
# clustering_methods1=(1cpv)
# clustering_methods1=(xmeans)
# clustering_methods1=(average)

clustering_methods2=(ward average)
# clustering_methods2=(ward)
# clustering_methods2=(average)

for clustering_method1 in ${clustering_methods1[@]}; do
    for clustering_method2 in ${clustering_methods2[@]}; do
        for vec_type in ${vec_types[@]}; do
            d1=${setting}/${pretrained_model_name}
            d2=${vec_type}/twostep-${clustering_method1}-${clustering_method2}

            python ${source_dir}/find_best_params_twostep_clustering.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --output_dir ${data_dir}/best_params_twostep_clustering/${d1}/${d2} \
                --vec_type ${vec_type} \
                --layers ${layers[@]} \
                --clustering_method1 ${clustering_method1} \
                --clustering_method2 ${clustering_method2}

            python ${source_dir}/perform_twostep_clustering.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --output_dir ${data_dir}/clustering/${d1}/${d2} \
                --input_params_file ${data_dir}/best_params_twostep_clustering/${d1}/${d2}/best_params.json \
                --clustering_method1 ${clustering_method1} \
                --clustering_method2 ${clustering_method2}

            splits=(dev test)
            for split in ${splits[@]}; do
                python ${source_dir}/evaluate_clustering.py \
                    --input_file ${data_dir}/clustering/${d1}/${d2}/exemplars_${split}.jsonl \
                    --input_params_file ${data_dir}/clustering/${d1}/${d2}/params.json \
                    --output_dir ${data_dir}/evaluate_clustering_ours/${d1}/${d2} \
                    --split ${split}
            done
        done
    done
done
