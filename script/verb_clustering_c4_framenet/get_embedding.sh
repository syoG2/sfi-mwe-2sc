#!/bin/bash

source_dir=../../source/verb_clustering_c4_framenet
data_dir=../../data/verb_clustering_c4_framenet

setting=20-100
pretrained_model_name=bert-base-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-large

vec_types=(word mask)
# vec_types=(word)
# vec_types=(mask)

# layers=(00 01 02 03 04 05 06 07 08 09 10 11 12)
# layers=(00 01 02 03)
# layers=(04 05 06 07)
layers=(08 09 10 11 12)

# layers=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

splits=(dev test)
# splits=(dev)
# splits=(test)

# device=cuda:0
# device=cuda:1
# device=cuda:2
device=cuda:3

for vec_type in ${vec_types[@]}; do
    for layer in ${layers[@]}; do
        for split in ${splits[@]}; do
            d1=${setting}
            d2=${pretrained_model_name}/${vec_type}/${layer}
            python ${source_dir}/get_embedding.py \
                --input_file ${data_dir}/dataset/${d1}/exemplars_"${split}".jsonl \
                --output_dir ${data_dir}/embedding/${d1}/"${d2}" \
                --pretrained_model_name ${pretrained_model_name} \
                --vec_type "${vec_type}" \
                --layer "${layer}" \
                --normalization false \
                --batch_size 32 \
                --device ${device} \
                --split "${split}"
        done
    done
done
