#!/bin/bash

src_dir=../../source/preprocessing_c4
data_dir=../../data/preprocessing_c4

split=train
# split=validation

file_id=0

# en: 0~356
part_id_start=0
part_id_end=30

python ${src_dir}/collect_exemplars_and_apply_stanza.py \
    --output_dir_path ${data_dir}/collect_exemplars_and_apply_stanza \
    --split ${split} \
    --file_id ${file_id} \
    --part_ids ${part_id_start} ${part_id_end}
