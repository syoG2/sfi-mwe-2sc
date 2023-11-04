#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering
input_dir=../../data/preprocessing

min_n_examples=20
max_n_examples=100

setting=${min_n_examples}-${max_n_examples}

d1=${setting}
python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/stanza/exemplars.jsonl \
    --output_dir ${data_dir}/dataset/${d1} \
    --min_n_examples ${min_n_examples} \
    --max_n_examples ${max_n_examples} \
    --random_state 0
