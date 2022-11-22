#!/bin/bash

CUDA=3

CUDA_VISIBLE_DEVICES=${CUDA} \
    python ../../preprocessing/apply_stanza.py \
    --input_file ../../data/preprocessing/exemplars/exemplars.jsonl \
    --output_dir ../../data/preprocessing/stanza
