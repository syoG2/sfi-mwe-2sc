#!/bin/bash

python ../../semantic_frame_induction/make_dataset.py \
    --input_file ../../data/preprocessing/stanza/exemplars.jsonl \
    --output_dir ../../data/semantic_frame_induction/dataset
