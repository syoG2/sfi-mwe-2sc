#!/bin/bash

source_dir=../../source/preprocessing
data_dir=../../data/preprocessing

python ${source_dir}/make_exemplars.py \
    --output_dir ${data_dir}/exemplars
