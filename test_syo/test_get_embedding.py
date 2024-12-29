import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sfimwe2sc.f_induc.embedding import BaseEmbedding
from sfimwe2sc.f_induc.model import BaseNet
from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl

testData = {"ex_id": 1359456, "ex_idx": 2908, "verb": "keep", "frame": "Activity_ongoing", "verb_frame": "keep:Activity_ongoing", "lu_id": 4347, "lu_name": "keep.v", "text_widx": "“ I keep getting interrupted . ", "target_widx": 2}
df = pd.DataFrame([testData])

pretrained_model_name = "bert-base-uncased"
normalization = False
device = "cuda:3"
layer = 12
vec_type = "word"
batch_size = 1

model = BaseNet(
    pretrained_model_name, normalization, device, layer
)
model.to(device).eval()

embedding = BaseEmbedding(
    model, pretrained_model_name, vec_type, batch_size
)

df_vec, vec_array = embedding.get_embedding(df)

vec_dict = {"vec": vec_array}

print("df_vec")
print(df_vec)

print("vec_dict")
print(vec_dict)