import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sfimwe2sc.f_induc.embedding import BaseEmbedding
from sfimwe2sc.f_induc.model import BaseNet
from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))

    model = BaseNet(
        args.pretrained_model_name, args.normalization, args.device, args.layer
    )
    model.to(args.device).eval()

    embedding = BaseEmbedding(
        model, args.pretrained_model_name, args.vec_type, args.batch_size
    )
    df_vec, vec_array = embedding.get_embedding(df)
    write_jsonl(
        df_vec.to_dict("records"),
        args.output_dir / f"exemplars_{args.split}.jsonl",
    )
    vec_dict = {"vec": vec_array}
    np.savez_compressed(args.output_dir / f"vec_{args.split}", **vec_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--pretrained_model_name", type=str, default="bert-base-uncased"
    )
    parser.add_argument("--vec_type", type=str, choices=["word", "mask"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--normalization", type=str, default="false")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--split", type=str, choices=["dev", "test"])
    args = parser.parse_args()
    print(args)
    main(args)
