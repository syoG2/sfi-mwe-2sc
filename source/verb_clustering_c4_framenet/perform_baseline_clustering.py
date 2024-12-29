import argparse
from pathlib import Path

import pandas as pd

from sfimwe2sc.f_induc.baseline_clustering import OnecpvClustering
from sfimwe2sc.utils.data_utils import read_jsonl, write_json, write_jsonl
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dev = pd.DataFrame(read_jsonl(args.input_dev_file))
    df_test = pd.DataFrame(read_jsonl(args.input_test_file))

    if args.clustering_name == "1cpv":
        clustering = OnecpvClustering()

    df_clu_dev = clustering.step(df_dev)
    df_clu_test = clustering.step(df_test)

    write_jsonl(
        df_clu_dev.to_dict("records"),
        args.output_dir / "exemplars_dev.jsonl",
    )
    write_jsonl(
        df_clu_test.to_dict("records"),
        args.output_dir / "exemplars_test.jsonl",
    )
    write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dev_file", type=Path, required=True)
    parser.add_argument("--input_test_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--clustering_name", type=str, choices=["1cpv"])
    args = parser.parse_args()
    print(args)
    main(args)
