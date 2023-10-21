import os
import random
from argparse import ArgumentParser, Namespace

import pandas as pd


def main(args: Namespace) -> None:
    output_dir = args.output_dir + "/"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_json(args.input_file, orient="records", lines=True)
    df = df.sample(frac=1, random_state=0)
    df = df.reset_index(drop=True).reset_index()
    df = df[df["lu_name"].apply(lambda x: x.split(".")[-1]) == "v"].reset_index(
        drop=True
    )

    df["target_widx"] = df["target_widx"].apply(lambda x: x[0])
    df["frame"] = df["frame_name"]
    df.loc[:, "source"] = "framenet"

    df = df[
        [
            "ex_id",
            "ex_idx",
            "lu_id",
            "lu_name",
            "frame",
            "text_widx",
            "target_widx",
        ]
    ]

    df_agg = pd.DataFrame(df.agg("lu_id").value_counts())
    ex_idx_list = []
    for lu_id in df_agg[df_agg["lu_id"] >= args.min_examples].index.to_list():
        df_vf = df[df["lu_id"] == lu_id]
        ex_idx_list += df_vf["ex_idx"].to_list()[: args.max_examples]
    df_used = df[df["ex_idx"].isin(ex_idx_list)]

    all_list, v2_list = [], []
    for lu_name, _ in df_used.groupby(["lu_name", "lu_id"]).count().index:
        if lu_name not in all_list:
            all_list.append(lu_name)
        else:
            if lu_name not in v2_list:
                v2_list.append(lu_name)
    v1_list = sorted(set(all_list) - set(v2_list))

    random.seed(0)
    random.shuffle(v1_list)
    random.shuffle(v2_list)
    dev_list = sorted(
        v1_list[: -int(len(v1_list) * 0.8)] + v2_list[: -int(len(v2_list) * 0.8)]
    )
    test_list = sorted(
        v1_list[-int(len(v1_list) * 0.8) :] + v2_list[-int(len(v2_list) * 0.8) :]
    )

    col = str(args.min_examples) + "-" + str(args.max_examples)
    df[col] = "nouse"
    for sets, lu_name_list in zip(["dev", "test"], [dev_list, test_list]):
        for lu_name in lu_name_list:
            df_lu_name = df_used[df_used["lu_name"] == lu_name]
            df.loc[list(df_lu_name.index), col] = sets

    df = df.sort_values("ex_idx")
    df.to_json(
        output_dir + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--min_examples", type=int, default=20)
    parser.add_argument("--max_examples", type=int, default=100)

    args = parser.parse_args()
    print(args)
    main(args)
