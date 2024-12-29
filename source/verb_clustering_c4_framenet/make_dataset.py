import json
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from transformers import BertTokenizer

from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_framenet = pd.DataFrame(read_jsonl(args.input_file_framenet))
    ex_idx_max = df_framenet["ex_idx"].max()
    df_framenet = df_framenet.sample(frac=1, random_state=args.random_state)
    df_framenet = df_framenet.reset_index(drop=True).reset_index()
    df_framenet = df_framenet[
        df_framenet["lu_name"].apply(lambda x: x.split(".")[-1]) == "v"
    ]
    df_framenet = df_framenet.reset_index(drop=True)

    df_framenet["target_widx"] = df_framenet["target_widx"].apply(lambda x: x[0])
    df_framenet["frame"] = df_framenet["frame_name"]
    df_framenet["verb"] = df_framenet["lu_name"].apply(lambda x: x.split(".v")[0])
    df_framenet["verb_frame"] = df_framenet["verb"].str.cat(
        df_framenet["frame"], sep=":"
    )
    df_framenet.loc[:, "source"] = "framenet"
    df_framenet["id_data"] = df_framenet.apply(
        lambda row: json.dumps({"ex_id": row["ex_id"], "ex_idx": row["ex_idx"]}), axis=1
    )

    df_framenet = df_framenet[
        [
            "ex_idx",
            "id_data",
            "verb",
            "frame",
            "verb_frame",
            "lu_id",
            "lu_name",
            "text_widx",
            "target_widx",
            "source",
        ]
    ]

    df_framenet_agg = pd.DataFrame(df_framenet.agg("lu_id").value_counts())
    ex_idx_list = []
    for lu_id in df_framenet_agg[
        df_framenet_agg["count"] >= args.min_n_examples
    ].index.to_list():
        df_vf = df_framenet[df_framenet["lu_id"] == lu_id]
        ex_idx_list += df_vf["ex_idx"].to_list()[: args.max_n_examples]
    df_ex = df_framenet[df_framenet["ex_idx"].isin(ex_idx_list)]

    all_list, v2_list = [], []
    for lu_name, _ in df_ex.groupby(["lu_name", "lu_id"]).count().index:
        if lu_name not in all_list:
            all_list.append(lu_name)
        else:
            if lu_name not in v2_list:
                v2_list.append(lu_name)
    v1_list = sorted(set(all_list) - set(v2_list))

    random.seed(args.random_state)
    random.shuffle(v1_list)
    random.shuffle(v2_list)
    dev_list = sorted(
        v1_list[: -int(len(v1_list) * 0.8)] + v2_list[: -int(len(v2_list) * 0.8)]
    )
    test_list = sorted(
        v1_list[-int(len(v1_list) * 0.8) :] + v2_list[-int(len(v2_list) * 0.8) :]
    )

    df_c4 = pd.DataFrame(read_jsonl(args.input_file_c4))
    for i in range(1, 20):
        part_id = f"{i:04}"
        c4_filename = f"exemplars_{part_id}.jsonl"
        c4_dir = args.input_file_c4.parent
        df_c4 = pd.concat(
            [df_c4, pd.DataFrame(read_jsonl(c4_dir.joinpath(c4_filename)))]
        )

    df_c4.loc[:, "source"] = "c4"
    df_c4["id_data"] = df_c4.apply(
        lambda row: json.dumps(
            {
                "split_name": row["split_name"],
                "file_id": row["file_id"],
                "part_id": row["part_id"],
                "exemplar_id": row["exemplar_id"],
                "doc_id": row["doc_id"],
                "sent_id": row["sent_id"],
            }
        ),
        axis=1,
    )

    df_c4["ex_idx"] = range(1 + ex_idx_max, len(df_c4) + 1 + ex_idx_max)
    df_c4["frame"] = "None"
    df_c4["verb_frame"] = df_c4["verb"].str.cat(df_c4["frame"], sep=":")
    df_c4["lu_id"] = -1
    df_c4["lu_name"] = "None"
    df_c4["text_widx"] = df_c4["text_widx"]
    df_c4["target_widx"] = df_c4["word_id"]
    df_c4 = df_c4[
        [
            "ex_idx",
            "id_data",
            "verb",
            "frame",
            "verb_frame",
            "lu_id",
            "lu_name",
            "text_widx",
            "target_widx",
            "source",
        ]
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for split, verb_list in zip(["dev", "test"], [dev_list, test_list]):
        df_verb = df_ex[df_ex["lu_name"].isin(verb_list)].copy()
        df_verb_c4 = df_c4[df_c4["verb"].isin(df_verb["verb"])].copy()
        df_verb_c4 = df_verb_c4[: len(df_verb) * 2]
        df_verb = pd.concat([df_verb, df_verb_c4], ignore_index=True)

        df_verb = df_verb[
            df_verb["text_widx"].apply(lambda x: len(tokenizer.tokenize(x))) <= 512
        ]
        # df_verb_tmp = pd.DataFrame()
        # for record in df_verb:
        #     print(record["text_widx"])
        #     # if len(tokenizer.tokenize(record["text_widx"])) <= 512:
        #     #     df_verb_tmp = df_verb_tmp.append(record)

        # df_verb = df_verb_tmp
        # for verb, size in df_verb.groupby("verb").size().items():
        #     extra_data = df_c4[df_c4["verb"] == verb]
        #     # additional_rows = extra_data.head(args.max_n_examples - size)
        #     # df_verb = pd.concat([df_verb,additional_rows], ignore_index=True)
        #     df_verb = pd.concat([df_verb, extra_data], ignore_index=True)

        df_verb = df_verb.sort_values("ex_idx")
        write_jsonl(
            df_verb.to_dict("records"),
            args.output_dir / f"exemplars_{split}.jsonl",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file_c4", type=Path, required=True)
    parser.add_argument("--input_file_framenet", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--min_n_examples", type=int, default=20)
    parser.add_argument("--max_n_examples", type=int, default=100)

    args = parser.parse_args()
    print(args)
    main(args)
