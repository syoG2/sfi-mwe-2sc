import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.utils.data_utils import read_json, read_jsonl


def main(args):
    if args.input_params_file is not None:
        best_params = read_json(args.input_params_file)
        for key, value in best_params.items():
            setattr(args, key, value)
    else:
        if args.alpha == 0:
            args.vec_type2layer = {"word": args.layer}
        elif args.alpha == 1:
            args.vec_type2layer = {"mask": args.layer}
        else:
            layer_word, layer_mask = args.layer.split("-")
            args.vec_type2layer = {"word": layer_word, "mask": layer_mask}

    df_vec, vec_array = read_embedding(
        args.input_dir, "dev", args.vec_type2layer, args.alpha
    )
    print(type(vec_array))
    df = pd.DataFrame(
        read_jsonl(
            "../data/verb_clustering/clustering/20-100/bert-base-uncased/word/twostep-xmeans-ward/exemplars_dev.jsonl"
        )
    )

    # print(df.groupby("verb").count())
    # for verb,size in df.groupby("verb").size().items():
    #     print(verb,size)
    # return 0
    # print("df_vec:")
    # print(df_vec)
    # print("df:")
    # print(df)
    # print("vec_array:")
    # print(vec_array)

    n_components = 2
    perplexity = 30

    y = df["frame"]
    y_items = y.unique()

    fig, ax = plt.subplots(figsize=(20, 20))
    tsne = TSNE(
        n_components=n_components, init="random", random_state=0, perplexity=perplexity
    )
    Y = tsne.fit_transform(vec_array)

    for each_cluster in y_items:
        c_plot_bool = y == each_cluster  # True/Falseのarrayを返す
        ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1])
    ax.legend()
    plt.savefig("test.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=False,
        default="../data/verb_clustering/embedding/20-100/bert-base-uncased",
    )
    parser.add_argument("--output_dir", type=Path, required=False)

    parser.add_argument(
        "--input_params_file",
        type=Path,
        required=False,
        default="../data/verb_clustering/best_params_twostep_clustering/20-100/bert-base-uncased/word/twostep-xmeans-ward/best_params.json",
    )

    parser.add_argument("--alpha", type=float, required=False, default=0)
    parser.add_argument("--layer", type=str, required=False)

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
        default="average",
    )
    args = parser.parse_args()
    print(args)
    main(args)
