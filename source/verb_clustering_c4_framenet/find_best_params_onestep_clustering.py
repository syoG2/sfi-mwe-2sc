import argparse
from pathlib import Path

from tqdm import tqdm

from sfimwe2sc.f_induc.clustering_onestep import OnestepClustering
from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.modules.score_clustering import calculate_bcubed
from sfimwe2sc.utils.data_utils import write_json
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vec_type == "word":
        vec_types = ["word"]
        alphas = [0]
    elif args.vec_type == "mask":
        vec_types = ["mask"]
        alphas = [1]
    elif args.vec_type == "wm":
        vec_types = ["word", "mask"]
        alphas = [i / 10 for i in range(11)]

    clustering = OnestepClustering(args.clustering_method)

    best_vec_type2layer = {}
    for vec_type in tqdm(vec_types):
        best_bcf = 0
        for layer in tqdm(args.layers):
            vec_type2layer = {vec_type: layer}
            alpha = 0 if vec_type == "word" else 1
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", vec_type2layer, alpha
            )

            params = clustering.make_params(df_vec, vec_array)
            df_output = clustering.step(df_vec, vec_array, params)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_vec_type2layer[vec_type] = layer

    if args.vec_type == "wm":
        best_bcf = 0
        for alpha in tqdm(alphas):
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", best_vec_type2layer, alpha
            )

            params = clustering.make_params(df_vec, vec_array)
            df_output = clustering.step(df_vec, vec_array, params)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_alpha = alpha
    else:
        best_alpha = alphas[0]

    best_params = {
        "alpha": best_alpha,
        "vec_type2layer": best_vec_type2layer,
    }
    best_params["clustering_method"] = args.clustering_method
    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vec_type", type=str, choices=["word", "mask", "wm"])
    parser.add_argument("--layers", type=str, nargs="*")

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
