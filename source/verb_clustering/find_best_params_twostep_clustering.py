import argparse
from pathlib import Path

from tqdm import tqdm

from sfimwe2sc.f_induc.clustering_twostep import TwostepClustering
from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.modules.score_clustering import calculate_bcubed
from sfimwe2sc.utils.data_utils import write_json
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clustering = TwostepClustering(args.clustering_method1, args.clustering_method2)

    if args.vec_type == "word":
        vec_types = ["word"]
    elif args.vec_type == "mask":
        vec_types = ["mask"]
    else:
        vec_types = ["word", "mask"]

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
            df_output = clustering.step(df_vec, vec_array, vec_array, params)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_vec_type2layer[vec_type] = layer

    if "wm" not in args.vec_type.split("-"):
        if args.vec_type in ["word", "mask"]:
            args.vec_type = f"{args.vec_type}-{args.vec_type}"
        for i, vec_type in enumerate(args.vec_type.split("-")):
            alpha = 0 if vec_type == "word" else 1
            if i == 0:
                best_alpha1 = alpha
            else:
                best_alpha2 = alpha
    else:
        if args.vec_type in ["wm"]:
            best_bcf, best_alpha1, best_alpha2 = 0, -1, -1
            for alpha in tqdm([i / 10 for i in range(11)]):
                df_vec, vec_array = read_embedding(
                    args.input_dir, "dev", best_vec_type2layer, alpha
                )
                params = clustering.make_params(df_vec, vec_array)
                df_output = clustering.step(df_vec, vec_array, vec_array, params)

                true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
                pred = df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
                bcf = calculate_bcubed(true, pred)[2]
                if best_bcf < bcf:
                    best_bcf = bcf
                    best_alpha1, best_alpha2 = alpha, alpha
        else:
            for i, vec_type in enumerate(args.vec_type.split("-")):
                if vec_type == "word":
                    alphas = [0]
                elif vec_type == "mask":
                    alphas = [1]
                elif vec_type == "wm":
                    alphas = [i / 10 for i in range(11)]

                if i == 0:
                    alphas1 = alphas
                else:
                    alphas2 = alphas

            best_bcf, best_alpha1, best_alpha2 = 0, -1, -1
            for alpha1 in tqdm(alphas1):
                df_vec, vec_array1 = read_embedding(
                    args.input_dir, "dev", best_vec_type2layer, alpha1
                )
                params = clustering.make_params(df_vec, vec_array1)
                for alpha2 in tqdm(alphas2):
                    _, vec_array2 = read_embedding(
                        args.input_dir, "dev", best_vec_type2layer, alpha2
                    )
                    df_output = clustering.step(df_vec, vec_array1, vec_array2, params)

                    true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
                    pred = (
                        df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
                    )
                    bcf = calculate_bcubed(true, pred)[2]
                    if best_bcf < bcf:
                        best_bcf = bcf
                        best_alpha1 = alpha1
                        best_alpha2 = alpha2

    best_params = {
        "alpha1": best_alpha1,
        "alpha2": best_alpha2,
        "vec_type2layer": best_vec_type2layer,
    }
    best_params["clustering_method1"] = args.clustering_method1
    best_params["clustering_method2"] = args.clustering_method2
    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--vec_type",
        type=str,
        choices=[
            "word",
            "mask",
            "wm",
            "word-mask",
            "mask-word",
            "word-wm",
            "mask-wm",
            "word-wm",
            "mask-wm",
            "wm-wm",
        ],
        required=False,
    )

    parser.add_argument("--layers", type=str, nargs="*")

    parser.add_argument(
        "--clustering_method1",
        type=str,
        choices=["xmeans", "average", "1cpv"],
        required=False,
    )
    parser.add_argument(
        "--clustering_method2",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
