import argparse
from pathlib import Path

from sfimwe2sc.f_induc.clustering_onestep import OnestepClustering
from sfimwe2sc.f_induc.clustering_twostep import TwostepClustering
from sfimwe2sc.f_induc.embedding import read_embedding
from sfimwe2sc.utils.data_utils import read_json, write_json, write_jsonl
from sfimwe2sc.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    clustering = OnestepClustering(args.clustering_method)

    df_vec, vec_array = read_embedding(
        args.input_dir, "dev", args.vec_type2layer, args.alpha
    )
    params = clustering.make_params(df_vec, vec_array)

    df_clu_dev = clustering.step(df_vec, vec_array, params)

    df_vec, vec_array = read_embedding(
        args.input_dir, "test", args.vec_type2layer, args.alpha
    )
    df_clu_test = clustering.step(df_vec, vec_array, params)
    write_jsonl(
        df_clu_dev.to_dict("records"),
        (args.output_dir / "exemplars_dev.jsonl"),
    )
    write_jsonl(
        df_clu_test.to_dict("records"),
        (args.output_dir / "exemplars_test.jsonl"),
    )
    write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--input_params_file", type=Path, required=False)

    parser.add_argument("--alpha", type=float, required=False)
    parser.add_argument("--layer", type=str, required=False)

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
