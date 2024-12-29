import argparse
from pathlib import Path

from nltk.corpus import framenet
from tqdm import tqdm

from lib.utils.data_utils import write_jsonl


def collect_exemplars(framenet_exemplars):
    ex_idx, exemplars = 0, []
    for exemplar in tqdm(framenet_exemplars):
        try:
            exemplar_dict = {}
            exemplar_dict["ex_idx"] = ex_idx
            exemplar_dict["ex_id"] = exemplar.ID
            exemplar_dict["frame_name"] = exemplar.frame.name
            exemplar_dict["frame_id"] = exemplar.frame.ID
            exemplar_dict["lu_name"] = exemplar.LU.name
            exemplar_dict["lu_id"] = exemplar.LU.ID
            exemplar_dict["target"] = exemplar.Target
            exemplar_dict["fe"] = exemplar.FE
            exemplar_dict["text"] = exemplar.text
            exemplars.append(exemplar_dict)
            ex_idx += 1
        except:
            continue
    return exemplars


def main(args):
    args.output_dir_path.mkdir(parents=True, exist_ok=True)

    exemplars = collect_exemplars(framenet.exemplars())
    write_jsonl(args.output_dir_path / "exemplars.jsonl", exemplars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
