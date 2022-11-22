from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

from nltk.corpus import framenet as fn
from tqdm import tqdm
from utils import write_json


def make_exemplar_list(exemplars: List[Any]) -> List[Dict[str, Any]]:
    ex_idx, exemplar_list = 0, []
    for exemplar in tqdm(exemplars):
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
            exemplar_list.append(exemplar_dict)
            ex_idx += 1

        except:
            continue
    return exemplar_list


def main(args: Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exemplar_list = make_exemplar_list(fn.exemplars())
    write_json(exemplar_list, str(output_dir / "exemplars.jsonl"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
