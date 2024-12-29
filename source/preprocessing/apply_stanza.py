import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Union

import tokenizations
from stanza.pipeline.core import Pipeline
from tqdm import tqdm

from sfimwe2sc.utils.data_utils import read_jsonl, write_jsonl


def make_alignment(
    text: Union[str, List[str]], new_text: Union[str, List[str]]
) -> Dict[int, int]:
    alignment = {}
    alignments = tokenizations.get_alignments(text, new_text)
    for idx, new_idx in enumerate(alignments[0] + [[]]):
        if len(new_idx) != 0:
            alignment[idx] = new_idx[0]
        else:
            if idx - 1 in alignment:
                alignment[idx] = alignment[idx - 1]
            else:
                alignment[idx] = 0
    return alignment


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exemplars = read_jsonl(args.input_file)
    nlp = Pipeline("en", processors="tokenize")

    exemplar_list = []
    for exemplar in tqdm(exemplars):
        text = exemplar["text"]
        target = exemplar["target"]
        fe = exemplar["fe"]

        text_norm = " ".join(
            (re.sub("\s", " ", text).rstrip() + " ").split() + [""]
        )
        doc = nlp(text_norm)
        text_widx = " ".join(
            [w.text for s in doc.sentences for w in s.words] + [""]
        )

        a1 = make_alignment(text, text_norm)
        a2 = make_alignment(text_norm, text_widx)
        a3 = make_alignment(list(text_widx), text_widx.split())
        target_widx = [a3[a2[a1[t]]] for t in target[0]]
        fe_widx = [[a3[a2[a1[b]]], a3[a2[a1[e]]], f] for b, e, f in fe[0]]

        exemplar.update(
            {
                "text_widx": text_widx,
                "target_widx": target_widx,
                "fe_widx": fe_widx,
            }
        )
        exemplar_list.append(exemplar)

    write_jsonl(exemplar_list, args.output_dir / "exemplars.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
    # main(args)
