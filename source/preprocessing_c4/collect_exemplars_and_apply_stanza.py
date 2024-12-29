import argparse
import re
import unicodedata

import pandas as pd
import stanza
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


def clean_text(text):
    text = re.sub(r"\s", " ", text)
    text = "".join(
        [
            char
            for char in text
            if unicodedata.category(char)
            not in ["Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp", "So"]
        ]
    )
    return text


def make_word_list(doc):
    word_list, count, word_count = [], 0, 0
    for sentence_id, sentence in enumerate(doc.sentences):
        child = {}
        for word in sentence.words:
            if word.head not in child:
                child[word.head] = [word.id]
            else:
                child[word.head].append(word.id)

        for word in sentence.words:
            word_dict = word.to_dict()
            word_dict.update(
                {
                    "id": count,
                    "sent_id": sentence_id,
                    "word_id": int(word.id) - 1,
                }
            )
            if word.head != 0:
                word_dict.update(
                    {
                        "head": word.head - 1 + word_count,
                        "head_text": sentence.words[word.head - 1].text,
                    }
                )
            else:
                word_dict.update({"head": -1, "head_text": "[ROOT]"})
            if word.id in child:
                word_dict.update(
                    {
                        "children": [i - 1 + word_count for i in child[word.id]],
                    }
                )
            else:
                word_dict.update({"children": []})
            word_list.append(word_dict)
            count += 1
        word_count += len(sentence.words)
    return word_list


def main(args):
    output_dir_path = args.output_dir_path / f"{args.split_name}-{args.file_id}"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        pos_batch_size=3000,
    )
    dataset = load_dataset(
        "json",
        data_files=(
            "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
            f"c4-{args.split_name}.{args.file_id:05}-of-01024.json.gz"
        ),
    )
    text_list = dataset[args.split_name]["text"]

    for part_id in tqdm(range(args.part_ids[0], args.part_ids[1] + 1)):
        output_list, output_word_list, exemplar_id = [], [], 0
        for doc_id in tqdm(range(part_id * 1000, (part_id + 1) * 1000)):
            if doc_id >= len(text_list):
                continue
            text = text_list[doc_id]
            cleaned_text = clean_text(text)
            doc = nlp(cleaned_text)
            word_list = make_word_list(doc)

            df_wl = pd.DataFrame(word_list)
            for sent_id in sorted(set(df_wl["sent_id"])):
                df_sent = df_wl[df_wl["sent_id"] == sent_id]
                for word_dict in df_sent.to_dict("records"):
                    if word_dict["upos"] == "VERB":
                        output_list.append(
                            {
                                "split_name": args.split_name,
                                "file_id": args.file_id,
                                "part_id": part_id,
                                "exemplar_id": exemplar_id,
                                "doc_id": doc_id,
                                "sent_id": sent_id,
                                "word_id": word_dict["word_id"],
                                "verb": word_dict["lemma"],
                                "word": word_dict["text"],
                                "text_widx": " ".join(df_sent["text"]),
                            }
                        )
                        output_word_list.append(
                            {
                                "exemplar_id": exemplar_id,
                                "word_list": list(df_sent.to_dict("records")),
                            }
                        )
                        exemplar_id += 1

        if len(output_list) != 0:
            df_output = pd.DataFrame(output_list)
            df_output.to_json(
                output_dir_path / f"exemplars_{part_id:04}.jsonl",
                orient="records",
                force_ascii=False,
                lines=True,
            )
            df_owl = pd.DataFrame(output_word_list)
            df_owl.to_json(
                output_dir_path / f"word_list_{part_id:04}.jsonl",
                orient="records",
                force_ascii=False,
                lines=True,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", type=Path, required=True)
    parser.add_argument("--split_name", type=str, choices=["train", "validation"])
    parser.add_argument("--file_id", type=int, default=0)
    parser.add_argument("--part_ids", type=int, nargs="*", default=[0, 356])
    args = parser.parse_args()
    print(args)
    main(args)
