import gzip
import json

import numpy as np


def read_txt(file):
    with open(file, "r") as f:
        return f.read()


def read_jsonl(file):
    with open(file, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(file, items):
    with open(file, "w") as f:
        for item in items:
            print(json.dumps(item, ensure_ascii=False), file=f)


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def write_json(file, item):
    with open(file, "w") as f:
        print(json.dumps(item, ensure_ascii=False), file=f)


def read_npz(file):
    return np.load(file)["item"]


def write_npz(file, item):
    np.savez_compressed(file, item=item)


def read_json_gz(file):
    with gzip.open(file, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_gz(file):
    with gzip.open(file, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
