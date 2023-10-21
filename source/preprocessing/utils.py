import gzip
import json
from typing import Any, Iterable


def write_json(items: Iterable[Any], file: str) -> None:
    if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
        with gzip.open(file, "wt") if file.endswith(".gz") else open(file, "w") as fo:
            for item in items:
                print(json.dumps(item, ensure_ascii=False), file=fo)
    else:
        with gzip.open(file, "wt") if file.endswith(".gz") else open(file, "w") as fo:
            json.dump(items, fo, ensure_ascii=False, indent=2)


def read_json(file: str) -> Iterable:
    if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") if file.endswith(".gz") else open(file) as f:
            for line in f:
                yield json.loads(line)
    else:
        with gzip.open(file, "rt") if file.endswith(".gz") else open(file) as f:
            for item in json.load(f):
                yield item
