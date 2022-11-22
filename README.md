# Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering (sfi_mwe_2sc)

The source code of paper: [Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering](https://aclanthology.org/2021.acl-short.102/), accepted to ACL-IJCNLP 2021.

## Installation

```sh
# Before installation, upgrade pip and setuptools.
$ pip install -U pip setuptools

# Install other dependencies.
$ pip install -r requirements.txt
```

## Usage

**All scripts to run the source codes are in `scripts/`.**
**The file names of the scripts are `(directory name)/(file name).sh`, respectively.**

### 1. Preprocessing (`preprocessing/`)

This extracts examples from the FrameNet 1.7 dataset contained in NLTK (`make_exemplars.py`) and creates word-by-word indexes of targets using stanza (`apply_stanza.py`).

### 2. Semantic Frame Induction (`semantic_frame_induction/`)

This is a semantic frame induction experiment.
All target verbs in the dataset have at least 20 example sentences for each frame they evoke, and we limited the maximum number of sentence examples for each LU to 100 and if there were more examples, we randomly selected 100 (`make_dataset.py`).
In this dataset, `20-100` indicates a development, test, or unused set.

## Citation

Please cite our paper if this source code is helpful in your work.

```bibtex
@inproceedings{yamada-etal-2021-semantic,
    title = "Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering",
    author = "Yamada, Kosuke  and
      Sasano, Ryohei  and
      Takeda, Koichi",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-short.102",
    pages = "811--816",
}
```
