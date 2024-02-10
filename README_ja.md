# Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering (sfi_mwe_2sc)

これは、ACL-IJCNLP 2021のMainで採択された[Semantic Frame Induction using Masked Word Embeddings and Two-Step Clustering](https://aclanthology.org/2021.acl-short.102/)のリポジトリです。

## インストール

以下のコマンドを実行することで、必要なパッケージをインストールすることができます。
```sh
# Before installation, upgrade pip and setuptools.
$ pip install -U pip setuptools

# Install other dependencies.
$ pip install -r requirements.txt
```

## 使用方法

**`source/`にあるソースコードを実行するためのスクリプトは全て`scripts/`に格納されていおり、スクリプトの各ファイルは`(directory name)/(file name).sh`という名称となっています。また、出力結果等は、初期設定では`data/`に格納されるようになっています。**

### 1. 前処理 (`preprocessing/`)

このディレクトリでは、データの前処理を行います。
まず、`make_exemplars.py`にて、NLTKライブラリにあるFrameNet 1.7から用例文を抽出します。
次に、`apply_stanza.py`にて、Stanzaと呼ばれるテキスト解析ツールを用いて、データの整形を行います。
主に文字レベルで付与されたラベルを単語レベルに変換しています。

### 2. 意味フレーム推定実験 (`verb_clustering/`)

このディレクトリでは、意味フレーム推定実験を行います。
まず、`make_dataset.py`にて、この実験に対応するようにデータセットを作成します。
初期設定として、いずれかのフレームにおいて20件以上の用例文が存在する動詞、および該当するLUの用例文を使用し、LUごとの最大用例数は無作為に選択された上限100件とするような論文通りの設定 (`20-100`)にしています。
ここで、開発セットとテストセットの分割も行っています。

次に、`get_embedding.py`にて、BERTによる動詞の埋め込みを獲得します。
その後、`find_best_params_*_clustering.py`を実行することで、1段階クラスタリングと2段階クラスタリングにおける最良パラメータを探索し、`perform_*_clustering.py`を実行することで、クラスタリングを実行します。

最後に`evaluate_clustering.py`を実行することで、クラスタリングした結果の評価を行います。


## 引用

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
