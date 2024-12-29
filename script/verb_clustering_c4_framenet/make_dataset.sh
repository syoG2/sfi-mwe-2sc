#!/bin/bash

# c4とframenetのデータを混ぜた用例集を作成する

source_dir=../../source/verb_clustering_c4_framenet     # pythonコードが格納されているディレクトリのパス
data_dir=../../data/verb_clustering_c4_framenet         # 出力データを格納するディレクトリのパス
input_dir_c4=../../data/preprocessing_c4                # c4の用例が格納されたディレクトリのパス
input_dir_framenet=../../data/preprocessing_framenet    # framenetの用例が格納されたディレクトリのパス

# c4のデータ分類(trainデータかvalidationデータか)
split=train
# split=validation

file_id=0   # c4のファイルid

# c4のどのpartを取得するか(en: 0~356)
part_id=0
part_id=$(printf "%04d" "${part_id}") # 4桁へ0パディング

min_n_examples=20   # 用例数の最小値の制限
max_n_examples=100  # 用例数の最大値の制限
setting=${min_n_examples}-${max_n_examples}

d1=${setting}
python ${source_dir}/make_dataset.py \
    --input_file_c4 ${input_dir_c4}/collect_exemplars_and_apply_stanza/${split}-${file_id}/exemplars_"${part_id}".jsonl \
    --input_file_framenet ${input_dir_framenet}/apply_stanza/exemplars.jsonl \
    --output_dir ${data_dir}/dataset/${d1} \
    --min_n_examples ${min_n_examples} \
    --max_n_examples ${max_n_examples} \
    --random_state 0
