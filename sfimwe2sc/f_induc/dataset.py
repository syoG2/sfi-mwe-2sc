from tokenizations import get_alignments
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def tokenize_text_and_target(tokenizer, text_widx, target_widx, vec_type):
    inputs = tokenizer(text_widx)
    text_tidx = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

    alignments, previous_char_idx_list = [], [1]

    for char_idx_list in get_alignments(text_widx.split(), text_tidx[1:-1])[0]:
        # get_alignments()は単語列とトークン列の対応関係(現われる位置)を示す配列を返す
        # 文章："Dougal started with the body . "
        # トークン列：['doug', '##al', 'started', 'with', 'the', 'body', '.']
        # 出力：([[0, 1], [2], [3], [4], [5], [6]], [[0], [0], [1], [2], [3], [4], [5]])
        if len(char_idx_list) == 0:
            # 単語に対応するトークンがない場合、一つ前の単語に統合する
            alignments.append(previous_char_idx_list)
        else:
            # トークン列には先頭に[CLS]が追加されているため、indexを1増加させる
            char_idx_list = [c + 1 for c in char_idx_list]
            alignments.append(char_idx_list)
            previous_char_idx_list = char_idx_list

    target_tidx = alignments[target_widx][0]
    if vec_type == "mask":
        # 動詞をmaskに置き換える際、動詞が複数のトークンに分割されている可能性がある。
        # そのため動詞のトークンの先頭をmaskに置き換え、以降の動詞のトークンを削除する。
        inputs["input_ids"][target_tidx] = tokenizer.mask_token_id
        if len(alignments[target_widx]) >= 2:
            for _ in alignments[target_widx][1:]:
                for k in inputs.keys():
                    del inputs[k][target_tidx + 1]
    inputs["target_tidx"] = target_tidx
    return inputs


class BaseDataset(Dataset):
    def __init__(self, df, pretrained_model_name, vec_type):
        self.df = df
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.vec_type = vec_type
        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            inputs = tokenize_text_and_target(
                self.tokenizer,
                df_dict["text_widx"],
                df_dict["target_widx"],
                self.vec_type,
            )
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)
