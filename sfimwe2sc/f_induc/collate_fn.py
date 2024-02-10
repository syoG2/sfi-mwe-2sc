import torch
import torch.nn as nn


def base_collate_fn(batch):
    output_dict = {"verb": [], "frame": [], "ex_idx": [], "batch_size": len(batch)}
    for i in ["input_ids", "token_type_ids", "attention_mask"]:
        if i in batch[0]:
            output_dict[i] = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(b[i]) for b in batch], batch_first=True
            )
    output_dict["target_tidx"] = torch.LongTensor([b["target_tidx"] for b in batch])

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict
