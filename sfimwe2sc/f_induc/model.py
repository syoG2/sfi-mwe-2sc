import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class BaseNet(nn.Module):
    def __init__(self, pretrained_model_name, normalization, device, layer=-1):
        super(BaseNet, self).__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        ).to(device)

        self.normalization = normalization
        self.layer = layer
        self.device = device

    def forward(self, inputs):
        if "token_type_ids" in inputs:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                token_type_ids=inputs["token_type_ids"].to(self.device),
            )["hidden_states"][self.layer]
        else:
            hidden_states = self.pretrained_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )["hidden_states"][self.layer]
        embeddings = hidden_states[
            torch.LongTensor(range(len(inputs["target_tidx"]))),
            inputs["target_tidx"],
        ]
        if self.normalization == "true":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
