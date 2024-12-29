from transformers import AutoConfig, AutoModel

pretrained_model_name = "bert-base-uncased"

config = AutoConfig.from_pretrained(
    pretrained_model_name, output_hidden_states=True
)
print(config)

pretrained_model = AutoModel.from_pretrained(
    pretrained_model_name, config=config
)