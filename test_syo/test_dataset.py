from tokenizations import get_alignments
from torch.utils.data import Dataset
from transformers import AutoTokenizer

pretrained_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
text_widx = "Dougal started with the body . "

inputs = tokenizer(text_widx)
print("inputs")
print(inputs)

text_tidx = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print("text_tidx")
print(text_tidx)

print("get_alignments()")
print(get_alignments(text_widx.split(), text_tidx[1:-1]))

