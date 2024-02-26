from transformers import AutoTokenizer

import torch

from modeling_gpt import GPTLMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("greyfoss/simple-gpt-doupo")

model = GPTLMHeadModel.from_pretrained("greyfoss/simple-gpt-doupo").to(device)


prefix = "萧炎"
input_ids = tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False).to(
    device
)
beam_output = model.generate(
    input_ids,
    max_length=512,
    num_beams=3,
    no_repeat_ngram_size=2,
    early_stopping=True,
    do_sample=True,
    repetition_penalty=1.25,
)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(beam_output[0], skip_special_tokens=True).replace(" ", ""))
