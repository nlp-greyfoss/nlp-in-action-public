from transformers import AutoTokenizer

from modeling_gpt2 import GPT2LMHeadModel


tokenizer = AutoTokenizer.from_pretrained("simple-gpt2-doupo")

model = GPT2LMHeadModel.from_pretrained(
    "simple-gpt2-doupo", pad_token_id=tokenizer.unk_token_id
)


prefix = "萧炎经过不懈地修炼，终于达到了斗帝级别，"
input_ids = tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False)

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
