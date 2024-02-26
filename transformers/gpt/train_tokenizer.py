from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

from config import train_args


def train(
    file_path: str,
    save_path="tokenizer.json",
    eos_token="<|endoftext|>",
    vocab_size: int = 5000,
) -> None:
    tokenizer = Tokenizer(BPE(unk_token=eos_token))

    # only has eos token
    trainer = BpeTrainer(special_tokens=[eos_token], vocab_size=vocab_size)

    tokenizer.pre_tokenizer = BertPreTokenizer()

    tokenizer.train([file_path], trainer)

    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos_token}",
        pair=f"$A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[
            (eos_token, tokenizer.token_to_id(eos_token)),
        ],
    )

    print(f"vocab size: {tokenizer.get_vocab_size()}")

    tokenizer.save(save_path)


if __name__ == "__main__":
    eos_token = "<|endoftext|>"

    train("./data/novel.txt", eos_token=eos_token)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer.json", model_max_length=512
    )

    tokenizer.unk_token = eos_token

    tokenizer.bos_token = tokenizer.unk_token
    tokenizer.eos_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.unk_token

    if train_args.from_remote:
        tokenizer.push_to_hub(f"{train_args.owner}/{train_args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.model_name}"
        )

    else:
        tokenizer.save_pretrained(train_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(train_args.model_name)

    encodes = tokenizer("三十年河东三十年河西，莫欺少年穷！", "突破斗者！")

    print(encodes)

    print(tokenizer.convert_ids_to_tokens(encodes["input_ids"]))
