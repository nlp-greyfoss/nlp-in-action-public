from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

from config import train_args


def get_tokenized_datasets(text_path: str, tokenizer: AutoTokenizer) -> Dataset:
    data_files = {"train": text_path}
    # load raw datasets
    raw_datasets = load_dataset("text", data_files=data_files, sample_by="document")

    max_seq_length = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns="text",
        desc="Running tokenizer on every text in dataset",
    )
    # just drop last example
    tokenized_datasets = tokenized_datasets.filter(
        lambda example: len(example["input_ids"]) == max_seq_length
    )

    tokenized_datasets = tokenized_datasets.remove_columns("overflow_to_sample_mapping")

    # split train and valid
    train_valid = tokenized_datasets["train"].train_test_split(test_size=0.05)
    tokenized_datasets = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": train_valid["test"],
        }
    )

    return tokenized_datasets


if __name__ == "__main__":
    if train_args.from_remote:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.model_name}"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(train_args.model_name)

    tokenized_datasets = get_tokenized_datasets(
        text_path="./data/novel.txt", tokenizer=tokenizer
    )

    print(tokenized_datasets)

    if train_args.from_remote:
        tokenized_datasets.push_to_hub(f"{train_args.owner}/{train_args.dataset_name}")
    else:
        tokenized_datasets.save_to_disk(train_args.dataset_name)
