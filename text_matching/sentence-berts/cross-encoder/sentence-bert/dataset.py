from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding

from datasets import Dataset as dt

from typing import Any

from utils import build_dataframe_from_csv


class PairDataset(Dataset):
    def __init__(
        self, data_path: str, tokenizer: PreTrainedTokenizer, max_len: int
    ) -> None:

        df = build_dataframe_from_csv(data_path)
        self.dataset = dt.from_pandas(df, split="train")

        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer

        self.max_len = max_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index) -> dict[str, Any]:
        query1 = self.dataset[index]["query1"]
        query2 = self.dataset[index]["query2"]
        label = self.dataset[index]["label"]

        encoding = self.tokenizer.encode_plus(
            query1,
            query2,
            truncation="only_second",
            max_length=self.max_len,
            padding=False,
        )

        encoding["label"] = label

        return encoding
