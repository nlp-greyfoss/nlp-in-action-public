from torch.utils.data import Dataset
from datasets import Dataset as dt
import pandas as pd

from utils import build_dataframe_from_csv


class PairDataset(Dataset):
    def __init__(self, data_path: str) -> None:

        df = build_dataframe_from_csv(data_path)
        self.dataset = dt.from_pandas(df, split="train")

        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index) -> dict[str, str]:
        query1 = self.dataset[index]["query1"]
        query2 = self.dataset[index]["query2"]
        label = self.dataset[index]["label"]
        return {"query1": query1, "query2": query2, "label": label}


class PairCollator:
    def __call__(self, features) -> dict[str, list[str]]:
        queries1 = []
        queries2 = []
        labels = []

        for feature in features:
            queries1.append(feature["query1"])
            queries2.append(feature["query2"])
            labels.append(feature["label"])

        return {"source": queries1, "target": queries2, "labels": labels}
