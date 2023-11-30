import os
import torch
import pandas as pd
import numpy as np
from typing import Tuple

import jieba


class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        self.patience = patience
        self.counter = 0
        self.best_value = 0.0
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        self.mode = mode

    def step(self, value: float) -> bool:
        if self.is_better(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False

    def is_better(self, a: float) -> bool:
        if self.mode == "min":
            return a < self.best_value
        return a > self.best_value


def make_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def tokenize(sentence: str):
    return list(jieba.cut(sentence))


def build_dataframe_from_csv(dataset_csv: str) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_csv,
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
    )

    df.sentence1 = df.sentence1.apply(tokenize)
    df.sentence2 = df.sentence2.apply(tokenize)

    return df


def metrics(y: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float, float]:
    TP = ((y_pred == 1) & (y == 1)).sum().float()  # True Positive
    TN = ((y_pred == 0) & (y == 0)).sum().float()  # True Negative
    FN = ((y_pred == 0) & (y == 1)).sum().float()  # False Negatvie
    FP = ((y_pred == 1) & (y == 0)).sum().float()  # False Positive
    p = TP / (TP + FP).clamp(min=1e-8)  # Precision
    r = TP / (TP + FN).clamp(min=1e-8)  # Recall
    F1 = 2 * r * p / (r + p).clamp(min=1e-8)  # F1 score
    acc = (TP + TN) / (TP + TN + FP + FN).clamp(min=1e-8)  # Accurary
    return acc, p, r, F1


def load_embedings(vocab, word2vec, embedding_dim=300, lower=True):
    embedding = np.random.randn(len(vocab), embedding_dim)
    load_count = 0
    for i, word in vocab:
        if lower:
            word = word.lower()
        if word in word2vec:
            embedding[i] = word2vec[word]
            load_count += 1
    print(f"loaded word count: {load_count}")
    return embedding.tolist()
