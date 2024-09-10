import torch
from transformers import HfArgumentParser

from dataclasses import dataclass, field

import numpy as np
from time import time


from modeling import SentenceBert
from utils import (
    build_dataframe_from_csv,
    compute_pearsonr,
    compute_spearmanr,
    compute_metrics,
    find_best_acc_and_threshold,
)


@dataclass
class TestArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    test_data_path: str = field(default=None, metadata={"help": "Path to test corpus"})
    max_length: int = field(default=64)
    batch_size: int = field(default=128)


def get_test_data(dataset_csv: str):
    texts, labels = [], []

    df = build_dataframe_from_csv(dataset_csv)
    for _, row in df.iterrows():
        texts.append((row["query1"], row["query2"]))
        labels.append(int(row["label"]))

    return texts, labels


def get_metrics(model: SentenceBert, dataset_csv, batch_size):
    texts, labels = get_test_data(dataset_csv)

    start = time()
    scores = model.predict(texts, batch_size, show_progress_bar=True).cpu().squeeze()
    end = time()

    similarities = scores.numpy().tolist()

    max_acc, best_threshold = find_best_acc_and_threshold(
        similarities, np.asarray(labels), True
    )
    print(f"max_acc: {max_acc:.4f}, best_threshold: {best_threshold:.6f}")

    preds = (scores > best_threshold).int()

    spearman_corr = compute_spearmanr(similarities, labels)
    pearson_corr = compute_pearsonr(similarities, labels)

    accuracy, precision, recal, f1 = compute_metrics(preds, torch.LongTensor(labels))
    print(
        f"spearman corr: {spearman_corr:.4f} |  pearson_corr corr: {pearson_corr:.4f} | compute time: {end-start:.2f}s\naccuracy={accuracy:.3f} precision={precision:.3f} recal={recal:.3f} f1 score={f1:.4f}"
    )


if __name__ == "__main__":
    parser = HfArgumentParser(TestArguments)
    args, *_ = parser.parse_args_into_dataclasses()
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceBert(
        args.model_name_or_path,
        max_length=args.max_length,
    )
    model.eval()
    model = model.to(device)

    get_metrics(model, args.test_data_path, args.batch_size)
