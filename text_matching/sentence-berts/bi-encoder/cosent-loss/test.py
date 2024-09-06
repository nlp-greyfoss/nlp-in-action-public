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
    queries1, queries2, labels = [], [], []

    df = build_dataframe_from_csv(dataset_csv)
    for _, row in df.iterrows():
        queries1.append(row["query1"])
        queries2.append(row["query2"])
        labels.append(int(row["label"]))

    return queries1, queries2, labels


def get_metrics(model: SentenceBert, dataset_csv, batch_size):
    queries1, queries2, labels = get_test_data(dataset_csv)

    start = time()
    embeddings1 = model.encode(queries1, batch_size=batch_size, show_progress_bar=True)
    embeddings2 = model.encode(queries2, batch_size=batch_size, show_progress_bar=True)
    end = time()

    scores = torch.cosine_similarity(embeddings1, embeddings2).cpu()
    similarities = scores.numpy().tolist()

    max_acc, best_threshold = find_best_acc_and_threshold(
        scores, np.asarray(labels), True
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

    model = SentenceBert(
        args.model_name_or_path,
        max_length=args.max_length,
    )
    model.eval()

    get_metrics(model, args.test_data_path, args.batch_size)
