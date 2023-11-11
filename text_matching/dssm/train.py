from argparse import Namespace
from datetime import datetime
import os
import torch
import pandas as pd
from typing import Tuple

import jieba

from dataset import TMVectorizer, TMDataset, Vocabulary
from tqdm import tqdm
from model import DSSM

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


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


def evaluate(data_iter: DataLoader, model: DSSM) -> Tuple[float, float, float, float]:
    y_list, y_pred_list = [], []
    model.eval()
    for x1, x2, y in tqdm(data_iter):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        y = torch.LongTensor(y).to(device)

        similarity = model(x1, x2)
        disparity = 1 - similarity

        output = torch.stack([disparity, similarity], 1).to(device)

        pred = torch.max(output, 1)[1]

        y_pred_list.append(pred)
        y_list.append(y)

    y_pred = torch.cat(y_pred_list, 0)
    y = torch.cat(y_list, 0)
    acc, p, r, f1 = metrics(y, y_pred)
    return acc, p, r, f1


def train(
    data_iter: DataLoader,
    model: DSSM,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    print_every: int = 500,
    verbose=True,
) -> None:
    model.train()

    for step, (x1, x2, y) in enumerate(tqdm(data_iter)):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        y = torch.LongTensor(y).to(device)
        # the similarity between x1 and x2
        similarity = model(x1, x2)
        # the disparity between x1 and x2
        disparity = 1 - similarity
        # CrossEntropyLoss requires two class result
        output = torch.stack([disparity, similarity], 1).to(device)
        # output (batch_size, num_classes=2)
        # y (batch_size, )
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % print_every == 0:
            # `torch.max` Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim.
            # And indices is the index location of each maximum value found (argmax).
            # get the indices
            pred = torch.max(output, 1)[1]
            acc, p, r, f1 = metrics(y, pred)

            print(
                f" TRAIN iter={step+1} loss={loss.item():.6f} accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
            )


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="text_matching/data/lcqmc/{}.txt",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir=f"text_matching/dssm/model_storage",
        reload_model=False,
        cuda=True,
        learning_rate=5e-4,
        batch_size=128,
        num_epochs=10,
        max_len=50,
        embedding_dim=512,
        activation="relu",
        dropout=0.1,
        internal_hidden_sizes=[256, 256, 128],
        min_freq=2,
        print_every=500,
        verbose=True,
    )

    make_dirs(args.save_dir)

    print(f"Arguments : {args}")

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}.")

    vectorizer_path = os.path.join(args.save_dir, args.vectorizer_file)

    train_df = build_dataframe_from_csv(args.dataset_csv.format("train"))
    test_df = build_dataframe_from_csv(args.dataset_csv.format("test"))
    dev_df = build_dataframe_from_csv(args.dataset_csv.format("dev"))

    if os.path.exists(vectorizer_path):
        print("Loading vectorizer file.")
        vectorizer = TMVectorizer.load_vectorizer(vectorizer_path)
    else:
        print("Creating a new Vectorizer.")

        train_sentences = train_df.sentence1.to_list() + train_df.sentence2.to_list()

        vocab = Vocabulary.build(train_sentences, args.min_freq)

        print(f"Builds vocabulary : {vocab}")

        vectorizer = TMVectorizer(vocab, args.max_len)

        vectorizer.save_vectorizer(vectorizer_path)

    train_dataset = TMDataset(train_df, vectorizer)
    test_dataset = TMDataset(test_df, vectorizer)
    dev_dataset = TMDataset(dev_df, vectorizer)

    model = DSSM(
        vocab_size=len(vectorizer.vocab),
        embedding_size=args.embedding_dim,
        activation=args.activation,
        internal_hidden_sizes=args.internal_hidden_sizes,
        dropout=args.dropout,
    )

    print(f"Model: {model}")

    model_saved_path = os.path.join(args.save_dir, args.model_state_file)
    if args.reload_model and os.path.exists(model_saved_path):
        model.load_state_dict(torch.load(args.model_saved_path))
        print("Reloaded model")
    else:
        print("New model")

    model = model.to(device)

    model_save_path = os.path.join(
        args.save_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}-model.pth"
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_data_loader = DataLoader(dev_dataset)
    test_data_loader = DataLoader(test_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(args.num_epochs):
        train(
            train_data_loader,
            model,
            criterion,
            optimizer,
            print_every=args.print_every,
            verbose=args.verbose,
        )
        print("Begin evalute on dev set.")
        with torch.no_grad():
            acc, p, r, f1 = evaluate(dev_data_loader, model)
            if best_f1 < f1:
                best_f1 = f1
                torch.save(model.state_dict(), model_save_path)

            print(
                f"EVALUATE [{epoch+1}/{args.num_epochs}]  accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f} best f1: {best_f1:.4f}"
            )

    # model = torch.load(model_save_path)
    # model.to(device)
    model.eval()

    acc, p, r, f1 = evaluate(test_data_loader, model)
    print(f"TEST accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}")

    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    acc, p, r, f1 = evaluate(test_data_loader, model)
    print(f"TEST accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}")
