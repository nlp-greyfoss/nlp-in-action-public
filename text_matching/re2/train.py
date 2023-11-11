from argparse import Namespace
from datetime import datetime
import os
import torch
import pandas as pd
from typing import Tuple

from dataset import TMVectorizer, TMDataset, Vocabulary
from tqdm import tqdm

import jieba

from torch.utils.data import DataLoader
import torch.nn as nn

from model import RE2


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


def evaluate(
    data_iter: DataLoader, model: nn.Module
) -> Tuple[float, float, float, float]:
    y_list, y_pred_list = [], []
    model.eval()
    for x1, x2, y in tqdm(data_iter):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        y = y.float().to(device)

        similarity = model(x1, x2)

        pred = (similarity > 0.5).int()

        y_pred_list.append(pred)
        y_list.append(y)

    y_pred = torch.cat(y_pred_list, 0)
    y = torch.cat(y_list, 0)
    acc, p, r, f1 = metrics(y, y_pred)
    return acc, p, r, f1


def train(
    data_iter: DataLoader,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    print_every: int = 500,
    verbose=True,
) -> None:
    model.train()

    for step, (x1, x2, mask1, mask2, y) in enumerate(tqdm(data_iter)):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        mask1 = mask1.to(device).bool().unsqueeze(2)
        mask2 = mask2.to(device).bool().unsqueeze(2)
        y = y.float().to(device)

        similarity = model(x1, x2, mask1, mask2)

        loss = criterion(similarity, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % print_every == 0:
            pred = (similarity > 0.5).int()
            acc, p, r, f1 = metrics(y, pred)

            print(
                f" TRAIN iter={step+1} loss={loss.item():.6f} accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
            )


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="text_matching/data/lcqmc/{}.txt",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        pandas_file="dataframe.{}.pkl",
        save_dir=f"{os.path.dirname(__file__)}{os.sep}model_storage",
        reload_model=False,
        cuda=True,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=10,
        max_len=50,
        embedding_dim=300,
        hidden_size=150,
        encoder_layers=2,
        num_blocks=2,
        kernel_sizes=[3],
        dropout=0.8,
        min_freq=2,
        project_func="linear",
        grad_clipping=5,
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

    train_dataframe_path = os.path.join(args.save_dir, args.pandas_file.format("train"))
    test_dataframe_path = os.path.join(args.save_dir, args.pandas_file.format("test"))
    dev_dataframe_path = os.path.join(args.save_dir, args.pandas_file.format("dev"))

    if os.path.exists(train_dataframe_path):
        train_df, test_df, dev_df = (
            pd.read_pickle(train_dataframe_path),
            pd.read_pickle(test_dataframe_path),
            pd.read_pickle(dev_dataframe_path),
        )
        print("Loads cached dataframes.")
    else:
        train_df = build_dataframe_from_csv(args.dataset_csv.format("train"))
        test_df = build_dataframe_from_csv(args.dataset_csv.format("test"))
        dev_df = build_dataframe_from_csv(args.dataset_csv.format("dev"))

        print("Created new dataframes.")

        train_df.to_pickle(train_dataframe_path)
        test_df.to_pickle(test_dataframe_path)
        dev_df.to_pickle(dev_dataframe_path)

    if os.path.exists(vectorizer_path):
        vectorizer = TMVectorizer.load_vectorizer(vectorizer_path)
        print("Loads vectorizer file.")
        args.vocab_size = len(vectorizer.vocab)
    else:
        print("Creating a new Vectorizer.")

        train_sentences = train_df.sentence1.to_list() + train_df.sentence2.to_list()

        vocab = Vocabulary.build(train_sentences, args.min_freq)

        args.vocab_size = len(vocab)

        print(f"Builds vocabulary : {vocab}")

        vectorizer = TMVectorizer(vocab, args.max_len)

        vectorizer.save_vectorizer(vectorizer_path)

    train_dataset = TMDataset(train_df, vectorizer)
    test_dataset = TMDataset(test_df, vectorizer)
    dev_dataset = TMDataset(dev_df, vectorizer)

    model = RE2(args)

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
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

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

            print(
                f"EVALUATE [{epoch+1}/{args.num_epochs}]  accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
            )

    model.eval()

    acc, p, r, f1 = evaluate(test_data_loader, model)
    print(f"TEST accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}")
