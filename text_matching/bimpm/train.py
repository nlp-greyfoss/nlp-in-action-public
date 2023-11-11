from argparse import Namespace
import os
import torch
import pandas as pd
from typing import Tuple

from dataset import TMVectorizer, TMDataset, Vocabulary
from tqdm import tqdm

import jieba

from torch.utils.data import DataLoader
import torch.nn as nn

from model import BiMPM


def tokenize(sentence: str):
    tokens = []
    for word in jieba.cut(sentence):
        if word.isdigit():
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


def build_dataframe_from_csv(dataset_csv: str) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_csv,
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
    )
    # remove all punctuations
    df.sentence1 = df.sentence1.str.replace(r"[^\u4e00-\u9fa50-9]", "", regex=True)
    df.sentence2 = df.sentence2.str.replace(r"[^\u4e00-\u9fa50-9]", "", regex=True)
    df = df.dropna()

    return df


def tokenize_df(df):
    df.sentence1 = df.sentence1.apply(tokenize)
    df.sentence2 = df.sentence2.apply(tokenize)
    return df


def make_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


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
    for x1, x2, c1, c2, y in tqdm(data_iter):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        c1 = c1.to(device).long()
        c2 = c2.to(device).long()
        y = torch.LongTensor(y).to(device)

        output = model(x1, x2, c1, c2)

        pred = torch.argmax(output, dim=1).long()

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

    for step, (x1, x2, c1, c2, y) in enumerate(tqdm(data_iter)):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        c1 = c1.to(device).long()
        c2 = c2.to(device).long()
        y = torch.LongTensor(y).to(device)

        output = model(x1, x2, c1, c2)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % print_every == 0:
            pred = torch.argmax(output, dim=1).long()
            acc, p, r, f1 = metrics(y, pred)

            print(
                f" TRAIN iter={step+1} loss={loss.item():.6f} accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
            )


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="text_matching/data/lcqmc/{}.txt",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir=f"{os.path.dirname(__file__)}/model_storage",
        reload_model=False,
        cuda=False,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=10,
        max_len=50,
        char_vocab_size=4699,
        word_embedding_dim=300,
        word_vocab_size=35092,
        max_word_len=8,
        char_embedding_dim=20,
        hidden_size=100,
        char_hidden_size=50,
        num_perspective=20,
        num_classes=2,
        dropout=0.2,
        epsilon=1e-8,
        min_word_freq=2,
        min_char_freq=1,
        print_every=500,
        verbose=True,
    )

    make_dirs(args.save_dir)

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}.")

    vectorizer_path = os.path.join(args.save_dir, args.vectorizer_file)

    train_df = build_dataframe_from_csv(args.dataset_csv.format("train"))
    test_df = build_dataframe_from_csv(args.dataset_csv.format("test"))
    dev_df = build_dataframe_from_csv(args.dataset_csv.format("dev"))

    print("Creating a new Vectorizer.")

    train_chars = train_df.sentence1.to_list() + train_df.sentence2.to_list()

    char_vocab = Vocabulary.build(train_chars, args.min_char_freq)

    args.char_vocab_size = len(char_vocab)

    train_word_df = tokenize_df(train_df)
    test_word_df = tokenize_df(test_df)
    dev_word_df = tokenize_df(dev_df)

    train_sentences = train_df.sentence1.to_list() + train_df.sentence2.to_list()

    word_vocab = Vocabulary.build(train_sentences, args.min_word_freq)

    args.word_vocab_size = len(word_vocab)

    words = [word_vocab.lookup_token(idx) for idx in range(args.word_vocab_size)]

    longest_word = ""

    for word in words:
        if len(word) > len(longest_word):
            longest_word = word

    args.max_word_len = len(longest_word)

    char_vectorizer = TMVectorizer(char_vocab, len(longest_word))
    word_vectorizer = TMVectorizer(word_vocab, args.max_len)

    train_dataset = TMDataset(train_df, char_vectorizer, word_vectorizer)
    test_dataset = TMDataset(test_df, char_vectorizer, word_vectorizer)
    dev_dataset = TMDataset(dev_df, char_vectorizer, word_vectorizer)

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"Arguments : {args}")
    model = BiMPM(args)

    print(f"Model: {model}")

    model = model.to(device)

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
