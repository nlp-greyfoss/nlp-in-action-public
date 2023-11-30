from argparse import Namespace
from datetime import datetime
import os
import torch
import pandas as pd
from typing import Tuple

import numpy as np

from dataset import TMVectorizer, TMDataset, Vocabulary
from tqdm import tqdm


from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gensim.models import KeyedVectors


from model import MatchPyramid
from utils import *


def evaluate(
    data_iter: DataLoader, model: nn.Module
) -> Tuple[float, float, float, float]:
    y_list, y_pred_list = [], []
    model.eval()
    for x1, x2, _, _, y in tqdm(data_iter):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        y = y.float().to(device)
        with torch.no_grad():
            output = model(x1, x2)

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
    grad_clipping: float,
) -> None:
    model.train()

    tqdm_iter = tqdm(data_iter)

    running_loss = 0.0

    for step, (x1, x2, _, _, y) in enumerate(tqdm_iter):
        x1 = x1.to(device).long()
        x2 = x2.to(device).long()
        y = torch.LongTensor(y).to(device)

        output = model(x1, x2)

        loss = criterion(output, y)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

        optimizer.step()

        description = f" TRAIN iter={step+1} loss={running_loss / (step + 1):.6f}"
        tqdm_iter.set_description(description)


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="text_matching/data/lcqmc/{}.txt",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        pandas_file="dataframe.{}.pkl",
        save_dir=f"{os.path.dirname(__file__)}{os.sep}model_storage",
        reload_model=False,
        cuda=True,
        learning_rate=5e-4,
        batch_size=128,
        num_epochs=50,
        max_len=50,
        embedding_dim=300,
        embedding_saved_path="text_matching/data/embeddings.pt",
        embedding_pretrained_path="./word2vec.zh.300.char.model",
        load_embeding=False,
        fix_embeddings=False,
        hidden_size=150,
        out_channels=[8, 16],
        kernel_sizes=[(5, 5), (3, 3)],
        pool_sizes=[(10, 10), (5, 5)],
        dropout=0.2,
        min_freq=2,
        project_func="linear",
        grad_clipping=2.0,
        num_classes=2,
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

    model = MatchPyramid(args)

    if args.load_embeding and os.path.exists(args.embedding_saved_path):
        model.embedding.load_state_dict(torch.load(args.embedding_saved_path))
        print("loading saved embedding")
    elif args.load_embeding and os.path.exists(args.embedding_pretrained_path):
        wv = KeyedVectors.load_word2vec_format(args.embedding_pretrained_path)

        embeddings = load_embedings(vocab, wv)

        model.embedding.load_state_dict({"weight": torch.tensor(embeddings)})

        torch.save(model.embedding.state_dict(), args.embedding_saved_path)
        print("loading pretrained embedding")
    else:
        print("init embedding from stratch")

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

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_value = 0.0

    early_stopper = EarlyStopper(mode="max")

    for epoch in range(args.num_epochs):
        train(train_data_loader, model, criterion, optimizer, args.grad_clipping)

        acc, p, r, f1 = evaluate(dev_data_loader, model)

        lr_scheduler.step(acc)

        if acc > best_value:
            best_value = acc
            print(f"Save model with best acc :{acc:4f}")
            torch.save(model.state_dict(), model_save_path)

        if early_stopper.step(acc):
            print(f"Stop from early stopping.")
            break

        print(
            f"EVALUATE [{epoch+1}/{args.num_epochs}]  accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
        )

    model.eval()

    acc, p, r, f1 = evaluate(test_data_loader, model)
    print(f"TEST accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}")

    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    acc, p, r, f1 = evaluate(test_data_loader, model)
    print(
        f"TEST[best score] accuracy={acc:.3f} precision={p:.3f} recal={r:.3f} f1 score={f1:.4f}"
    )
