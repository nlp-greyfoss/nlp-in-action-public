from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

from tqdm import tqdm

from config import train_args, model_args
from module import TranslationHead
from dataset import NMTDataset

import sentencepiece as spm

from dataclasses import asdict
import sacrebleu

import time


import pandas as pd


from utils import (
    build_dataframe_from_json,
    WarmupScheduler,
    count_parameters,
    EarlyStopper,
    set_random_seed,
    LabelSmoothingLoss,
)


def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> float:
    model.train()  # train mode

    total_loss = 0.0

    tqdm_iter = tqdm(data_loader)

    for batch in tqdm_iter:
        source = batch.source.to(device)
        target = batch.target.to(device)
        labels = batch.labels.to(device)

        logits = model(source, target)

        # loss calculation
        loss = criterion(logits, labels)

        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()

        description = f" TRAIN  loss={loss.item():.6f}, learning rate={scheduler.get_last_lr()[0]:.7f}"

        del loss

        tqdm_iter.set_description(description)

    # average training loss
    avg_loss = total_loss / len(data_loader)

    return avg_loss


@torch.no_grad()
def calculate_bleu(
    model: TranslationHead,
    tgt_tokenizer: spm.SentencePieceProcessor,
    data_loader: DataLoader,
    max_len: int,
    device: torch.device,
    save_result: bool = False,
    save_path: str = "result.txt",
    use_cache: bool = True,
    generation_mode: str = "beam_search",
    num_beams: int = 5,
) -> float:
    candidates = []
    references = []

    model.eval()

    for batch in tqdm(data_loader):
        source = batch.source.to(device)

        token_indices = model.translate(
            source,
            max_gen_len=max_len,
            use_cache=use_cache,
            num_beams=num_beams,
            generation_mode=generation_mode,
        )
        token_indices = token_indices.cpu().tolist()
        candidates.extend(tgt_tokenizer.decode_ids(token_indices))

        references.extend(batch.tgt_text)

    if save_result:
        with open(save_path, "w", encoding="utf-8") as f:
            for i in range(len(references)):
                f.write(
                    f"idx: {i:5} | reference: {references[i]} | candidate: {candidates[i]} \n"
                )

    bleu = sacrebleu.corpus_bleu(candidates, [references], tokenize="zh")

    return float(bleu.score)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    model.eval()  # eval mode

    total_loss = 0

    for batch in tqdm(data_loader):
        source = batch.source.to(device)
        target = batch.target.to(device)
        labels = batch.labels.to(device)

        # feed forward
        logits = model(source, target)
        # loss calculation
        loss = criterion(logits, labels)

        total_loss += loss.item()

        del loss

    # average validation loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss


if __name__ == "__main__":
    assert os.path.exists(
        train_args.src_tokenizer_file
    ), "should first run train_tokenizer.py to train the tokenizer"
    assert os.path.exists(
        train_args.tgt_tokenizer_path
    ), "should first run train_tokenizer.py to train the tokenizer"
    source_tokenizer = spm.SentencePieceProcessor(
        model_file=train_args.src_tokenizer_file
    )
    target_tokenizer = spm.SentencePieceProcessor(
        model_file=train_args.tgt_tokenizer_path
    )

    if train_args.only_test:
        train_args.use_wandb = False

    if train_args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"source tokenizer size: {source_tokenizer.vocab_size()}")
    print(f"target tokenizer size: {target_tokenizer.vocab_size()}")

    set_random_seed(12345)

    train_dataframe_path = os.path.join(
        train_args.save_dir, train_args.dataframe_file.format("train")
    )
    test_dataframe_path = os.path.join(
        train_args.save_dir, train_args.dataframe_file.format("test")
    )
    valid_dataframe_path = os.path.join(
        train_args.save_dir, train_args.dataframe_file.format("dev")
    )

    if os.path.exists(train_dataframe_path) and train_args.use_dataframe_cache:
        train_df, test_df, valid_df = (
            pd.read_pickle(train_dataframe_path),
            pd.read_pickle(test_dataframe_path),
            pd.read_pickle(valid_dataframe_path),
        )
        print("Loads cached dataframes.")
    else:
        print("Create new dataframes.")

        valid_df = build_dataframe_from_json(
            f"{train_args.dataset_path}/dev.json", source_tokenizer, target_tokenizer
        )
        print("Create valid dataframe")
        test_df = build_dataframe_from_json(
            f"{train_args.dataset_path}/test.json", source_tokenizer, target_tokenizer
        )
        print("Create train dataframe")
        train_df = build_dataframe_from_json(
            f"{train_args.dataset_path}/train.json", source_tokenizer, target_tokenizer
        )
        print("Create test dataframe")

        train_df.to_pickle(train_dataframe_path)
        test_df.to_pickle(test_dataframe_path)
        valid_df.to_pickle(valid_dataframe_path)

    pad_idx = model_args.pad_idx

    train_dataset = NMTDataset(train_df, pad_idx)
    valid_dataset = NMTDataset(valid_df, pad_idx)
    test_dataset = NMTDataset(test_df, pad_idx)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=valid_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=train_args.train_batch_size,
        collate_fn=test_dataset.collate_fn,
    )

    model = TranslationHead(
        model_args,
        target_tokenizer.pad_id(),
        target_tokenizer.bos_id(),
        target_tokenizer.eos_id(),
    )

    # print(model)

    print(f"The model has {count_parameters(model)} trainable parameters")

    model.to(device)

    args = asdict(model_args)
    args.update(asdict(train_args))

    if train_args.use_wandb:
        import wandb

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="transformer",
            config=args,
        )

    train_criterion = LabelSmoothingLoss(train_args.label_smoothing, model_args.pad_idx)
    # no label smoothing for validation
    valid_criterion = LabelSmoothingLoss(pad_idx=model_args.pad_idx)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=train_args.betas, eps=train_args.eps
    )
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=train_args.warmup_steps,
        d_model=model_args.d_model,
        factor=train_args.warmup_factor,
    )

    best_score = 0.0

    early_stopper = EarlyStopper(mode="max", patience=train_args.patient)

    print(f"begin train with arguments: {args}")

    print(f"total train steps: {len(train_dataloader) * train_args.num_epochs}")

    if not train_args.only_test:
        for epoch in range(train_args.num_epochs):
            train_loss = train(
                model,
                train_dataloader,
                train_criterion,
                optimizer,
                device,
                train_args.grad_clipping,
                scheduler,
            )
            torch.cuda.empty_cache()

            valid_loss = evaluate(model, valid_dataloader, device, valid_criterion)
            torch.cuda.empty_cache()

            valid_bleu_score = calculate_bleu(
                model,
                target_tokenizer,
                valid_dataloader,
                train_args.max_gen_len,
                device,
                save_result=True,
                save_path="result-dev.txt",
                use_cache=train_args.use_kv_cache,
                generation_mode=train_args.generation_mode,
                num_beams=train_args.num_beams,
            )
            torch.cuda.empty_cache()

            print(
                f"end of epoch {epoch+1:3d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f} |  valid bleu_score {valid_bleu_score:.2f}"
            )
            if train_args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_bleu_score": valid_bleu_score,
                        "valid_loss": valid_loss,
                    }
                )
                wandb.save(f"result-dev.txt")

            if valid_bleu_score > best_score:
                best_score = valid_bleu_score
                print(f"Save model with best bleu score :{valid_bleu_score:.2f}")

                torch.save(model.state_dict(), train_args.model_save_path)

            if early_stopper.step(valid_bleu_score):
                print(f"Stop from early stopping.")
                break

    model.load_state_dict(torch.load(train_args.model_save_path))
    # calculate bleu score
    test_bleu_score = calculate_bleu(
        model,
        target_tokenizer,
        test_dataloader,
        train_args.max_gen_len,
        device,
        save_result=True,
        use_cache=train_args.use_kv_cache,
        generation_mode=train_args.generation_mode,
        num_beams=train_args.num_beams,
    )
    print(f"Test bleu score: {test_bleu_score:.2f}")
