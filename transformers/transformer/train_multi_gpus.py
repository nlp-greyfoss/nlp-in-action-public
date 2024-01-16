from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


import torch.distributed as dist
import torch

import time
import os
import GPUtil

from tqdm import tqdm

from config import train_args, model_args
from module import TranslationHead
from dataset import NMTDataset

import sentencepiece as spm


from dataclasses import asdict
import sacrebleu


import pandas as pd


from utils import (
    build_dataframe_from_json,
    WarmupScheduler,
    count_parameters,
    EarlyStopper,
    set_random_seed,
    LabelSmoothingLoss,
)


def setup(rank: int, world_size: int) -> None:
    """

    Args:
        rank (int): within the process group, each process is identified by its rank, from 0 to world_size - 1
        world_size (int): the number of processes in the group
    """

    # Initialize the process group
    # world_size process forms a group which is supported by a backend(nccl)
    # rank 0 as master node
    # master node: the main gpu responsible for synchronizations, making copies, loading models, writing logs.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def train(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip: float,
    gradient_accumulation_steps: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    rank: int,
) -> float:
    model.train()  # train mode

    # let all processes sync up before starting with a new epoch of training
    dist.barrier()

    total_loss = 0.0

    tqdm_iter = tqdm(data_loader)

    for step, batch in enumerate(tqdm_iter, start=1):
        source, target, labels = [
            x.to(rank) for x in (batch.source, batch.target, batch.labels)
        ]
        logits = model(source, target)

        # loss calculation
        loss = criterion(logits, labels)

        loss.backward()

        if step % gradient_accumulation_steps == 0:
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()

        description = f"[GPU{rank}] TRAIN  loss={loss.item():.6f}, learning rate={scheduler.get_last_lr()[0]:.7f}"

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
    rank: int,
    save_result: bool = False,
    save_path: str = "result.txt",
    use_cache: bool = True,
    generation_mode: str = "beam_search",
    num_beams: int = 1,
) -> float:
    candidates = []
    references = []
    sources = []

    model.eval()

    for batch in tqdm(data_loader):
        source = batch.source.to(rank)
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
        sources.extend(batch.src_text)

    if save_result:
        with open(save_path, "w", encoding="utf-8") as f:
            for i in range(len(references)):
                f.write(
                    f"idx: {i:5} | source: {sources[i]}  -> reference: {references[i]} | candidate: {candidates[i]} \n"
                )

    bleu = sacrebleu.corpus_bleu(candidates, [references], tokenize="zh")

    return float(bleu.score)


@torch.no_grad()
def evaluate(
    model: nn.Module, data_loader: DataLoader, criterion: torch.nn.Module, rank: int
) -> float:
    model.eval()  # eval mode

    total_loss = 0

    for batch in tqdm(data_loader):
        source, target, labels = [
            x.to(rank) for x in (batch.source, batch.target, batch.labels)
        ]

        # feed forward
        logits = model(source, target)
        # loss calculation
        loss = criterion(logits, labels)

        total_loss += loss.item()

        del loss

    # average validation loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def prepare_dataloader(
    dataset, rank, world_size, batch_size, pin_memory=False, num_workers=0
):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def load_tokenizer(rank):
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
    if rank == 0:
        print(f"source tokenizer size: {source_tokenizer.vocab_size()}")
        print(f"target tokenizer size: {target_tokenizer.vocab_size()}")

    return source_tokenizer, target_tokenizer


def get_dataset(rank, source_tokenizer, target_tokenizer, mode: str = "train"):
    dataframe_path = os.path.join(
        train_args.save_dir, train_args.dataframe_file.format(mode)
    )

    dataframe_path = os.path.join(
        train_args.save_dir, train_args.dataframe_file.format(mode)
    )

    if os.path.exists(dataframe_path) and train_args.use_dataframe_cache:
        df = pd.read_pickle(dataframe_path)
        if rank == 0:
            print(f"Loads cached {mode} dataframe.")
    else:
        df = build_dataframe_from_json(
            f"{train_args.dataset_path}/{mode}.json", source_tokenizer, target_tokenizer
        )

        df.to_pickle(dataframe_path)
        if rank == 0:
            print(f"Create new {mode} dataframe.")

    return NMTDataset(df, model_args.pad_idx)


def main(rank, world_size):
    print(f"Running  DDP on rank {rank}.")

    torch.cuda.set_device(rank)

    setup(rank, world_size)

    source_tokenizer, target_tokenizer = load_tokenizer(rank)

    set_random_seed(train_args.seed)

    train_dataset = get_dataset(rank, source_tokenizer, target_tokenizer, "train")
    valid_dataset = get_dataset(rank, source_tokenizer, target_tokenizer, "dev")

    train_dataloader = prepare_dataloader(
        train_dataset, rank, world_size, train_args.train_batch_size
    )

    valid_dataloader = prepare_dataloader(
        valid_dataset, rank, world_size, train_args.eval_batch_size
    )

    model = TranslationHead(
        model_args,
        target_tokenizer.pad_id(),
        target_tokenizer.bos_id(),
        target_tokenizer.eos_id(),
    ).to(rank)

    is_main_process = rank == 0

    if is_main_process:
        print(f"The model has {count_parameters(model)} trainable parameters")

    model = DDP(model, device_ids=[rank])
    module = model.module  # the wrapped model

    args = asdict(model_args)
    args.update(asdict(train_args))

    if train_args.use_wandb and is_main_process:
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

    if train_args.calc_bleu_during_train:
        # bleu score
        early_stopper = EarlyStopper(mode="max", patience=train_args.patient)
        best_score = 0.0
    else:
        # dev loss
        early_stopper = EarlyStopper(mode="min", patience=train_args.patient)
        best_score = 1000

    if is_main_process:
        print(f"begin train with arguments: {args}")

        print(f"total train steps: {len(train_dataloader) * train_args.num_epochs}")

    for epoch in range(train_args.num_epochs):
        start = time.time()
        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        train_loss = train(
            model,
            train_dataloader,
            train_criterion,
            optimizer,
            train_args.grad_clipping,
            train_args.gradient_accumulation_steps,
            scheduler,
            rank,
        )

        if is_main_process:
            print()
            GPUtil.showUtilization()

        torch.cuda.empty_cache()
        if is_main_process:
            print("begin evaluate")
        valid_loss = evaluate(model, valid_dataloader, valid_criterion, rank)
        torch.cuda.empty_cache()

        if train_args.calc_bleu_during_train:
            if is_main_process:
                print("calculate bleu score for dev dataset")
            valid_bleu_score = calculate_bleu(
                model.module,
                target_tokenizer,
                valid_dataloader,
                train_args.max_gen_len,
                rank,
                save_result=True,
                save_path="result-dev.txt",
                use_cache=train_args.use_kv_cache,
                generation_mode=train_args.generation_mode,
                num_beams=train_args.num_beams,
            )
            torch.cuda.empty_cache()
            metric_score = valid_bleu_score
        else:
            valid_bleu_score = 0
            metric_score = valid_loss

        elapsed = time.time() - start

        print(
            f"[GPU{rank}] end of epoch {epoch+1:3d} [{elapsed:4.0f}s]| train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f} |  valid bleu_score {valid_bleu_score:.2f}"
        )

        if is_main_process:
            if train_args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "valid_bleu_score": valid_bleu_score,
                        "valid_loss": valid_loss,
                    }
                )
                wandb.save(f"result-dev.txt")

            if train_args.calc_bleu_during_train:
                if metric_score > best_score:
                    best_score = metric_score

                    print(f"Save model with best bleu score :{metric_score:.2f}")
                    torch.save(module.state_dict(), train_args.model_save_path)
            else:
                if metric_score < best_score:
                    best_score = metric_score
                    print(f"Save model with best valid loss :{metric_score:.4f}")
                    torch.save(module.state_dict(), train_args.model_save_path)

            if early_stopper.step(metric_score):
                print(f"stop from early stopping.")
                break
    cleanup()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, train_args.gpus))

    # Sets up the process group and configuration for PyTorch Distributed Data Parallelism
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = min(torch.cuda.device_count(), len(train_args.gpus))

    print(f"Number of GPUs used: {world_size}")

    mp.spawn(main, args=(world_size,), nprocs=world_size)
