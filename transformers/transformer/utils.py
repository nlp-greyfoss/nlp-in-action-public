from torch import Tensor
import torch
import torch.nn as nn

import pandas as pd

from collections import UserDict

import os
import json
from tqdm import tqdm
import numpy as np


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from zhconv import convert
import sentencepiece as spm
import torch.nn.functional as F





class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, pad_idx: int = 0) -> None:
        """

        Args:
            label_smoothing (float, optional): label smoothing value. Defaults to 0.0.
            pad_idx (int, optional): pad index. Defaults to 0.
        """
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=label_smoothing
        )

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """

        Args:
            logits (Tensor): (batch_size, max_target_seq_len, vocab_size) output of the model.
            labels (Tensor): (batch_size, max_target_seq_len) the label index list.
            num_tokens (int): number of unpadded tokens

        Returns:
            Tensor: loss
        """
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)

        return self.loss_func(logits, labels)


class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        d_model: int,
        factor: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): warmup steps
            d_model (int): dimension of embeddings.
            last_epoch (int, optional): the index of last epoch. Defaults to -1.
            verbose (bool, optional): if True, prints a message to stdout for each update. Defaults to False.

        """
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.num_parm_groups = len(optimizer.param_groups)
        self.factor = factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        lr = (
            self.factor
            * self.d_model**-0.5
            * min(
                self._step_count**-0.5, self._step_count * self.warmup_steps**-1.5
            )
        )
        return [lr] * self.num_parm_groups


def convert_to_zh(text: str) -> str:
    return convert(text, "zh-cn")


def set_random_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataframe_from_json(
    json_path: str,
    source_tokenizer: spm.SentencePieceProcessor = None,
    target_tokenizer: spm.SentencePieceProcessor = None,
) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data, columns=["source", "target"])

    def _source_vectorize(text: str) -> list[str]:
        return source_tokenizer.EncodeAsIds(text, add_bos=True, add_eos=True)

    def _target_vectorize(text: str) -> list[str]:
        return target_tokenizer.EncodeAsIds(text, add_bos=True, add_eos=True)

    tqdm.pandas()

    if source_tokenizer:
        df["source_indices"] = df.source.progress_apply(lambda x: _source_vectorize(x))
    if target_tokenizer:
        df["target_indices"] = df.target.progress_apply(lambda x: _target_vectorize(x))

    return df


def build_dataframe_from_csv(
    dataset_csv: str, to_simplified_chinese=True
) -> pd.DataFrame:
    """
    data:

    I won!	我赢了。
    Go away!	走開！
    """
    df = pd.read_csv(
        dataset_csv,
        sep="\t",
        header=None,
        names=["source", "target"],
        skip_blank_lines=True,
    )
    # convert traditional Chinese to simplified Chinese
    if to_simplified_chinese:
        df.target = df.target.apply(convert_to_zh)

    return df


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dirs(dirpath: str) -> None:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


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