from torch import Tensor
import torch
import torch.nn as nn

import pandas as pd

from collections import UserDict

import os
import json
from tqdm import tqdm
import numpy as np

from typing import Tuple

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

        self.best_value = 0.0 if mode == "max" else float("inf")
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


class BeamSearchScorer:
    """
    Adapted from https://github.com/huggingface/transformers/blob/v3.5.1/src/transformers/generation_beam_search.py
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = True,
        num_beam_hyps_to_keep: int = 1,
    ):
        """

        Args:
            batch_size (int):  Batch Size of `input_ids` for which beam search decoding is run in parallel.
            max_length (int):  The maximum length of the sequence to be generated.
            num_beams (int):  Number of beams for beam search.
            device (torch.device): the device.
            length_penalty (float, optional): Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences. Defaults to 1.0.
            do_early_stopping (bool, optional):   Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not. Defaults to True.
            num_beam_hyps_to_keep (int, optional): The number of beam hypotheses that shall be returned upon calling. Defaults to 1.
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._beam_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, do_early_stopping)
            for _ in range(batch_size)
        ]

        self._done = torch.tensor(
            [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
        )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.Tensor]:
        """

        Args:
            input_ids (torch.LongTensor): (batch_size * num_beams, seq_len)  Indices of input sequence tokens in the vocabulary.
            next_scores (torch.FloatTensor): (batch_size, 2 * num_beams) Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (torch.LongTensor): (batch_size, 2 * num_beams) `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (torch.LongTensor): (batch_size, 2 * num_beams)  Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            pad_token_id (int, optional):  The id of the `padding` token.
            eos_token_id (int, optional): The id of the `end-of-sequence` token.

        Returns:
            Tuple[torch.Tensor]:
                next_beam_scores： (batch_size * num_beams)  Updated scores of all non-finished beams.
                next_beam_tokens： (batch_size * num_beams)  Next tokens to be added to the non-finished beam_hypotheses.
                next_beam_indices： (batch_size * num_beams)  Beam indices indicating to which beam the next tokens shall be added.
        """
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)

        assert batch_size == (input_ids.shape[0] // self.num_beams)

        device = input_ids.device
        next_beam_scores = torch.zeros(
            (batch_size, self.num_beams), dtype=next_scores.dtype, device=device
        )
        next_beam_tokens = torch.zeros(
            (batch_size, self.num_beams), dtype=next_tokens.dtype, device=device
        )
        next_beam_indices = torch.zeros(
            (batch_size, self.num_beams), dtype=next_indices.dtype, device=device
        )

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(
                    next_tokens[batch_idx],
                    next_scores[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if next_token.item() == eos_token_id:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.num_beams
                    )
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    # add next predicted token since it not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it
                if beam_idx == self.num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """

        Args:
            input_ids (torch.LongTensor): (batch_size * num_beams, seq_len)  Indices of input sequence tokens in the vocabulary.
            final_beam_scores (torch.FloatTensor): (batch_size * num_beams)  The final scores of all non-finished beams.
            pad_token_id (int, optional):
            eos_token_id (int, optional):

        Returns:
            torch.LongTensor: (batch_size * num_return_sequences, seq_len)  The generated sequences. The second dimension (seq_len)
            is either equal to `max_length` or shorter if allbatches finished early due to the `eos_token_id`.
        """
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded


class BeamHypotheses:
    def __init__(
        self,
        num_beams: int,
        max_length: int,
        length_penalty: float,
        early_stopping: bool,
    ):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float) -> None:
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
