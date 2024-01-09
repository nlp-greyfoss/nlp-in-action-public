from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor

from dataclasses import dataclass


@dataclass
class Batch:
    source: Tensor
    target: Tensor
    labels: Tensor
    num_tokens: int
    src_text: str = None
    tgt_text: str = None


class NMTDataset(Dataset):
    """Dataset for translation"""

    def __init__(self, text_df: pd.DataFrame, pad_idx: int = 0) -> None:
        """

        Args:
            text_df (pd.DataFrame): a DataFrame which contains the processed source and target sentences
        """
        self.text_df = text_df

        self.padding_index = pad_idx

    def __getitem__(
        self, index: int
    ) -> Tuple[list[int], list[int], list[str], list[str]]:
        row = self.text_df.iloc[index]

        return (row.source_indices, row.target_indices, row.source, row.target)

    def collate_fn(
        self, batch: list[Tuple[list[int], list[int], list[str]]]
    ) -> Tuple[LongTensor, LongTensor, LongTensor]:
        source_indices = [x[0] for x in batch]
        target_indices = [x[1] for x in batch]
        source_text = [x[2] for x in batch]
        target_text = [x[3] for x in batch]

        source_indices = [torch.LongTensor(indices) for indices in source_indices]
        target_indices = [torch.LongTensor(indices) for indices in target_indices]

        # The <eos> was added before the <pad> token to ensure that the model can correctly identify the end of a sentence.
        source = pad_sequence(
            source_indices, padding_value=self.padding_index, batch_first=True
        )

        target = pad_sequence(
            target_indices, padding_value=self.padding_index, batch_first=True
        )

        labels = target[:, 1:]
        target = target[:, :-1]

        num_tokens = (labels != self.padding_index).data.sum()

        return Batch(source, target, labels, num_tokens, source_text, target_text)

    def __len__(self) -> int:
        return len(self.text_df)
