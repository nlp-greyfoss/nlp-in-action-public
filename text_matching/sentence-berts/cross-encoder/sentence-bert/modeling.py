import torch
from torch import nn
import numpy as np

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding


import logging

logger = logging.getLogger(__name__)


class SentenceBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.config.num_labels = 1

        # reranker
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config, trust_remote_code=trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        self.max_length = max_length

        self.loss_fct = nn.BCEWithLogitsLoss()

    def batching_collate(self, batch: list[tuple[str, str]]) -> BatchEncoding:
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(
            *texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length
        ).to(self.model.device)

        return tokenized

    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 64,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
    ):
        dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=self.batching_collate,
            shuffle=False,
        )

        preds = []

        for batch in tqdm(
            dataloader, disable=not show_progress_bar, desc="Running Inference"
        ):
            with torch.no_grad():
                logits = self.model(**batch).logits
                logits = torch.sigmoid(logits)

                preds.extend(logits)

        if convert_to_tensor:
            preds = torch.stack(preds)
        else:
            preds = np.asarray([pred.cpu().detach().float().numpy() for pred in preds])

        return preds

    def forward(self, inputs, labels=None):

        outputs = self.model(**inputs, return_dict=True)

        if labels is not None:
            labels = labels.float()

            logits = outputs.logits
            logits = logits.view(-1)

            loss = self.loss_fct(logits, labels)

            return SequenceClassifierOutput(loss, **outputs)

        return outputs

    def save_pretrained(self, output_dir: str) -> None:
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu().contiguous() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)
