from dataclasses import dataclass

import torch
from torch import Tensor, nn

from transformers.file_utils import ModelOutput

from transformers import (
    AutoModel,
    AutoTokenizer,
)

import numpy as np
from tqdm.autonotebook import trange
from typing import Optional

from enum import Enum
import torch.nn.functional as F


class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


@dataclass
class BiOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class SentenceBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        max_length: int = None,
        margin: float = 0.5,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        pooling_mode: str = "mean",
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        ).to(self.device)

        self.max_length = max_length
        self.pooling_mode = pooling_mode

        self.distance_metric = distance_metric
        self.margin = margin

    def sentence_embedding(self, last_hidden_state, attention_mask):
        if self.pooling_mode == "mean":
            attention_mask = attention_mask.unsqueeze(-1).float()
            return torch.sum(last_hidden_state * attention_mask, dim=1) / torch.clamp(
                attention_mask.sum(1), min=1e-9
            )
        else:
            # cls
            return last_hidden_state[:, 0]

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 64,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            batch = sentences[start_index : start_index + batch_size]

            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=self.max_length,
            ).to(self.device)

            out_features = self.model(**features, return_dict=True)
            embeddings = self.sentence_embedding(
                out_features.last_hidden_state, features["attention_mask"]
            )
            if not self.training:
                embeddings = embeddings.detach()

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            if not convert_to_tensor:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        return all_embeddings

    def compute_loss(self, source_embed, target_embed, labels):
        labels = torch.tensor(labels).float().to(self.device)

        distances = self.distance_metric(source_embed, target_embed)

        loss = 0.5 * (
            labels * distances.pow(2)
            + (1 - labels) * F.relu(self.margin - distances).pow(2)
        )
        return loss.mean()

    def forward(self, source, target, labels) -> BiOutput:
        """
        Args:
            source :
            target :
        """
        source_embed = self.encode(source)
        target_embed = self.encode(target)

        loss = self.compute_loss(source_embed, target_embed, labels)
        return BiOutput(loss, None)

    def save_pretrained(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu().contiguous() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)
