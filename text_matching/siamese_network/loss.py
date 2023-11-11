import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, m: float = 0.2) -> None:
        """

        Args:
            m (float, optional): margin. Defaults to 0.2.
        """
        super().__init__()
        self.m = m

    def forward(self, energy: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Computes the contrastive loss between the embeddings of x1 and x2

        Args:
            energy (torch.Tensor):  the cosine similarity between the embeddings of x1 and x2
            label (torch.Tensor): an integer indicates whether x1 and x2 are similar (= 1) or dissimilar (= 0).

        Returns:
            torch.Tensor:
        """
        loss_pos = 0.25 * (1 - energy) ** 2
        loss_neg = (
            torch.where(
                energy < self.m,
                torch.full_like(energy, 0),
                energy,
            )
            ** 2
        )

        loss = label * loss_pos + (1 - label) * loss_neg

        return loss.sum()
