import torch.nn as nn
import torch
from argparse import Namespace


class SiameseNet(nn.Module):
    """The Siamese Network implemention."""

    def __init__(self, args: Namespace) -> None:
        """

        Args:
            args (Namespace): arguments for the whole network
        """
        super().__init__()

        if args.activation.lower() == "relu":
            activate_func = nn.ReLU()
        else:
            activate_func = nn.Tanh()

        self.embedding = nn.Sequential(
            nn.Embedding(args.vocab_size, args.embedding_dim),
            nn.Dropout(args.dropout),
            nn.LSTM(
                args.embedding_dim,
                args.lstm_hidden_dim,
                num_layers=args.lstm_num_layers,
                dropout=args.lstm_dropout,
                batch_first=True,
                bidirectional=True,
            ),
        )

        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_hidden_dim, args.linear_hidden_dim),
            activate_func,
            nn.Dropout(args.dropout),
        )

    def forward(self, sentence1: torch.Tensor, sentence2: torch.Tensor) -> torch.Tensor:
        """Using the same network to compute the representations of two sentences

        Args:
            sentence1 (torch.Tensor): shape (batch_size, seq_len)
            sentence2 (torch.Tensor): shape (batch_size, seq_len)

        Returns:
            torch.Tensor: the cosine similarity between sentence1 and sentence2
        """

        embed_1, _ = self.embedding(sentence1)
        embed_2, _ = self.embedding(sentence2)

        vector_1 = self.dense(torch.mean(embed_1, dim=1))
        vector_2 = self.dense(torch.mean(embed_2, dim=1))

        return torch.cosine_similarity(vector_1, vector_2, dim=1, eps=1e-8)
