import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from argparse import Namespace


class MatchPyramid(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0
        )

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=args.out_channels[0],
            kernel_size=args.kernel_sizes[0],
        )
        self.conv2 = nn.Conv2d(
            in_channels=args.out_channels[0],
            out_channels=args.out_channels[1],
            kernel_size=args.kernel_sizes[1],
        )
        self.pool1 = nn.AdaptiveMaxPool2d(args.pool_sizes[0])
        self.pool2 = nn.AdaptiveMaxPool2d(args.pool_sizes[1])

        self.linear = torch.nn.Linear(
            args.out_channels[1] * args.pool_sizes[1][0] * args.pool_sizes[1][1],
            args.hidden_size,
            bias=True,
        )
        self.prediction = torch.nn.Linear(args.hidden_size, args.num_classes, bias=True)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """

        Args:
            a (Tensor): (batch_size, a_seq_len)
            b (Tensor): (batch_size, b_seq_len)

        Returns:
            Tensor: (batch_size, num_classes)
        """
        batch_size = a.size()[0]

        # (batch_size, a_seq_len, embedding_dim)
        a = self.embedding(a)
        # (batch_size, b_seq_len, embedding_dim)
        b = self.embedding(b)
        # (batch_size, a_seq_len, 1, embedding_dim) x  (batch_size, 1, b_seq_len, embedding_dim)
        # -> (batch_size, a_seq_len, b_seq_len)
        similarity_matrix = F.cosine_similarity(a.unsqueeze(2), b.unsqueeze(1), dim=-1)
        # (batch_size, 1, a_seq_len, b_seq_len)
        similarity_matrix = similarity_matrix.unsqueeze(1)
        # (batch_size, out_channels[0], a_seq_len - kernel_sizes[0][0] + 1, b_seq_len - kernel_sizes[0][1] + 1)
        similarity_matrix = F.relu(self.conv1(similarity_matrix))
        # (batch_size, out_channels[0], pool_sizes[0][0], pool_sizes[0][1])
        similarity_matrix = self.pool1(similarity_matrix)
        # (batch_size, out_channels[1], pool_sizes[1][0] - kernel_sizes[1][0] + 1, pool_sizes[1][1] - kernel_sizes[1][1] + 1)
        similarity_matrix = F.relu(self.conv2(similarity_matrix))
        # (batch_size, out_channels[1], pool_sizes[1][0], pool_sizes[1][1])
        similarity_matrix = self.pool2(similarity_matrix)
        # (batch_size, out_channels[1] * pool_sizes[1][0] * pool_sizes[1][1])
        similarity_matrix = similarity_matrix.view(batch_size, -1)
        # (batch_size, num_classes)
        return self.prediction(F.relu(self.linear(similarity_matrix)))
