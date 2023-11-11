"""Implements model proposed by paper https://arxiv.org/pdf/1611.01747.pdf

"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Preprocess(nn.Module):
    """Implements the preprocess layer"""

    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        """

        Args:
            embedding_dim (int): embedding size
            hidden_size (int): hidden size
        """
        super().__init__()
        self.Wi = nn.Parameter(torch.randn(embedding_dim, hidden_size))
        self.bi = nn.Parameter(torch.randn(hidden_size))

        self.Wu = nn.Parameter(torch.randn(embedding_dim, hidden_size))
        self.bu = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): the input sentence with shape (batch_size, seq_len, embedding_size)

        Returns:
            torch.Tensor:
        """
        # e_xi (batch_size, seq_len, hidden_size)
        e_xi = torch.matmul(x, self.Wi)
        # gate (batch_size, seq_len, hidden_size)
        gate = torch.sigmoid(e_xi + self.bi)
        # e_xu (batch_size, seq_len, hidden_size)
        e_xu = torch.matmul(x, self.Wu)
        # value (batch_size, seq_len, hidden_size)
        value = torch.tanh(e_xu + self.bu)
        # x_bar (batch_size, seq_len, hidden_size)
        x_bar = gate * value

        return x_bar


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.Wg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.randn(hidden_size))

    def forward(self, q_bar: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
        """forward in attention layer

        Args:
            q_bar (torch.Tensor): the question sentencce with shape (batch_size, q_seq_len, hidden_size)
            a_bar (torch.Tensor): the answer sentence with shape (batch_size, a_seq_len, hidden_size)

        Returns:
            torch.Tensor: weighted sum of q_bar
        """
        # e_q_bar (batch_size, q_seq_len, hidden_size)
        e_q = torch.matmul(q_bar, self.Wg)
        # transform (batch_size, q_seq_len, hidden_size)
        transform = e_q + self.bg
        # g (batch_size, q_seq_len, a_seq_len)
        g = torch.softmax(torch.matmul(transform, a_bar.permute(0, 2, 1)), dim=1)
        # h (batch_size, a_seq_len, hidden_size)
        h = torch.matmul(g.permute(0, 2, 1), a_bar)

        return h


class Compare(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.b = nn.Parameter(torch.randn(hidden_size))

    def forward(self, h: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
        """

        Args:
            h (torch.Tensor): the output of Attention layer  (batch_size, a_seq_len, hidden_size)
            a_bar (torch.Tensor): proprecessed a (batch_size, a_seq_len, hidden_size)

        Returns:
            torch.Tensor:
        """
        # sub (batch_size, a_seq_len, hidden_size)
        sub = (h - a_bar) ** 2
        # mul (batch_size, a_seq_len, hidden_size)
        mul = h * a_bar
        # t (batch_size, a_seq_len, hidden_size)
        t = torch.relu(torch.matmul(torch.cat([sub, mul], dim=-1), self.W) + self.b)

        return t


class Aggregation(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_filter: int,
        filter_sizes: list[int],
        output_dim: int,
        conv_activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        """_summary_

        Args:
            embedding_dim (int): embedding size
            num_filter (int): the output dim of each convolution layer
            filter_sizes (list[int]): the size of the convolving kernel
            output_dim: (int) the number of classes
            conv_activation (str, optional): activation to use after the convolution layer. Defaults to "relu".
            dropout (float): the dropout ratio
        """

        super().__init__()

        if conv_activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=num_filter,
                        kernel_size=(fs, embedding_dim),
                    ),
                    activation,
                )
                for fs in filter_sizes
            ]
        )

        pooled_output_dim = num_filter * len(filter_sizes)

        self.linear = nn.Linear(pooled_output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """

        Args:
            t (torch.Tensor):  the output of Compare  (batch_size, a_seq_len, hidden_size)

        Returns:
            torch.Tensor:
        """
        # t (batch_size, 1, a_seq_len, hidden_size)
        t = t.unsqueeze(1)
        # the shape of convs_out(t) is (batch_size, num_filter, a_seq_len - filter_size + 1, 1)
        # element in convs_out with shape (batch_size, num_filter, a_seq_len - filter_size + 1)
        convs_out = [self.dropout(conv(t).squeeze(-1)) for conv in self.convs]
        # adaptive_avg_pool1d applies a 1d adaptive max pooling over an input
        # adaptive_avg_pool1d(o, output_size=1) returns an output with shape (batch_size, num_filter, 1)
        # so the elements in maxpool_out have a shape of (batch_size, num_filter)
        maxpool_out = [
            F.adaptive_avg_pool1d(o, output_size=1).squeeze(-1) for o in convs_out
        ]
        # cat (batch_size, num_filter * len(filter_sizes))
        cat = torch.cat(maxpool_out, dim=1)
        # (batch_size, output_dim)
        return self.linear(cat)


class ComAgg(nn.Module):
    """The Compare aggregate MODEL model implemention."""

    def __init__(self, args) -> None:
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.preprocess = Preprocess(args.embedding_dim, args.hidden_size)
        self.attention = Attention(args.hidden_size)
        self.compare = Compare(args.hidden_size)
        self.aggregate = Aggregation(
            args.hidden_size,
            args.num_filter,
            args.filter_sizes,
            args.num_classes,
            args.conv_activation,
            args.dropout,
        )
        self.dropouts = [nn.Dropout(args.dropout) for _ in range(4)]

    def forward(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            q (torch.Tensor): the inputs of q (batch_size, q_seq_len)
            a (torch.Tensor): the inputs of a (batch_size, a_seq_len)

        Returns:
            torch.Tensor: _description_
        """
        q_embed = self.dropouts[0](self.embedding(q))
        a_embed = self.dropouts[0](self.embedding(a))

        q_bar = self.dropouts[1](self.preprocess(q_embed))
        a_bar = self.dropouts[1](self.preprocess(a_embed))

        h = self.dropouts[2](self.attention(q_bar, a_bar))
        # t (batch_size, a_seq_len, hidden_size)
        t = self.dropouts[3](self.compare(h, a_bar))
        # out (batch_size, num_filter * len(filter_sizes))
        out = self.aggregate(t)

        return out
