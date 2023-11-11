import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from argparse import Namespace
import math

from torch.nn.utils import weight_norm


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (batch_size, seq_len)

        Returns:
            Tensor: (batch_size, seq_len, embedding_dim)
        """
        return self.dropout(self.embedding(x))


class GeLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class Conv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_sizes: list[int]
    ) -> None:
        """

        Args:
            in_channels (int): the embedding_dim
            out_channels (int): number of filters
            kernel_sizes (list[int]): the size of kernels
        """
        super().__init__()

        out_channels = out_channels // len(kernel_sizes)

        convs = []
        # (in_channels + 2 * padding - kernel_size + 1)
        # (in_channels + kernel_size - 1 -  kernel_size + 1)
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            )
            convs.append(nn.Sequential(weight_norm(conv), GeLU()))

        self.model = nn.ModuleList(convs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for seq in self.model:
            conv = seq[0]
            nn.init.normal_(
                conv.weight,
                std=math.sqrt(2.0 / (conv.in_channels * conv.kernel_size[0])),
            )
            nn.init.zeros_(conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor):

        Returns:
            Tensor:
        """
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_sizes: list[int],
        encoder_layers: int,
        dropout: float,
    ) -> None:
        """_summary_

        Args:
            input_size (int): the input size of the encoder
            hidden_size (int): hidden size
            kernel_sizes (list[int]): the size of kernels
            encoder_layers (int): number of conv layers
            dropout (float): dropout ratio
        """
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                Conv1d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_sizes=kernel_sizes,
                )
                for i in range(encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """forward in encoder

        Args:
            x (Tensor): (batch_size, seq_len, embedding_dim)
            mask (Tensor): (batch_size, seq_len, 1)

        Returns:
            Tensor: _description_
        """
        # x (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        # mask (batch_size, 1, seq_len)
        mask = mask.transpose(1, 2)

        for i, encoder in enumerate(self.encoders):
            # fills elements of x with 0.0 where mask(padding) is False
            x.masked_fill_(~mask, 0.0)
            # using dropout
            if i > 0:
                x = self.dropout(x)
            #
            x = encoder(x)

        #
        x = self.dropout(x)
        #
        return x.transpose(1, 2)


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, activations: bool = True
    ) -> None:
        super().__init__()

        linear = nn.Linear(in_features, out_features)
        modules = [weight_norm(linear)]
        if activations:
            modules.append(GeLU())

        self.model = nn.Sequential(*modules)
        self.reset_parameters(activations)

    def reset_parameters(self, activations: bool) -> None:
        linear = self.model[0]
        nn.init.normal_(
            linear.weight,
            std=math.sqrt((2.0 if activations else 1.0) / linear.in_features),
        )
        nn.init.zeros_(linear.bias)

    def forward(self, x):
        return self.model(x)


class AugmentedResidualConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, emb_x, output_sum, i):
        if i > 1:
            output_sum *= math.sqrt(0.5)
        return torch.cat([emb_x, output_sum], dim=-1)


class Alignment(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, project_func: str
    ) -> None:
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(hidden_size)))

        if project_func != "identity":
            self.projection = nn.Sequential(
                nn.Dropout(dropout), Linear(input_size, hidden_size)
            )
        else:
            self.projection = nn.Identity()

    def forward(self, a: Tensor, b: Tensor, mask_a: Tensor, mask_b: Tensor):
        attn = (
            torch.matmul(self.projection(a), self.projection(b).transpose(1, 2))
            * self.temperature
        )

        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        mask = mask.bool()

        attn.masked_fill_(~mask, -1e7)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)

        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)

        return feature_a, feature_b


class RE2(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.embedding = Embedding(args.vocab_size, args.embedding_dim, args.dropout)

        self.connection = AugmentedResidualConnection()

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "encoder": Encoder(
                            args.embedding_dim,
                            args.hidden_size,
                            args.kernel_sizes,
                            args.encoder_layers,
                            args.dropout,
                        ),
                        "alignment": Alignment(
                            args.hidden_size,
                            args.hidden_size,
                            args.dropout,
                            args.project_func,
                        ),
                    }
                )
                for i in range(args.num_blocks)
            ]
        )

    def forward(self, a: Tensor, b: Tensor, mask_a: Tensor, mask_b: Tensor) -> Tensor:
        """
        Args:
            a (Tensor): (batch_size, seq_len)
            b (Tensor): (batch_size, seq_len)
            mask_a (Tensor): (batch_size, seq_len, 1)
            mask_b (Tensor): (batch_size, seq_len, 1)

        Returns:
            Tensor: (batch_size, num_classes)
        """
        # a (batch_size, seq_len, embedding_dim)
        a = self.embedding(a)
        # b (batch_size, seq_len, embedding_dim)
        b = self.embedding(b)

        # the outputs of embedding layer
        emb_a, emb_b = a, b
        # the outputs of upper two layer
        res_a, res_b = torch.zeros_like(a), torch.zeros_like(b)

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(emb_a, a + res_a, i)
                b = self.connection(emb_b, b + res_b, i)
                res_a, res_b = a, b
            #
            a_enc = block["encoder"](a, mask_a)
            #
            b_enc = block["encoder"](b, mask_b)
            # concating the input and output of encoder
            #
            a = torch.cat([a, a_enc], dim=-1)
            #
            b = torch.cat([b, b_enc], dim=-1)
            #
            align_a, align_b = block["alignment"](a, b, mask_a, mask_b)
