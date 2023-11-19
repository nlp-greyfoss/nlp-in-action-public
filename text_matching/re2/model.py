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
            kernel_sizes (list[int]): the size of kernel
        """
        super().__init__()

        out_channels = out_channels // len(kernel_sizes)

        convs = []
        # L_in is seq_len, L_out is output_dim of conv
        # L_out = (L_in + 2 * padding - kernel_size + 1)
        # and padding=(kernel_size - 1) // 2
        # L_out = (L_in + kernel_size - 1 - kernel_size + 1) = L_in
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            )
            convs.append(nn.Sequential(weight_norm(conv), GeLU()))
        # output shape of each conv is (batch_size, out_channels(new), seq_len)

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
            x (Tensor): shape (batch_size, embedding_dim, seq_len)

        Returns:
            Tensor:
        """
        # back to (batch_size, out_channels, seq_len)
        return torch.cat([encoder(x) for encoder in self.model], dim=1)


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
            input_size (int): embedding_dim or embedding_dim + hidden_size
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
            x (Tensor): (batch_size, seq_len, input_size)
            mask (Tensor): (batch_size, seq_len, 1)

        Returns:
            Tensor: _description_
        """
        # x (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        # mask (batch_size, 1, seq_len)
        mask = mask.transpose(1, 2)

        for i, encoder in enumerate(self.encoders):
            # fills elements of x with 0.0 where mask is False
            x.masked_fill_(~mask, 0.0)
            # using dropout
            if i > 0:
                x = self.dropout(x)
            # returned x (batch_size, hidden_size, seq_len)
            x = encoder(x)

        # apply dropout
        x = self.dropout(x)
        # (batch_size, seq_len, hidden_size)
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

    def forward(self, x: Tensor, res: Tensor, i: int) -> Tensor:
        """

        Args:
            x (Tensor): the output of pre block (batch_size, seq_len, hidden_size)
            res (Tensor): (batch_size, seq_len, embedding_size) or (batch_size, seq_len, embedding_size + hidden_size)
                res[:,:,hidden_size:] is the output of Embedding layer
                res[:,:,:hidden_size] is the output of previous two block
            i (int): layer index

        Returns:
            Tensor: (batch_size, seq_len,  hidden_size  + embedding_size)
        """
        if i == 1:
            # (batch_size, seq_len,  hidden_size  + embedding_size)
            return torch.cat([x, res], dim=-1)
        hidden_size = x.size(-1)
        # (res[:, :, :hidden_size] + x) is the summation of the output of previous two blocks
        # x (batch_size, seq_len, hidden_size)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        # (batch_size, seq_len,  hidden_size  + embedding_size)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)


class Alignment(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, project_func: str
    ) -> None:
        """

        Args:
            input_size (int): embedding_dim  + hidden_size  or embedding_dim  + hidden_size * 2
            hidden_size (int): hidden size
            dropout (float): dropout ratio
            project_func (str): identity or linear
        """
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(hidden_size)))

        if project_func != "identity":
            self.projection = nn.Sequential(
                nn.Dropout(dropout), Linear(input_size, hidden_size)
            )
        else:
            self.projection = nn.Identity()

    def forward(self, a: Tensor, b: Tensor, mask_a: Tensor, mask_b: Tensor) -> Tensor:
        """

        Args:
            a (Tensor): (batch_size, seq_len, input_size)
            b (Tensor): (batch_size, seq_len, input_size)
            mask_a (Tensor):  (batch_size, seq_len, 1)
            mask_b (Tensor):  (batch_size, seq_len, 1)

        Returns:
            Tensor: _description_
        """
        # if projection == 'linear' : self.projection(*) -> (batch_size, seq_len,  hidden_size) -> transpose(*) -> (batch_size, hidden_size,  seq_len)
        # if projection == 'identity' : self.projection(*) -> (batch_size, seq_len, input_size) -> transpose(*) -> (batch_size, input_size,  seq_len)
        # attn (batch_size, seq_len_a,  seq_len_b)
        attn = (
            torch.matmul(self.projection(a), self.projection(b).transpose(1, 2))
            * self.temperature
        )
        # mask (batch_size, seq_len_a, seq_len_b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        mask = mask.bool()
        # fills elements of x with 0.0(after exp) where mask is False
        attn.masked_fill_(~mask, -1e7)
        # attn_a (batch_size, seq_len_a,  seq_len_b)
        attn_a = F.softmax(attn, dim=1)
        # attn_b (batch_size, seq_len_a,  seq_len_b)
        attn_b = F.softmax(attn, dim=2)
        # feature_b  (batch_size, seq_len_b,  seq_len_a) x (batch_size, seq_len_a, input_size)
        # -> (batch_size, seq_len_b,  input_size)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        # feature_a  (batch_size, seq_len_a,  seq_len_b) x (batch_size, seq_len_b, input_size)
        # -> (batch_size, seq_len_a,  input_size)
        feature_a = torch.matmul(attn_b, b)

        return feature_a, feature_b


class Fusion(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        """

        Args:
            input_size (int): embedding_dim  + hidden_size  or embedding_dim  + hidden_size * 2
            hidden_size (int): hidden size
            dropout (float): dropout ratio
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fusion1 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion = Linear(hidden_size * 3, hidden_size, activations=True)

    def forward(self, x: Tensor, align: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): input (batch_size, seq_len, input_size)
            align (Tensor): output of Alignment (batch_size, seq_len,  input_size)

        Returns:
            Tensor: (batch_size, seq_len, hidden_size)
        """
        # x1 (batch_size, seq_len, hidden_size)
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        # x2 (batch_size, seq_len, hidden_size)
        x2 = self.fusion1(torch.cat([x, x - align], dim=-1))
        # x3 (batch_size, seq_len, hidden_size)
        x3 = self.fusion1(torch.cat([x, x * align], dim=-1))
        # x (batch_size, seq_len, hidden_size * 3)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.dropout(x)
        # (batch_size, seq_len, hidden_size)
        return self.fusion(x)


class Pooling(nn.Module):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, hidden_size)
            mask (Tensor): (batch_size, seq_len, 1)

        Returns:
            Tensor: (batch_size, hidden_size)
        """
        # max returns a namedtuple (values, indices), we only need values
        return x.masked_fill(~mask, -float("inf")).max(dim=1)[0]


class Prediction(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            Linear(hidden_size * 4, hidden_size, activations=True),
            nn.Dropout(dropout),
            Linear(hidden_size, num_classes),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """

        Args:
            a (Tensor): (batch_size, hidden_size)
            b (Tensor): (batch_size, hidden_size)

        Returns:
            Tensor: (batch_size, num_classes)
        """
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


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
                            args.embedding_dim
                            if i == 0
                            else args.embedding_dim + args.hidden_size,
                            args.hidden_size,
                            args.kernel_sizes,
                            args.encoder_layers,
                            args.dropout,
                        ),
                        "alignment": Alignment(
                            args.embedding_dim + args.hidden_size
                            if i == 0
                            else args.embedding_dim + args.hidden_size * 2,
                            args.hidden_size,
                            args.dropout,
                            args.project_func,
                        ),
                        "fusion": Fusion(
                            args.embedding_dim + args.hidden_size
                            if i == 0
                            else args.embedding_dim + args.hidden_size * 2,
                            args.hidden_size,
                            args.dropout,
                        ),
                    }
                )
                for i in range(args.num_blocks)
            ]
        )

        self.pooling = Pooling()
        self.prediction = Prediction(args.hidden_size, args.num_classes, args.dropout)

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

        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                # a (batch_size, seq_len, embedding_dim + hidden_size)
                a = self.connection(a, res_a, i)
                # b (batch_size, seq_len, embedding_dim + hidden_size)
                b = self.connection(b, res_b, i)
                # now embeddings saved to res_a[:,:,hidden_size:]
                res_a, res_b = a, b
            # a_enc (batch_size, seq_len, hidden_size)
            a_enc = block["encoder"](a, mask_a)
            # b_enc (batch_size, seq_len, hidden_size)
            b_enc = block["encoder"](b, mask_b)
            # concating the input and output of encoder
            # a (batch_size, seq_len, embedding_dim + hidden_size or embedding_dim + hidden_size * 2)
            a = torch.cat([a, a_enc], dim=-1)
            # b (batch_size, seq_len, embedding_dim + hidden_size or embedding_dim + hidden_size * 2)
            b = torch.cat([b, b_enc], dim=-1)
            # align_a (batch_size, seq_len,  embedding_dim + hidden_size or embedding_dim + hidden_size * 2)
            # align_b (batch_size, seq_len,  embedding_dim + hidden_size or embedding_dim + hidden_size * 2)
            align_a, align_b = block["alignment"](a, b, mask_a, mask_b)
            # a (batch_size, seq_len,  hidden_size)
            a = block["fusion"](a, align_a)
            # b (batch_size, seq_len,  hidden_size)
            b = block["fusion"](b, align_b)
        # a (batch_size, hidden_size)
        a = self.pooling(a, mask_a)
        # b (batch_size, hidden_size)
        b = self.pooling(b, mask_b)
        # (batch_size, num_classes)
        return self.prediction(a, b)
