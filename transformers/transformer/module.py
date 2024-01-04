import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple
from dataclasses import asdict
from config import ModelArugment


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        """

        Args:
            vocab_size (int): size of vocabulary
            d_model (int): dimension of embeddings
        """
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.sqrt_d_model = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_length)

        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        # multiply embedding values by the square root of the embedding dimension
        return self.embed(x) * self.sqrt_d_model


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int = 512, dropout: float = 0.1, max_positions: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pe (max_positions, d_model)
        pe = torch.zeros(max_positions, d_model)
        # position (max_positions, 1)
        # create position column
        position = torch.arange(0, max_positions).unsqueeze(1)
        # div_term (d_model/2)
        # cauculate the divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # calculate sine values on even indices
        # position * div_term will be broadcast to (max_positions, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        # calculate cosine values on odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension: pe (1, max_positions, d_model)
        pe = pe.unsqueeze(0)
        # buffers will not be trained
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, d_model) embeddings

        Returns:
            Tensor: (batch_size, seq_len, d_model)
        """
        # x.size(1) is the max sequence length
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """

        Args:
            d_model (int, optional): dimension of embeddings. Defaults to 512.
            n_heads (int, optional): numer of self attention heads. Defaults to 8.
            dropout (float, optional): dropout ratio. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads  # dimension of every head

        self.q = nn.Linear(d_model, d_model, bias=False)  # query matrix
        self.k = nn.Linear(d_model, d_model, bias=False)  # key matrix
        self.v = nn.Linear(d_model, d_model, bias=False)  # value matrix
        self.concat = nn.Linear(d_model, d_model, bias=False)  # output

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: Tensor, is_key: bool = False) -> Tensor:
        batch_size = x.size(0)
        # x (batch_size, seq_len, n_heads, d_key)
        x = x.view(batch_size, -1, self.n_heads, self.d_key)
        if is_key:
            # (batch_size, n_heads, d_key, seq_len)
            return x.permute(0, 2, 3, 1)
        # (batch_size, n_heads, seq_len, d_key)
        return x.transpose(1, 2)

    def merge_heads(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        return x

    def attenion(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
        keep_attentions: bool = False,
    ):
        scores = torch.matmul(query, key) / math.sqrt(self.d_key)

        if mask is not None:
            # Fill those positions of product as -1e9 where mask positions are 0, because exp(-1e9) will get zero.
            # Note that we cannot set it to negative infinity, as there may be a situation where negative infinity is divided by negative infinity.
            scores = scores.masked_fill(mask == 0, -1e9)

        # weights (batch_size, n_heads, q_length, k_length)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)
        # assert k_length == v_length
        # attn_output (batch_size, n_heads, q_length, d_key)
        attn_output = torch.matmul(weights, value)

        if keep_attentions:
            self.weights = weights
        else:
            del weights

        return attn_output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            query (Tensor): (batch_size, q_length, d_model)
            key (Tensor): (batch_size, k_length, d_model)
            value (Tensor): (batch_size, v_length, d_model)
            mask (Tensor, optional): mask for padding or decoder. Defaults to None.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            output (Tensor): (batch_size, q_length, d_model) attention output
        """
        query, key, value = self.q(query), self.k(key), self.v(value)

        query, key, value = (
            self.split_heads(query),
            self.split_heads(key, is_key=True),
            self.split_heads(value),
        )

        attn_output = self.attenion(query, key, value, mask, keep_attentions)

        del query
        del key
        del value

        # Concat
        concat_output = self.merge_heads(attn_output)
        # the final liear
        # output (batch_size, q_length, d_model)
        output = self.concat(concat_output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_length, d_model)

        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """

        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            d_ff (int): dimension of feed-forward network
            dropout (float, optional): dropout ratio. Defaults to 0.1.
        """
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_length, d_model) output from attention

        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        return self.ff2(self.dropout(F.relu(self.ff1(x))))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool = False,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float): dropout ratio
            norm_first (bool): if True, layer norm is done prior to attention and feedforward operations(Pre-Norm).
                Otherwise it's done after(Post-Norm). Default to False.

        """
        super().__init__()

        self.norm_first = norm_first

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # self attention sub layer
    def _sa_sub_layer(
        self, x: Tensor, attn_mask: Tensor, keep_attentions: bool
    ) -> Tensor:
        x = self.attention(x, x, x, attn_mask, keep_attentions)
        return self.dropout1(x)

    def _ff_sub_layer(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout2(x)

    def forward(
        self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            src (Tensor): (batch_size, seq_length, d_model)
            src_mask (Tensor, optional): (batch_size,  1, seq_length)
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: (batch_size, seq_length, d_model) output of encoder block
        """
        # pass througth multi-head attention
        # src (batch_size, seq_length, d_model)
        # attn_score (batch_size, n_heads, seq_length, k_length)
        x = src
        if self.norm_first:
            x = x + self._sa_sub_layer(self.norm1(x), src_mask, keep_attentions)
            x = x + self._ff_sub_layer(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_sub_layer(x, src_mask, keep_attentions))
            x = self.norm2(x + self._ff_sub_layer(x))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_layers (int): number of encoder blocks
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float, optional): dropout ratio. Defaults to 0.1.
        """
        super().__init__()
        # stack n_layers encoder blocks
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout, norm_first)
                for _ in range(n_layers)
            ]
        )

        self.norm = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tensor:
        """

        Args:
            src (Tensor): (batch_size, seq_length, d_model)
            src_mask (Tensor, optional): (batch_size, 1, seq_length)
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, seq_length, d_model)
        """
        x = src
        # pass through each layer
        for layer in self.layers:
            x = layer(x, src_mask, keep_attentions)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool = False,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float): dropout ratio
            norm_first (bool): if True, layer norm is done prior to attention and feedforward operations(Pre-Norm).
                Otherwise it's done after(Post-Norm). Default to False.
        """
        super().__init__()
        self.norm_first = norm_first
        # masked multi-head attention
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        # cross multi-head attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        # position-wise feed-forward network
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    # self attention sub layer
    def _sa_sub_layer(
        self, x: Tensor, attn_mask: Tensor, keep_attentions: bool
    ) -> Tensor:
        x = self.masked_attention(x, x, x, attn_mask, keep_attentions)
        return self.dropout1(x)

    # cross attention sub layer
    def _ca_sub_layer(
        self, x: Tensor, mem: Tensor, attn_mask: Tensor, keep_attentions: bool
    ) -> Tensor:
        x = self.cross_attention(x, mem, mem, attn_mask, keep_attentions)
        return self.dropout2(x)

    def _ff_sub_layer(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout3(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Args:
            tgt (Tensor):   (batch_size, tgt_seq_length, d_model) the (target) sequence to the decoder block.
            memory (Tensor):  (batch_size, src_seq_length, d_model) the sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional):  (batch_size, 1, tgt_seq_length, tgt_seq_length) the mask for the tgt sequence.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_length) the mask for the memory sequence.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            tgt (Tensor): (batch_size, tgt_seq_length, d_model) output of decoder block
        """

        # pass througth masked multi-head attention
        # tgt_ (batch_size, tgt_seq_length, d_model)
        # masked_attn_score (batch_size, n_heads, tgt_seq_length, tgt_seq_length)
        x = tgt
        if self.norm_first:
            x = x + self._sa_sub_layer(self.norm1(x), tgt_mask, keep_attentions)
            x = x + self._ca_sub_layer(
                self.norm2(x), memory, memory_mask, keep_attentions
            )
            x = x + self._ff_sub_layer(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_sub_layer(x, tgt_mask, keep_attentions))
            x = self.norm2(
                x + self._ca_sub_layer(x, memory, memory_mask, keep_attentions)
            )
            x = self.norm3(x + self._ff_sub_layer(x))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_layers (int): number of encoder blocks
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float, optional): dropout ratio. Defaults to 0.1.
        """
        super().__init__()
        # stack n_layers decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, d_ff, dropout, norm_first)
                for _ in range(n_layers)
            ]
        )

        self.norm = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tensor:
        """

        Args:
            tgt (Tensor): (batch_size, tgt_seq_length, d_model) the (target) sequence to the decoder.
            memory (Tensor):  (batch_size, src_seq_length, d_model) the  sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional):  (batch_size, 1, tgt_seq_length, tgt_seq_length) the mask for the tgt sequence.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_length) the mask for the memory sequence.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: (batch_size, tgt_seq_length, d_model) model output (logits)
        """
        x = tgt
        # pass through each layer
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask, keep_attentions)

        x = self.norm(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_positions: int = 5000,
        pad_idx: int = 0,
        norm_first: bool = False,
    ) -> None:
        """

        Args:
            source_vocab_size (int): size of the source vocabulary.
            target_vocab_size (int): size of the target vocabulary.
            d_model (int, optional): dimension of embeddings. Defaults to 512.
            n_heads (int, optional): number of heads. Defaults to 8.
            num_encoder_layers (int, optional): number of encoder blocks. Defaults to 6.
            num_decoder_layers (int, optional): number of decoder blocks. Defaults to 6.
            d_ff (int, optional): dimension of inner feed-forward network. Defaults to 2048.
            dropout (float, optional): dropout ratio. Defaults to 0.1.
            max_positions (int, optional): maximum sequence length for positional encoding. Defaults to 5000.
            pad_idx (int, optional): pad index. Defaults to 0.
            norm_first (bool): if True, layer norm is done prior to attention and feedforward operations(Pre-Norm).
                Otherwise it's done after(Post-Norm). Default to False.
        """
        super().__init__()
        self.src_embedding = Embedding(source_vocab_size, d_model)
        self.tgt_embedding = Embedding(target_vocab_size, d_model)

        self.enc_pos = PositionalEncoding(d_model, dropout, max_positions)
        self.dec_pos = PositionalEncoding(d_model, dropout, max_positions)

        self.encoder = Encoder(
            d_model, num_encoder_layers, n_heads, d_ff, dropout, norm_first
        )
        self.decoder = Decoder(
            d_model, num_decoder_layers, n_heads, d_ff, dropout, norm_first
        )

        self.pad_idx = pad_idx

    def encode(
        self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tensor:
        """

        Args:
            src (Tensor): (batch_size, src_seq_length) the sequence to the encoder
            src_mask (Tensor, optional): (batch_size, 1, src_seq_length) the mask for the sequence
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, seq_length, d_model) encoder output
        """
        # src_embed (batch_size, src_seq_length, d_model)
        src_embed = self.enc_pos(self.src_embedding(src))
        return self.encoder(src_embed, src_mask, keep_attentions)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tensor:
        """

        Args:
            tgt (Tensor):  (batch_size, tgt_seq_length) the sequence to the decoder.
            memory (Tensor): (batch_size, src_seq_length, d_model) the  sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional): (batch_size, 1, 1, tgt_seq_length) the mask for the target sequence. Defaults to None.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_length) the mask for the memory sequence. Defaults to None.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: output (batch_size, tgt_seq_length, tgt_vocab_size)
        """
        # tgt_embed (batch_size, tgt_seq_length, d_model)
        tgt_embed = self.dec_pos(self.tgt_embedding(tgt))
        # logits (batch_size, tgt_seq_length, d_model)
        logits = self.decoder(tgt_embed, memory, tgt_mask, memory_mask, keep_attentions)

        return logits

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tensor:
        """

        Args:
            src (Tensor): (batch_size, src_seq_length) the sequence to the encoder
            tgt (Tensor):  (batch_size, tgt_seq_length) the sequence to the decoder
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, tgt_seq_length, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask, keep_attentions)
        return self.decode(tgt, memory, tgt_mask, src_mask, keep_attentions)


class TranslationHead(nn.Module):
    def __init__(
        self, config: ModelArugment, pad_idx: int, bos_idx: int, eos_idx: int
    ) -> None:
        super().__init__()
        self.config = config

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.transformer = Transformer(**asdict(config))
        self.lm_head = nn.Linear(config.d_model, config.target_vocab_size, bias=False)
        self.reset_parameters()

    @staticmethod
    def generate_subsequent_mask(seq_len: int, device: torch.device) -> Tensor:
        """
        Subsequent mask(seq_len=6):
            [[1, 0, 0, 0, 0, 0]]
            [[1, 1, 0, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]]
            [[1, 1, 1, 1, 0, 0]]
            [[1, 1, 1, 1, 1, 0]]
            [[1, 1, 1, 1, 1, 1]]
        """
        subseq_mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=device)
        ).int()
        return subseq_mask

    @staticmethod
    def create_masks(
        src: Tensor, tgt: Tensor = None, pad_idx: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """make mask tensor for src and target sequences

        Args:
            src (Tensor): (batch_size, src_seq_length)  raw source sequences with padding
            tgt (Tensor): (batch_size, tgt_seq_length)  raw target sequences with padding

        Returns:
            Tensor: (batch_size, 1, 1, src_seq_length) src mask
            Tensor(optional): (batch_size, 1, tgt_seq_length, tgt_seq_length) tgt mask
        """
        # pad mask
        # src_mask  (batch_size, 1, 1, src_seq_length)
        src_mask = (src != pad_idx).int().unsqueeze(1).unsqueeze(2)

        tgt_mask = None

        if tgt is not None:
            tgt_seq_len = tgt.size()[-1]
            # pad mask
            # tgt_mask  (batch_size, 1, 1, tgt_seq_length)
            tgt_mask = (tgt != pad_idx).int().unsqueeze(1).unsqueeze(2)

            # subsequcen mask
            # subseq_mask  (1, 1, tgt_seq_length, tgt_seq_length)
            subseq_mask = TranslationHead.generate_subsequent_mask(
                tgt_seq_len, src.device
            )

            """
            PAD mask             &  Subsequent mask     => Target mask
            [[1, 1, 1, 0, 0, 0]] & [[1, 0, 0, 0, 0, 0]] => [[1, 0, 0, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]] & [[1, 1, 0, 0, 0, 0]] => [[1, 1, 0, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 0, 0, 0]] => [[1, 1, 1, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 0, 0]] => [[1, 1, 1, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 1, 0]] => [[1, 1, 1, 0, 0, 0]]
            [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 1, 1]] => [[1, 1, 1, 0, 0, 0]]

            """
            # tgt_mask (batch_size, 1, tgt_seq_len, tgt_seq_len)
            tgt_mask = tgt_mask & subseq_mask

        return src_mask, tgt_mask

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
        keep_attentions: bool = False,
    ) -> Tensor:
        if src_mask is None and tgt_mask is None:
            src_mask, tgt_mask = self.create_masks(src, tgt, self.pad_idx)
        output = self.transformer(src, tgt, src_mask, tgt_mask, keep_attentions)

        return self.lm_head(output)

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        src_mask: Tensor = None,
        max_gen_len: int = 60,
        num_beams: int = 3,
        keep_attentions: bool = False,
        generation_mode: str = "greedy_search",
    ):
        if src_mask is None:
            src_mask = self.create_masks(src, pad_idx=self.pad_idx)[0]
        generation_mode = generation_mode.lower()
        if generation_mode == "greedy_search":
            return self._greedy_search(src, src_mask, max_gen_len, keep_attentions)
        else:
            return self._beam_search(
                src, src_mask, max_gen_len, num_beams, keep_attentions
            )

    def reset_parameters(self) -> None:
        # initrange = 0.1
        # self.src_embedding.embed.weight.data.uniform_(-initrange, initrange)
        # self.tgt_embedding.embed.weight.data.uniform_(-initrange, initrange)
        # self.lm_head.weight.data.uniform_(-initrange, initrange)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _beam_search(
        self,
        src: Tensor,
        src_mask: Tensor,
        max_gen_len: int,
        num_beams: int,
        keep_attentions: bool,
    ):
        raise NotImplementedError()

    def _greedy_search(
        self, src: Tensor, src_mask: Tensor, max_gen_len: int, keep_attentions: bool
    ):
        memory = self.transformer.encode(src, src_mask)

        batch_size = src.shape[0]

        device = src.device

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        decoder_inputs = torch.LongTensor(batch_size, 1).fill_(self.bos_idx).to(device)

        eos_idx_tensor = torch.tensor([self.eos_idx]).to(device)

        finished = False

        while True:
            tgt_mask = self.generate_subsequent_mask(decoder_inputs.size(1), device)

            logits = self.lm_head(
                self.transformer.decode(
                    decoder_inputs,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=src_mask,
                    keep_attentions=keep_attentions,
                )
            )

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

            # finished sentences should have their next token be a pad token
            next_tokens = next_tokens * unfinished_sequences + self.pad_idx * (
                1 - unfinished_sequences
            )

            decoder_inputs = torch.cat([decoder_inputs, next_tokens[:, None]], dim=-1)

            # set sentence to finished if eos_idx was found
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_idx_tensor.shape[0], 1)
                .ne(eos_idx_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # all sentences have eos_idx
            if unfinished_sequences.max() == 0:
                finished = True

            if decoder_inputs.shape[-1] >= max_gen_len:
                finished = True

            if finished:
                break

        return decoder_inputs
