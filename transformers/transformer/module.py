import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Tuple, Union, Callable
from dataclasses import asdict
from config import ModelArugment

from utils import BeamSearchScorer, BeamHypotheses


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
            x (Tensor): (batch_size, seq_len)

        Returns:
            Tensor: (batch_size, seq_len, d_model)
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

    def forward(self, x: Tensor, position_ids: Union[int, list[int]] = None) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, d_model) embeddings
            position_ids (Union[int, list[int]]): singe position id or list

        Returns:
            Tensor: (batch_size, seq_len, d_model)
        """
        if position_ids is None:
            position_ids = range(x.size(1))
        return self.dropout(x + self.pe[:, position_ids, :])


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        is_decoder: bool = False,
    ) -> None:
        """

        Args:
            d_model (int, optional): dimension of embeddings. Defaults to 512.
            n_heads (int, optional): numer of self attention heads. Defaults to 8.
            dropout (float, optional): dropout ratio. Defaults to 0.1.
            bias (bool, optional):If True, the attention linear layer add an additive bias. Default to True.

        """
        super().__init__()
        assert d_model % n_heads == 0
        self.is_decoder = is_decoder
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads  # dimension of every head

        self.q = nn.Linear(d_model, d_model, bias=bias)  # query matrix
        self.k = nn.Linear(d_model, d_model, bias=bias)  # key matrix
        self.v = nn.Linear(d_model, d_model, bias=bias)  # value matrix
        self.concat = nn.Linear(d_model, d_model, bias=bias)  # output

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

        # weights (batch_size, n_heads, q_len, k_len)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        # (batch_size, n_heads, q_len, k_len) x (batch_size, n_heads, v_len, d_key) -> (batch_size, n_heads, q_len, d_key)
        # assert k_len == v_len
        # attn_output (batch_size, n_heads, q_len, d_key)
        attn_output = torch.matmul(weights, value)

        if keep_attentions:
            self.weights = weights
        else:
            del weights

        return attn_output

    def _transform_and_split(
        self, transform: Callable, x: Tensor, is_key: bool = False
    ) -> Tensor:
        x = transform(x)
        return self.split_heads(x, is_key)

    def _concat_key_value(
        self, key: Tensor, value: Tensor, past_key_value: Tuple[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if past_key_value is not None:
            key_, value_ = past_key_value
            # append current key and value
            # (batch_size, n_heads, d_key, k_len)
            key = torch.cat([key_, key], dim=-1)
            # (batch_size, n_heads, v_len, d_key)
            value = torch.cat([value_, value], dim=2)
        return key, value

    def forward(
        self,
        query: Tensor,
        key_value: Tensor = None,
        mask: Tensor = None,
        past_key_value: Tuple[Tensor] = None,
        use_cache: bool = False,
        keep_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            query (Tensor): (batch_size, q_len, d_model)
            key_value (Tensor, optional): (batch_size, k_len/v_len, d_model) key and value are same.
            mask (Tensor, optional): mask for padding or decoder. Defaults to None.
            past_key_value (Tuple[Tensor], optional): cached past key and value states. Defaults to None.
            use_cache (bool, optional): whether to use kv cache during inference. Defaults to False.
            keep_attentions (bool): whether to keep attention weigths or not. Defaults to False.

        Returns:
            output (Tensor): (batch_size, q_len, d_model) attention output
            present_key_value (Tuple[Tensor], optional): Cached present key and value states
        """

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"

        is_self_attention = key_value is None

        _query = query

        query = self._transform_and_split(self.q, query)

        if is_self_attention:
            # the 'self' attention
            key = self._transform_and_split(self.k, _query, is_key=True)
            value = self._transform_and_split(self.v, _query)
            key, value = self._concat_key_value(key, value, past_key_value)
        elif past_key_value is None:
            # the cross attention, key_value is memory
            key = self._transform_and_split(self.k, key_value, is_key=True)
            value = self._transform_and_split(self.v, key_value)
        else:
            # if is_self_attention == False and past_key_value is not None
            # key_value is memory and use cache(past_key_value not None) we do not need to calculate the key and value again because it was cached.
            # since memory will not change during inference.
            key, value = past_key_value

        if self.is_decoder and use_cache:
            # cache newest key and value
            present_key_value = (key, value)
        else:
            present_key_value = None

        attn_output = self.attenion(query, key, value, mask, keep_attentions)

        # Concat
        concat_output = self.merge_heads(attn_output)
        # the final liear
        # output (batch_size, q_len, d_model)
        output = self.concat(concat_output)

        return output, present_key_value


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, d_model)

        Returns:
            Tensor: (batch_size, seq_len, d_model)
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
            x (Tensor): (batch_size, seq_len, d_model) output from attention

        Returns:
            Tensor: (batch_size, seq_len, d_model)
        """
        return self.ff2(self.dropout(F.relu(self.ff1(x))))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention_bias: bool = True,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float): dropout ratio

        """
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, attention_bias)
        self.norm1 = LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # self attention sub layer
    def _sa_sub_layer(
        self, x: Tensor, attn_mask: Tensor, keep_attentions: bool
    ) -> Tensor:
        x, _ = self.attention(query=x, mask=attn_mask, keep_attentions=keep_attentions)
        return self.dropout1(x)

    def _ff_sub_layer(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout2(x)

    def forward(
        self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            src (Tensor): (batch_size, seq_len, d_model)
            src_mask (Tensor, optional): (batch_size,  1, seq_len)
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: (batch_size, seq_len, d_model) output of encoder block
        """
        # src (batch_size, seq_len, d_model)
        x = src
        x = x + self._sa_sub_layer(self.norm1(x), src_mask, keep_attentions)
        x = x + self._ff_sub_layer(self.norm2(x))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_bias: bool = True,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_layers (int): number of encoder blocks
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float, optional): dropout ratio. Defaults to 0.1.
            attention_bias (bool):If True, the attention linear layer add an additive bias. Default to True.
        """
        super().__init__()
        # stack n_layers encoder blocks
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout, attention_bias)
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
            src (Tensor): (batch_size, seq_len, d_model)
            src_mask (Tensor, optional): (batch_size, 1, seq_len)
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, seq_len, d_model)
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
        attention_bias: bool = True,
    ) -> None:
        """

        Args:
            d_model (int): dimension of embeddings
            n_heads (int): number of heads
            d_ff (int): dimension of inner feed-forward network
            dropout (float): dropout ratio

        """
        super().__init__()
        # masked multi-head attention
        self.masked_attention = MultiHeadAttention(
            d_model, n_heads, dropout, attention_bias, is_decoder=True
        )
        self.norm1 = LayerNorm(d_model)
        # cross multi-head attention
        self.cross_attention = MultiHeadAttention(
            d_model, n_heads, dropout, attention_bias, is_decoder=True
        )
        self.norm2 = LayerNorm(d_model)
        # position-wise feed-forward network
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    # self attention sub layer
    def _sa_sub_layer(
        self,
        x: Tensor,
        attn_mask: Tensor,
        past_key_value: Tensor,
        use_cache: bool,
        keep_attentions: bool,
    ) -> Tensor:
        residual = x
        x, present_key_value = self.masked_attention(
            query=self.norm1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            mask=attn_mask,
            keep_attentions=keep_attentions,
        )
        x = self.dropout1(x) + residual
        return x, present_key_value

    # cross attention sub layer
    def _ca_sub_layer(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor,
        past_key_value: Tensor,
        use_cache: bool,
        keep_attentions: bool,
    ) -> Tensor:
        residual = x
        x, present_key_value = self.cross_attention(
            query=self.norm2(x),
            key_value=mem,
            mask=attn_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            keep_attentions=keep_attentions,
        )
        x = self.dropout2(x) + residual
        return x, present_key_value

    def _ff_sub_layer(self, x: Tensor) -> Tensor:
        residual = x
        x = self.ff(self.norm3(x))
        return self.dropout3(x) + residual

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        past_key_value: Tuple[Tensor] = None,
        use_cache: bool = True,
        keep_attentions: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            tgt (Tensor):   (batch_size, tgt_seq_len, d_model) the (target) sequence to the decoder block.
            memory (Tensor):  (batch_size, src_seq_len, d_model) the sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional):  (batch_size, 1, tgt_seq_len, tgt_seq_len) the mask for the tgt sequence.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_len) the mask for the memory sequence.
            past_key_values (Tuple[Tensor], optional): the cached key and value states. Defaults to None.
            use_cache (bool, optional): whether use kv cache during inference or not. Defaults to False.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            tgt (Tensor): (batch_size, tgt_seq_len, d_model) output of decoder block
        """
        if past_key_value is not None:
            # first two elements in the past_key_value tuple are self-attention
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value = None
            cross_attn_past_key_value = None

        x = tgt
        self_attn_outputs = self._sa_sub_layer(
            x,
            tgt_mask,
            self_attn_past_key_value,
            use_cache,
            keep_attentions,
        )
        # self attention output and present key value state
        x, present_key_value_state = self_attn_outputs

        cross_attn_outputs = self._ca_sub_layer(
            x,
            memory,
            memory_mask,
            cross_attn_past_key_value,
            use_cache,
            keep_attentions,
        )

        x = cross_attn_outputs[0]
        if present_key_value_state is not None:
            # append the cross-attention key and value states to present key value states
            present_key_value_state = present_key_value_state + cross_attn_outputs[1]

        x = self._ff_sub_layer(x)

        return x, present_key_value_state


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_bias: bool = True,
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
                DecoderBlock(d_model, n_heads, d_ff, dropout, attention_bias)
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
        past_key_values: Tuple[Tensor] = None,
        use_cache: bool = False,
        keep_attentions: bool = False,
    ) -> Tensor:
        """

        Args:
            tgt (Tensor): (batch_size, tgt_seq_len, d_model) the (target) sequence to the decoder.
            memory (Tensor):  (batch_size, src_seq_len, d_model) the  sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional):  (batch_size, 1, tgt_seq_len, tgt_seq_len) the mask for the tgt sequence.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_len) the mask for the memory sequence.
            past_key_values (Tuple[Tensor], optional): the cached key and value states. Defaults to None.
            use_cache (bool, optional): whether use kv cache during inference or not. Defaults to False.
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: (batch_size, tgt_seq_len, d_model) model output (logits)
        """
        present_key_value_states = () if use_cache else None

        x = tgt
        # pass through each layer
        for layer, past_key_value in zip(self.layers, past_key_values):
            x, present_key_value_state = layer(
                x,
                memory,
                tgt_mask,
                memory_mask,
                past_key_value,
                use_cache,
                keep_attentions,
            )
            if use_cache:
                # we cache the key and value states for each layer.
                # the length of present_key_value_states equals to the number of layers
                present_key_value_states += (present_key_value_state,)

        x = self.norm(x)

        return x, present_key_value_states


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
        attention_bias: bool = True,
        pad_idx: int = 0,
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
            attention_bias (bool):If True, the attention linear layer add an additive bias. Default to True.
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
            d_model, num_encoder_layers, n_heads, d_ff, dropout, attention_bias
        )
        self.decoder = Decoder(
            d_model, num_decoder_layers, n_heads, d_ff, dropout, attention_bias
        )

        self.pad_idx = pad_idx

    def encode(
        self, src: Tensor, src_mask: Tensor = None, keep_attentions: bool = False
    ) -> Tensor:
        """

        Args:
            src (Tensor): (batch_size, src_seq_len) the sequence to the encoder
            src_mask (Tensor, optional): (batch_size, 1, src_seq_len) the mask for the sequence
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, seq_len, d_model) encoder output
        """
        # src_embed (batch_size, src_seq_len, d_model)
        src_embed = self.enc_pos(self.src_embedding(src))
        return self.encoder(src_embed, src_mask, keep_attentions)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, tgt_len: int = None):
        """
        Expands target attention mask from (batch_size, src_len) to (batch_size, 1, tgt_len, src_len).
        """
        batch_size, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len)

        return expanded_mask

    def _create_target_mask(self, tgt_mask, seq_len, tgt):
        return self._expand_mask(tgt_mask, seq_len).to(tgt.device)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        past_key_values: Tuple[Tensor] = None,
        use_cache: bool = False,
        keep_attentions: bool = False,
    ) -> Tensor:
        """

        Args:
            tgt (Tensor):  (batch_size, tgt_seq_len) the sequence to the decoder.
            memory (Tensor): (batch_size, src_seq_len, d_model) the  sequence from the last layer of the encoder.
            tgt_mask (Tensor, optional): (batch_size, 1, 1, tgt_seq_len) the mask for the target sequence. Defaults to None.
            memory_mask (Tensor, optional): (batch_size, 1, 1, src_seq_len) the mask for the memory sequence. Defaults to None.
            past_key_values (Tuple[Tensor], optional): the cached key and value states. Defaults to None.
            use_cache (bool, optional): whether use kv cache during inference or not. Defaults to False.
            keep_attentions (bool, optional): whether keep attention weigths or not. Defaults to False.

        Returns:
            Tensor: output (batch_size, tgt_seq_len, tgt_vocab_size)
        """

        if past_key_values is None:
            past_key_values = [None] * len(self.decoder.layers)
            position_ids = None
        else:
            # when use_cache we only care about the current position.
            # we set position_ids to the seq_len of value from cache because the seq_len of tgt is 1, and the position id starts from 0.
            position_ids = past_key_values[0][1].size(2)

        tgt_embed = self.dec_pos(self.tgt_embedding(tgt), position_ids)
        # logits (batch_size, tgt_seq_len, d_model)
        logits, past_key_values = self.decoder(
            tgt_embed,
            memory,
            tgt_mask,
            memory_mask,
            past_key_values,
            use_cache,
            keep_attentions,
        )

        return logits, past_key_values

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
            src (Tensor): (batch_size, src_seq_len) the sequence to the encoder
            tgt (Tensor):  (batch_size, tgt_seq_len) the sequence to the decoder
            keep_attentions (bool): whether keep attention weigths or not. Defaults to False.


        Returns:
            Tensor: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask, keep_attentions)
        return self.decode(
            tgt, memory, tgt_mask, src_mask, keep_attentions=keep_attentions
        )[0]


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
            src (Tensor): (batch_size, src_seq_len)  raw source sequences with padding
            tgt (Tensor): (batch_size, tgt_seq_len)  raw target sequences with padding

        Returns:
            Tensor: (batch_size, 1, 1, src_seq_len) src mask
            Tensor(optional): (batch_size, 1, tgt_seq_len, tgt_seq_len) tgt mask
        """
        # pad mask
        # src_mask  (batch_size, 1, 1, src_seq_len)
        # src_mask = (src != pad_idx).int().unsqueeze(1).unsqueeze(2)
        src_mask = src.ne(pad_idx).long().unsqueeze(1).unsqueeze(2)

        tgt_mask = None

        if tgt is not None:
            tgt_seq_len = tgt.size()[-1]
            # pad mask
            # tgt_mask  (batch_size, 1, 1, tgt_seq_len)
            tgt_mask = tgt.ne(pad_idx).long().unsqueeze(1).unsqueeze(2)

            # subsequcen mask
            # subseq_mask  (1, 1, tgt_seq_len, tgt_seq_len)
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
        num_beams: int = 5,
        keep_attentions: bool = False,
        use_cache: bool = True,
        generation_mode: str = "beam_search",
    ):
        if src_mask is None:
            src_mask = self.create_masks(src, pad_idx=self.pad_idx)[0]
        generation_mode = generation_mode.lower()
        if generation_mode == "greedy_search":
            return self._greedy_search(
                src, src_mask, max_gen_len, use_cache, keep_attentions
            )
        else:
            return self._beam_search(
                src, src_mask, max_gen_len, num_beams, use_cache, keep_attentions
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
        use_cache: bool,
        keep_attentions: bool,
    ):
        # memory (batch_size, seq_len, d_model)
        memory = self.transformer.encode(src, src_mask)

        batch_size = memory.size(0)

        memory = memory.repeat_interleave(num_beams, dim=0)
        src_mask = src_mask.repeat_interleave(num_beams, dim=0)

        device = src.device

        batch_beam_size = memory.size(0)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_gen_len,
            num_beams=num_beams,
            device=device,
        )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        beam_scores[:, 1:] = -1e9

        beam_scores = beam_scores.view((batch_beam_size,))

        decoder_inputs = (
            torch.LongTensor(batch_beam_size, 1).fill_(self.bos_idx).to(device)
        )
        input_ids = decoder_inputs

        past_key_values = None
        tgt_mask = None

        while True:
            if not use_cache:
                tgt_mask = self.generate_subsequent_mask(decoder_inputs.size(1), device)

            outputs = self.transformer.decode(
                input_ids,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                keep_attentions=keep_attentions,
            )
            # logits (batch_beam_size, seq_len, vocab_size)
            logits = self.lm_head(outputs[0])

            past_key_values = outputs[1]

            # next_token_logits (batch_beam_size, vocab_size)
            next_token_logits = logits[:, -1, :]
            # next_token_scores (batch_beam_size, vocab_size)
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            # next_token_scores (batch_beam_size, vocab_size)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            vocab_size = next_token_scores.shape[-1]
            # next_token_scores (batch_size, num_beams * vocab_size)
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )
            # next_token_scores (batch_size, 2 * num_beams) top 2 * num_beams scores of beams
            # next_tokens (batch_size, 2 * num_beams) which are the indices of the chosen tokens
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * num_beams,  # prevent finishing beam search with eos
                dim=1,
                largest=True,
                sorted=True,
            )
            #  next_tokens (batch_size, 2 * num_beams)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                decoder_inputs,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_idx,
                eos_token_id=self.eos_idx,
            )
            # beam_scores (2 * num_beams)
            beam_scores = beam_outputs["next_beam_scores"]
            # beam_next_tokens (2 * num_beams)
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            # beam_idx (2 * num_beams)
            beam_idx = beam_outputs["next_beam_indices"]
            # decoder_inputs (2 * num_beams, cur_seq_len)
            decoder_inputs = torch.cat(
                [decoder_inputs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            if beam_scorer.is_done or decoder_inputs.shape[-1] >= max_gen_len:
                break
            if use_cache:
                input_ids = beam_next_tokens.unsqueeze(-1)
                past_key_values = self._reorder_cache(past_key_values, beam_idx)
            else:
                input_ids = decoder_inputs

        return beam_scorer.finalize(
            decoder_inputs,
            beam_scores,
            pad_token_id=self.pad_idx,
            eos_token_id=self.eos_idx,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        This function is used to re-order the `past_key_values` when doing beam search.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.


        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def _greedy_search(
        self,
        src: Tensor,
        src_mask: Tensor,
        max_gen_len: int,
        use_cache: bool,
        keep_attentions: bool,
    ):
        memory = self.transformer.encode(src, src_mask)

        batch_size = src.shape[0]

        device = src.device

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        decoder_inputs = torch.LongTensor(batch_size, 1).fill_(self.bos_idx).to(device)

        input_ids = decoder_inputs

        eos_idx_tensor = torch.tensor([self.eos_idx]).to(device)

        finished = False

        past_key_values = None

        tgt_mask = None

        while True:
            if not use_cache:
                tgt_mask = self.generate_subsequent_mask(decoder_inputs.size(1), device)

            outputs = self.transformer.decode(
                input_ids,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                keep_attentions=keep_attentions,
            )

            logits = self.lm_head(outputs[0])

            past_key_values = outputs[1]

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

            if use_cache:
                # only need the last tokens
                input_ids = next_tokens[:, None]
            else:
                input_ids = decoder_inputs

            # all sentences have eos_idx
            if unfinished_sequences.max() == 0:
                finished = True

            if decoder_inputs.shape[-1] >= max_gen_len:
                finished = True

            if finished:
                break

        return decoder_inputs
