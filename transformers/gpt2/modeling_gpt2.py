import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from typing import Tuple, Union, Any

from configuration_gpt2 import GPT2Config


class Conv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        """1D-convolutional layer as defined by Radford et al. for OpenAI GPT.

        Args:
            in_features (int): the number of input features.
            out_features (int): the number of output features.
        """
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)

        Returns:
            Tensor: (batch_size, seq_len, out_features)
        """
        # size_out (batch_size, seq_len, out_features)
        size_out = x.size()[:-1] + (self.out_features,)
        # self.bias + x @ self.weight
        # x -view-> (batch_size *  seq_len,n_embd)
        # (batch_size * seq_len,n_embd) x (n_embd, out_features)
        # -> (batch_size * seq_len, out_features)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x (batch_size, seq_len, out_features)
        x = x.view(size_out)

        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_embd, n_embd * 4)
        self.c_proj = Conv1D(n_embd * 4, n_embd)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)

        Returns:
            Tensor: (batch_size, seq_len, n_embd)
        """
        # h (batch_size, seq_len, n_embd * 4)
        h = self.act(self.c_fc(x))
        # h (batch_size, seq_len, n_embd)
        h = self.c_proj(h)
        return self.dropout(h)


class Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head

        self.c_attn = Conv1D(self.n_embd, self.n_embd * 3)
        self.c_proj = Conv1D(self.n_embd, self.n_embd)
        # use flash attention or not
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,  # will not be saved alongside parameters
            )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

    def split_heads(self, x: Tensor, is_key: bool = False) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)
            is_key (bool, optional): is key or not. Defaults to False.

        Returns:
            Tensor: (batch_size, n_head, n_embd / n_head, seq_len) if is_key = True ,
              else (batch_size,  n_head, seq_len, n_embd / n_head)
        """
        # (batch_size, seq_len, n_head, n_embd / n_head)
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # x (batch_size, seq_len, n_head, n_embd / n_head)
        x = x.view(*new_shape)
        if is_key:
            # (batch_size, n_head, n_embd / n_head, seq_len)
            return x.permute(0, 2, 3, 1)
        # (batch_size,  n_head, seq_len, n_embd / n_head)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor):  (batch_size,  n_head, seq_len, n_embd / n_head)

        Returns:
            Tensor: (batch_size, seq_len, n_embd)
        """
        # x (batch_size,  seq_len, n_head, n_embd / n_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (batch_size, seq_len, n_embd)
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def _attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
    ) -> list[Tensor]:
        """

        Args:
            q (Tensor): (batch_size,  n_head, seq_len, n_embd / n_head)
            k (Tensor): (batch_size, n_head, n_embd / n_head, seq_len)
            v (Tensor): (batch_size,  n_head, seq_len, n_embd / n_head)

        Returns:
            Tensor: (batch_size,  n_head, seq_len, n_embd / n_head) attn_output
            Tensor(optional): (batch_size, n_head, seq_len, seq_len) attn_weights

        """
        if self.flash and attention_mask is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=True,
            )
            weights = None
        else:
            # scores (batch_size,  n_head, seq_len, seq_len)
            scores = torch.matmul(q, k) / math.sqrt(v.size(-1))
            # scores = scores.masked_fill(
            #    self.bias[:, :, : scores.size(-2), : scores.size(-1)] == 0, float("-inf")
            # )
            bias = self.bias[:, :, : scores.size(-2), : scores.size(-1)]
            # more efficient than masked_fill
            scores = scores * bias + -1e9 * (1 - bias)

            # weights (batch_size,  n_head, seq_len, seq_len)
            weights = self.attn_dropout(F.softmax(scores, dim=-1))

            if attention_mask is not None:
                weights = weights + attention_mask

            del scores
            # attn_output (batch_size,  n_head, seq_len, n_embd / n_head)
            attn_output = torch.matmul(weights, v)

        outputs = [attn_output]
        if output_attentions:
            outputs.append(weights)

        return outputs

    def forward(
        self, x: Tensor, attention_mask: Tensor = None, output_attentions: bool = False
    ) -> list[Tensor]:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)

        Returns:
            Tensor: (batch_size, seq_len, n_embd) attn_output
            Tensor(optional): (batch_size, n_head, seq_len, seq_len) attn_weights

        """
        # calculate query, key ,value for all heads in batch
        # x (batch_size, seq_len, n_embd * 3)
        x = self.c_attn(x)
        #  query, key, value (batch_size, seq_len, n_embd)
        query, key, value = x.split(self.n_embd, dim=2)
        # query (batch_size,  n_head, seq_len, n_embd / n_head)
        query = self.split_heads(query)
        # key (batch_size, n_head, n_embd / n_head, seq_len)
        key = self.split_heads(key, is_key=not self.flash)
        # value (batch_size,  n_head, seq_len, n_embd / n_head)
        value = self.split_heads(value)
        # attn_output (batch_size,  n_head, seq_len, n_embd / n_head)
        attn_outputs = self._attn(query, key, value, attention_mask, output_attentions)
        attn_output = attn_outputs[0]

        del query, key, value

        # output (batch_size, seq_len, n_embd)
        output = self.merge_heads(attn_output)
        # (batch_size, seq_len, n_embd)
        output = self.c_proj(output)

        output = self.proj_dropout(output)

        outputs = [output] + attn_outputs[1:]
        return outputs


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        n_embd = config.n_embd
        self.attn = Attention(config)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(
        self, x: Tensor, attention_mask: Tensor = None, output_attentions: bool = False
    ) -> Tensor:
        """

        Args:
            x (Tensor): (batch_size, seq_len, n_embd)
            attention_mask (Tensor, optional)
            output_attentions (bool, optional)

        Returns:
            Tensor: (batch_size, seq_len, n_embd) block output
            Tensor(optional): (batch_size, n_head, seq_len, seq_len) attn_weights

        """
        residual = x
        x = self.ln_1(x)
        attn_outputs = self.attn(x, attention_mask, output_attentions)
        # attn_output (batch_size, n_head, seq_len, n_embd / n_head)
        attn_output = attn_outputs[0]
        # resident connection
        x = attn_output + residual

        residual = x
        # x (batch_size, seq_len, n_embd)
        x = self.ln_2(x)
        # m (batch_size, seq_len, n_embd)
        x = self.mlp(x)
        # resident connection
        # x (batch_size, seq_len, n_embd)
        x = x + residual
        outputs = [x] + attn_outputs[1:]

        return outputs


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.n_layer)
                    ),
                )


class GPT2Model(GPT2PreTrainedModel):
    """
    The bare GPT transformer model outputting raw hidden-states without any specific head on top.
    See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py .
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.config = config
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln = nn.LayerNorm(config.n_embd)

        self.register_buffer(
            "position_ids", torch.arange(config.n_positions), persistent=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        """
        Args:
            input_ids (torch.LongTensor): (batch_size, seq_len)
            output_attentions (bool, optional): whether or not to return the attentions tensors of all attention layers. Defaults to False.
            output_hidden_states (bool, optional): whether or not to return the hidden states of all layers. Defaults to False.
            return_dict (bool, optional): whether or not to return a ModelOutput instead of a plain tuple. Defaults to False.



        Returns:
            Union[Tuple[torch.Tensor], BaseModelOutput]: tuple or BaseModelOutput
        """

        input_shape = input_ids.size()

        inputs_embeds = self.tokens_embed(input_ids)
        # generate position ids
        position_ids = self.position_ids[None, : input_shape[-1]]

        position_embeds = self.positions_embed(position_ids)

        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.dropout(hidden_states)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for _, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(hidden_states, attention_mask, output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = self.ln(hidden_states)

        # add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPT2LMHeadModel(GPT2PreTrainedModel):
    """The LM head model for the GPT model."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        """

        Args:
            input_ids (torch.LongTensor): (batch_size, sequence_length) Indices of input sequence tokens in the vocabulary.
            labels (torch.LongTensor, optional): _description_. Defaults to None.
            attention_mask (torch.FloatTensor) (batch_size, sequence_length) Mask to avoid performing attention on padding token indices.
            output_attentions (bool, optional):  Whether or not to return the attentions tensors of all attention layers. Defaults to False.
            output_hidden_states (bool, optional):  Whether or not to return the hidden states of all layers. Defaults to False.
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple. Defaults to True.

        Returns:
            Union[Tuple[torch.Tensor], CausalLMOutput]:
        """

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        transformer_outputs = self.transformer(
            input_ids,
            output_attentions=output_attentions,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # hidden_states (batch_size, seq_len, n_embd)
        hidden_states = transformer_outputs[0]
        # lm_logits (batch_size, seq_len, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()  # remove last tokens
            shift_labels = labels[..., 1:].contiguous()  # remove first tokens
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            # add hidden states and attention if they are here
            output = (lm_logits,) + transformer_outputs[1:]
            # return (loss, output) if loss is not None else output
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> dict[str, Any]:
        return {"input_ids": input_ids}
