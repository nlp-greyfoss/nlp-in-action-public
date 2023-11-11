import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from argparse import Namespace


class WordRepresentation(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()

        self.char_embed = nn.Embedding(
            args.char_vocab_size, args.char_embedding_dim, padding_idx=0
        )

        self.char_lstm = nn.LSTM(
            input_size=args.char_embedding_dim,
            hidden_size=args.char_hidden_size,
            batch_first=True,
        )

        self.word_embed = nn.Embedding(args.word_vocab_size, args.word_embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.char_embed.weight, -0.005, 0.005)
        # zere vectors for padding index
        self.char_embed.weight.data[0].fill_(0)

        nn.init.uniform_(self.word_embed.weight, -0.005, 0.005)

        nn.init.kaiming_normal_(self.char_lstm.weight_ih_l0)
        nn.init.constant_(self.char_lstm.bias_ih_l0, val=0)

        nn.init.orthogonal_(self.char_lstm.weight_hh_l0)
        nn.init.constant_(self.char_lstm.bias_hh_l0, val=0)

    def forward(self, x: Tensor, x_char: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): word input sequence a with shape (batch_size, seq_len)
            x_char (Tensor): character input sequence a with shape (batch_size, seq_len, word_len)

        Returns:
            Tensor: concatenated word and char embedding  (batch_size, seq_len, word_embedding_dim + char_hidden_size)
        """

        batch_size, seq_len, word_len = x_char.shape
        # (batch_size, seq_len, word_len) -> (batch_size * seq_len, word_len)
        x_char = x_char.view(-1, word_len)
        # x_char_embed (batch_size * seq_len, word_len, char_embedding_dim)
        x_char_embed = self.char_embed(x_char)
        # x_char_hidden (1, batch_size * seq_len, char_hidden_size)
        _, (x_char_hidden, _) = self.char_lstm(x_char_embed)

        # x_char_hidden (batch_size, seq_len, char_hidden_size)
        x_char_hidden = x_char_hidden.view(batch_size, seq_len, -1)

        # x_embed (batch_size, seq_len, word_embedding_dim),
        x_embed = self.word_embed(x)

        # (batch_size, seq_len, word_embedding_dim + char_hidden_size)
        return torch.cat([x_embed, x_char_hidden], dim=-1)


class ContextRepresentation(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.context_lstm = nn.LSTM(
            input_size=args.word_embedding_dim + args.char_hidden_size,
            hidden_size=args.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0)
        nn.init.constant_(self.context_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0)
        nn.init.constant_(self.context_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_hh_l0_reverse, val=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the contextual information about input.
        Args:
            x (Tensor): (batch_size, seq_len, hidden_size)

        Returns:
            Tensor: (batch_size, seq_len, 2 * hidden_size)
        """

        # (batch_size, seq_len, 2 * hidden_size)
        return self.context_lstm(x)[0]


class MatchingLayer(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.l = args.num_perspective

        self.epsilon = args.epsilon  # prevent dividing zero

        for i in range(1, 9):
            self.register_parameter(
                f"mp_w{i}",
                nn.Parameter(torch.rand(self.l, args.hidden_size)),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for _, parameter in self.named_parameters():
            nn.init.kaiming_normal_(parameter)

    def extra_repr(self) -> str:
        return ",".join([p[0] for p in self.named_parameters()])

    def forward(self, p: Tensor, q: Tensor) -> Tensor:
        """
        p: (batch_size, seq_len_p, 2 * hidden_size)
        q: (batch_size, seq_len_q, 2 * hidden_size)
        """
        # both p_fw and p_bw are (batch_size, seq_len_p, hidden_size)
        p_fw, p_bw = torch.split(p, self.args.hidden_size, -1)
        # both q_fw and q_bw are (batch_size, seq_len_q, hidden_size)
        q_fw, q_bw = torch.split(q, self.args.hidden_size, -1)

        # 1. Full Matching
        # (batch_size, seq_len1, 2 * l)
        m1 = torch.cat(
            [
                self._full_matching(p_fw, q_fw[:, -1, :], self.mp_w1),
                self._full_matching(p_bw, q_bw[:, 0, :], self.mp_w2),
            ],
            dim=-1,
        )

        # 2. Maxpooling Matching
        # (batch_size, seq_len1, 2 * l)
        m2 = torch.cat(
            [
                self._max_pooling_matching(p_fw, q_fw, self.mp_w3),
                self._max_pooling_matching(p_bw, q_bw, self.mp_w4),
            ],
            dim=-1,
        )

        # 3. Attentive Matching
        # (batch_size, seq_len1, seq_len2)
        consine_matrix_fw = self._consine_matrix(p_fw, q_fw)
        # (batch_size, seq_len1, seq_len2)
        consine_matrix_bw = self._consine_matrix(p_bw, q_bw)
        # (batch_size, seq_len1, 2 * l)
        m3 = torch.cat(
            [
                self._attentive_matching(p_fw, q_fw, consine_matrix_fw, self.mp_w5),
                self._attentive_matching(p_bw, q_bw, consine_matrix_bw, self.mp_w6),
            ],
            dim=-1,
        )

        # 4. Max Attentive Matching
        # (batch_size, seq_len1, 2 * l)
        m4 = torch.cat(
            [
                self._max_attentive_matching(p_fw, q_fw, consine_matrix_fw, self.mp_w7),
                self._max_attentive_matching(p_bw, q_bw, consine_matrix_bw, self.mp_w8),
            ],
            dim=-1,
        )

        # (batch_size, seq_len1, 8 * l)
        return torch.cat([m1, m2, m3, m4], dim=-1)

    def _cosine_similarity(self, v1: Tensor, v2: Tensor) -> Tensor:
        """compute cosine similarity between v1 and v2.

        Args:
            v1 (Tensor): (..., hidden_size)
            v2 (Tensor): (..., hidden_size)

        Returns:
            Tensor: (..., l)
        """
        # element-wise multiply
        cosine = v1 * v2
        # caculate on hidden_size dimenstaion
        cosine = cosine.sum(-1)
        # caculate on hidden_size dimenstaion
        v1_norm = torch.sqrt(torch.sum(v1**2, -1).clamp(min=self.epsilon))
        v2_norm = torch.sqrt(torch.sum(v2**2, -1).clamp(min=self.epsilon))
        # (batch_size, seq_len, l) or (batch_size, seq_len1, seq_len2, l)
        return cosine / (v1_norm * v2_norm)

    def _time_distributed_multiply(self, x: Tensor, w: Tensor) -> Tensor:
        """element-wise multiply vector and weights.

        Args:
            x (Tensor): sequence vector (batch_size, seq_len, hidden_size) or singe vector (batch_size, hidden_size)
            w (Tensor): weights (num_perspective, hidden_size)

        Returns:
            Tensor: (batch_size, seq_len, num_perspective, hidden_size) or (batch_size, num_perspective, hidden_size)
        """

        # dimension of x
        n_dim = x.dim()
        hidden_size = x.size(-1)
        # if n_dim == 3
        seq_len = x.size(1)

        # (batch_size * seq_len, hidden_size) for n_dim == 3
        # (batch_size, hidden_size) for n_dim == 2
        x = x.contiguous().view(-1, hidden_size)

        # (batch_size * seq_len, 1, hidden_size) for n_dim == 3
        # (batch_size, 1, hidden_size) for n_dim == 2
        x = x.unsqueeze(1)

        # (1, num_perspective, hidden_size)
        w = w.unsqueeze(0)

        # (batch_size * seq_len, num_perspective, hidden_size) for n_dim == 3
        # (batch_size, num_perspective, hidden_size) for n_dim == 2
        x = x * w

        # reshape to original shape
        if n_dim == 3:
            # (batch_size, seq_len, num_perspective, hidden_size)
            x = x.view(-1, seq_len, self.l, hidden_size)
        elif n_dim == 2:
            # (batch_size, num_perspective, hidden_size)
            x = x.view(-1, self.l, hidden_size)
        # (batch_size, seq_len, num_perspective, hidden_size) for n_dim == 3
        # (batch_size, num_perspective, hidden_size) for n_dim == 2
        return x

    def _full_matching(self, v1: Tensor, v2_last: Tensor, w: Tensor) -> Tensor:
        """full matching operation.

        Args:
            v1 (Tensor): the full embedding vector sequence (batch_size, seq_len1, hidden_size)
            v2_last (Tensor): single embedding vector (batch_size, hidden_size)
            w (Tensor): weights of one direction (num_perspective, hidden_size)

        Returns:
            Tensor: (batch_size, seq_len1, num_perspective)
        """

        # (batch_size, seq_len1, num_perspective, hidden_size)
        v1 = self._time_distributed_multiply(v1, w)
        # (batch_size, num_perspective, hidden_size)
        v2 = self._time_distributed_multiply(v2_last, w)
        # (batch_size, 1, num_perspective, hidden_size)
        v2 = v2.unsqueeze(1)
        # (batch_size, seq_len1, num_perspective)
        return self._cosine_similarity(v1, v2)

    def _max_pooling_matching(self, v1: Tensor, v2: Tensor, w: Tensor) -> Tensor:
        """max pooling matching operation.

        Args:
            v1 (Tensor): (batch_size, seq_len1, hidden_size)
            v2 (Tensor): (batch_size, seq_len2, hidden_size)
            w (Tensor): (num_perspective, hidden_size)

        Returns:
            Tensor: (batch_size, seq_len1, num_perspective)
        """

        # (batch_size, seq_len1, num_perspective, hidden_size)
        v1 = self._time_distributed_multiply(v1, w)
        # (batch_size, seq_len2, num_perspective, hidden_size)
        v2 = self._time_distributed_multiply(v2, w)
        # (batch_size, seq_len1, 1, num_perspective, hidden_size)
        v1 = v1.unsqueeze(2)
        # (batch_size, 1, seq_len2, num_perspective, hidden_size)
        v2 = v2.unsqueeze(1)
        # (batch_size, seq_len1, seq_len2, num_perspective)
        cosine = self._cosine_similarity(v1, v2)
        # (batch_size, seq_len1, num_perspective)
        return cosine.max(2)[0]

    def _consine_matrix(self, v1: Tensor, v2: Tensor) -> Tensor:
        """
        Args:
            v1 (Tensor): (batch_size, seq_len1, hidden_size)
            v2 (Tensor): _description_

        Returns:
            Tensor: (batch_size, seq_len1, seq_len2)
        """

        # (batch_size, seq_len1, 1, hidden_size)
        v1 = v1.unsqueeze(2)
        # (batch_size, 1, seq_len2, hidden_size)
        v2 = v2.unsqueeze(1)
        # (batch_size, seq_len1, seq_len2)
        return self._cosine_similarity(v1, v2)

    def _mean_attentive_vectors(self, v2: Tensor, cosine_matrix: Tensor) -> Tensor:
        """
        calculte mean attentive vector for the entire sentence by weighted summing all
        the contextual embeddings of the entire sentence.

        Args:
            v2 (Tensor):  v2 (batch_size, seq_len2, hidden_size)
            cosine_matrix (Tensor): cosine_matrix (batch_size, seq_len1, seq_len2)

        Returns:
            Tensor: (batch_size, seq_len1, hidden_size)
        """

        #  (batch_size, seq_len1, seq_len2, 1)
        expanded_cosine_matrix = cosine_matrix.unsqueeze(-1)
        # (batch_size, 1, seq_len2, hidden_size)
        v2 = v2.unsqueeze(1)
        # (batch_size, seq_len1, hidden_size)
        weighted_sum = (expanded_cosine_matrix * v2).sum(2)
        # (batch_size, seq_len1, 1)
        sum_cosine = (cosine_matrix.sum(-1) + self.epsilon).unsqueeze(-1)
        # (batch_size, seq_len1, hidden_size)
        return weighted_sum / sum_cosine

    def _max_attentive_vectors(self, v2: Tensor, cosine_matrix: Tensor) -> Tensor:
        """
        calculte max attentive vector for the entire sentence by picking
        the contextual embedding with the highest cosine similarity.

        Args:
            v2 (Tensor): v2 (batch_size, seq_len2, hidden_size)
            cosine_matrix (Tensor): cosine_matrix (batch_size, seq_len1, seq_len2)

        Returns:
            Tensor: (batch_size, seq_len1, hidden_size)
        """

        # (batch_size, seq_len1) index value between [0, seq_len2)
        _, max_v2_step_idx = cosine_matrix.max(-1)

        hidden_size = v2.size(-1)
        seq_len1 = max_v2_step_idx.size(-1)

        # (batch_size * seq_len2, hidden_size)
        v2 = v2.contiguous().view(-1, hidden_size)

        # (batch_size * seq_len1, )
        max_v2_step_idx = max_v2_step_idx.contiguous().view(-1)

        # (batch_size * seq_len1, hidden_size)
        max_v2 = v2[max_v2_step_idx]
        # (batch_size, seq_len1, hidden_size)
        return max_v2.view(-1, seq_len1, hidden_size)

    def _attentive_matching(
        self, v1: Tensor, v2: Tensor, cosine_matrix: Tensor, w: Tensor
    ) -> Tensor:
        """

        Args:
            v1 (Tensor): (batch_size, seq_len1, hidden_size)
            v2 (Tensor): (batch_size, seq_len2, hidden_size)
            cosine_matrix (Tensor): (batch_size, seq_len1, seq_len2)
            w (Tensor):  (l, hidden_size)

        Returns:
            Tensor:
        """

        # (batch_size, seq_len1, hidden_size)
        attentive_vec = self._mean_attentive_vectors(v2, cosine_matrix)
        # (batch_size, seq_len1, num_perspective, hidden_size)
        attentive_vec = self._time_distributed_multiply(attentive_vec, w)

        # (batch_size, seq_len, num_perspective, hidden_size)
        v1 = self._time_distributed_multiply(v1, w)
        # (batch_size, seq_len1, num_perspective)
        return self._cosine_similarity(v1, attentive_vec)

    def _max_attentive_matching(
        self, v1: Tensor, v2: Tensor, cosine_matrix: Tensor, w: Tensor
    ) -> Tensor:
        """

        Args:
            v1 (Tensor): (batch_size, seq_len1, hidden_size)
            v2 (Tensor): (batch_size, seq_len2, hidden_size)
            cosine_matrix (Tensor): (batch_size, seq_len1, seq_len2)
            w (Tensor): (num_perspective, hidden_size)

        Returns:
            Tensor: (batch_size, seq_len1, num_perspective)
        """

        # (batch_size, seq_len1, num_perspective, hidden_size)
        v1 = self._time_distributed_multiply(v1, w)
        # (batch_size, seq_len1, embedding_szie)
        max_attentive_vec = self._max_attentive_vectors(v2, cosine_matrix)
        # (batch_size, seq_len1, num_perspective, hidden_size)
        max_attentive_vec = self._time_distributed_multiply(max_attentive_vec, w)

        # (batch_size, seq_len1, num_perspective)
        return self._cosine_similarity(v1, max_attentive_vec)


class AggregationLayer(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()

        self.agg_lstm = nn.LSTM(
            input_size=args.num_perspective * 8,
            hidden_size=args.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.agg_lstm.weight_ih_l0)
        nn.init.constant_(self.agg_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.agg_lstm.weight_hh_l0)
        nn.init.constant_(self.agg_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.agg_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.agg_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.agg_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.agg_lstm.bias_hh_l0_reverse, val=0)

    def forward(self, v1: Tensor, v2: Tensor) -> Tensor:
        """

        Args:
            v1 (Tensor): (batch_size, seq_len1, l * 8)
            v2 (Tensor): (batch_size, seq_len2, l * 8)


        Returns:
            Tensor: (batch_size, 4 * hidden_size)
        """

        batch_size = v1.size(0)

        # v1_last (2, batch_size, hidden_size)
        _, (v1_last, _) = self.agg_lstm(v1)
        # v2_last (2, batch_size, hidden_size)
        _, (v2_last, _) = self.agg_lstm(v2)

        # v1_last (batch_size, 2, hidden_size)
        v1_last = v1_last.transpose(1, 0)
        v2_last = v2_last.transpose(1, 0)
        # v1_last (batch_size, 2 * hidden_size)
        v1_last = v1_last.contiguous().view(batch_size, -1)
        # v2_last (batch_size, 2 * hidden_size)
        v2_last = v2_last.contiguous().view(batch_size, -1)

        # (batch_size, 4 * hidden_size)
        return torch.cat([v1_last, v2_last], dim=-1)


class Prediction(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.predict = nn.Sequential(
            nn.Linear(args.hidden_size * 4, args.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 2, args.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.predict[0].weight, -0.005, 0.005)
        nn.init.constant_(self.predict[0].bias, val=0)

        nn.init.uniform_(self.predict[-1].weight, -0.005, 0.005)
        nn.init.constant_(self.predict[-1].bias, val=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.predict(x)


class BiMPM(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        # the concatenated embedding size of word
        self.d = args.word_embedding_dim + args.char_hidden_size
        self.l = args.num_perspective

        self.word_rep = WordRepresentation(args)
        self.context_rep = ContextRepresentation(args)
        self.matching_layer = MatchingLayer(args)
        self.aggregation = AggregationLayer(args)
        self.prediction = Prediction(args)

    def dropout(self, x: Tensor) -> Tensor:
        return F.dropout(input=x, p=self.args.dropout, training=self.training)

    def forward(self, p: Tensor, q: Tensor, char_p: Tensor, char_q: Tensor) -> Tensor:
        """

        Args:
            p (Tensor): word inpute sequence (batch_size, seq_len1)
            q (Tensor): word inpute sequence (batch_size, seq_len2)
            char_p (Tensor): character input sequence (batch_size, seq_len1, word_len)
            char_q (Tensor): character input sequence (batch_size, seq_len1, word_len)

        Returns:
            Tensor: (batch_size,  2)
        """

        # (batch_size, seq_len1, word_embedding_dim + char_hidden_size)
        p_rep = self.dropout(self.word_rep(p, char_p))
        # (batch_size, seq_len2, word_embedding_dim + char_hidden_size)
        q_rep = self.dropout(self.word_rep(q, char_q))
        #  batch_size, seq_len1, 2 * hidden_size)
        p_context = self.dropout(self.context_rep(p_rep))
        #  batch_size, seq_len2, 2 * hidden_size)
        q_context = self.dropout(self.context_rep(q_rep))
        # (batch_size, seq_len1, 8 * l)
        p_match = self.dropout(self.matching_layer(p_context, q_context))
        # (batch_size, seq_len2, 8 * l)
        q_match = self.dropout(self.matching_layer(q_context, p_context))

        # (batch_size,  4 * hidden_size)
        aggregation = self.dropout(self.aggregation(p_match, q_match))
        # (batch_size,  2)
        return self.prediction(aggregation)
