import torch.nn as nn
import torch


class ESIM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_classes: int,
        lstm_dropout: float = 0.1,
        dropout: float = 0.5,
    ) -> None:
        """_summary_

        Args:
            vocab_size (int): the size of the Vocabulary
            embedding_size (int): the size of each embedding vector
            hidden_size (int): the size of the hidden layer
            num_classes (int): the output size
            lstm_dropout (float, optional): dropout ratio in lstm layer. Defaults to 0.1.
            dropout (float, optional): dropout ratio in linear layer. Defaults to 0.5.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # lstm for input embedding
        self.lstm_a = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.lstm_b = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        # lstm for augment inference vector
        self.lstm_v_a = nn.LSTM(
            8 * hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.lstm_v_b = nn.LSTM(
            8 * hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.predict = nn.Sequential(
            nn.Linear(8 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """

        Args:
            a (torch.Tensor): input sequence a with shape (batch_size, a_seq_len)
            b (torch.Tensor): input sequence b with shape (batch_size, b_seq_len)

        Returns:
            torch.Tensor:
        """
        # a (batch_size, a_seq_len, embedding_size)
        a_embed = self.embedding(a)
        # b (batch_size, b_seq_len, embedding_size)
        b_embed = self.embedding(b)

        # a_bar (batch_size, a_seq_len, 2 * hidden_size)
        a_bar, _ = self.lstm_a(a_embed)
        # b_bar (batch_size, b_seq_len, 2 * hidden_size)
        b_bar, _ = self.lstm_b(b_embed)

        # score (batch_size, a_seq_len, b_seq_len)
        score = torch.matmul(a_bar, b_bar.permute(0, 2, 1))

        # softmax (batch_size, a_seq_len, b_seq_len) x (batch_size, b_seq_len, 2 * hidden_size)
        # a_tilde (batch_size, a_seq_len, 2 * hidden_size)
        a_tilde = torch.matmul(torch.softmax(score, dim=2), b_bar)
        # permute (batch_size, b_seq_len, a_seq_len) x (batch_size, a_seq_len, 2 * hidden_size)
        # b_tilde (batch_size, b_seq_len, 2 * hidden_size)
        b_tilde = torch.matmul(torch.softmax(score, dim=1).permute(0, 2, 1), a_bar)

        # m_a (batch_size, a_seq_len, 8 * hidden_size)
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        # m_b (batch_size, b_seq_len, 8 * hidden_size)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)

        # v_a (batch_size, a_seq_len, 2 * hidden_size)
        v_a, _ = self.lstm_v_a(m_a)
        # v_b (batch_size, b_seq_len, 2 * hidden_size)
        v_b, _ = self.lstm_v_b(m_b)

        # (batch_size, 2 * hidden_size)
        avg_a = torch.mean(v_a, dim=1)
        avg_b = torch.mean(v_b, dim=1)

        max_a, _ = torch.max(v_a, dim=1)
        max_b, _ = torch.max(v_b, dim=1)
        # (batch_size, 8 * hidden_size)
        v = torch.cat([avg_a, max_a, avg_b, max_b], dim=-1)

        return self.predict(v)
