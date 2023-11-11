import torch.nn as nn
import torch


class DSSM(nn.Module):
    """The DSSM model implemention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        activation: str = "relu",
        internal_hidden_sizes: list[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        """

        Args:
            vocab_size (int): the size of the Vocabulary
            embedding_size (int): the size of each embedding vector
            activation (str, optional): the activate function. Defaults to "relu".
            internal_hidden_sizes (list[int], optional): the hidden size of inernal Linear Layer. Defaults to [256, 128, 64].
            dropout (float, optional): dropout ratio. Defaults to 0.1.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        assert activation.lower() in [
            "relu",
            "tanh",
        ], "activation only supports relu or tanh"

        if activation.lower() == "relu":
            activate_func = nn.ReLU()
        else:
            activate_func = nn.Tanh()

        self.dnn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, internal_hidden_sizes[0]),
            activate_func,
            nn.Dropout(dropout),
            nn.Linear(internal_hidden_sizes[0], internal_hidden_sizes[1]),
            activate_func,
            nn.Dropout(dropout),
            nn.Linear(internal_hidden_sizes[1], internal_hidden_sizes[2]),
            activate_func,
            nn.Dropout(dropout),
        )

        self._init_weights()

    def forward(self, sentence1: torch.Tensor, sentence2: torch.Tensor) -> torch.Tensor:
        """Using the same network to compute the representations of two sentences

        Args:
            sentence1 (torch.Tensor): shape (batch_size, seq_len)
            sentence2 (torch.Tensor): shape (batch_size, seq_len)

        Returns:
            torch.Tensor: the cosine similarity between sentence1 and sentence2
        """
        # shape (batch_size, seq_len) ->  (batch_size, seq_len, embedding_size) -> (batch_size, embedding_size)
        embed_1 = self.embedding(sentence1).sum(1)
        embed_2 = self.embedding(sentence2).sum(1)
        # (batch_size, embedding_size) -> (batch_size, internal_hidden_sizes[2])
        vector_1 = self.dnn(embed_1)
        vector_2 = self.dnn(embed_2)
        # (batch_size, internal_hidden_sizes[2]) -> (batch_size, )
        return torch.cosine_similarity(vector_1, vector_2, dim=1, eps=1e-8)

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
