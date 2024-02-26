from transformers import PretrainedConfig


class GPTConfig(PretrainedConfig):
    model_type = "openai-gpt"

    def __init__(
        self,
        vocab_size=5000,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        initializer_range=0.02,
        **kwargs
    ) -> None:
        """

        Args:
            vocab_size (int, optional): vocabulary size. Defaults to 5000.
            n_positions (int, optional): the maximum sequence length that this model might ever be used with. Defaults to 512.
            n_embd (int, optional): dimensionality of the embeddings and hidden states. Defaults to 768.
            n_layer (int, optional): number of hidden layers. Defaults to 12.
            n_head (int, optional): number of attention heads for each attention layer. Defaults to 12.
            dropout (float, optional): the dropout probability. Defaults to 0.1.
            initializer_range (tuple, optional): the standard deviation of the truncated_normal_initializer for initializing all weight matrices. Defaults to (0.02,).
        """
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.initializer_range = initializer_range

        super().__init__(**kwargs)
