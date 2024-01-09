import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainArugment:
    """
    Create a 'data' directory and store the dataset under it
    """

    dataset_path: str = f"{os.path.dirname(__file__)}/data/wmt"
    save_dir = f"{os.path.dirname(__file__)}/model_storage"

    src_tokenizer_file: str = f"{save_dir}/source.model"
    tgt_tokenizer_path: str = f"{save_dir}/target.model"
    model_save_path: str = f"{save_dir}/best_transformer.pt"

    dataframe_file: str = "dataframe.{}.pkl"
    use_dataframe_cache: bool = True
    cuda: bool = True
    num_epochs: int = 1
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    grad_clipping: int = 0  # 0 dont use grad clip
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    label_smoothing: float = 0
    warmup_steps: int = 4000
    warmup_factor: float = 0.5
    only_test: bool = False
    max_gen_len: int = 60
    use_wandb: bool = False
    patient: int = 5
    gpus = [0] # [0, 1, 2, 3]
    seed = 12345
    calc_bleu_during_train: bool = True


@dataclass
class ModelArugment:
    d_model: int = 512  # dimension of embeddings
    n_heads: int = 8  # numer of self attention heads
    num_encoder_layers: int = 6  # number of encoder layers
    num_decoder_layers: int = 6  # number of decoder layers
    d_ff: int = d_model * 4  # dimension of feed-forward network
    dropout: float = 0.1  # dropout ratio in the whole network
    max_positions: int = (
        5000  # supported max length of the sequence in positional encoding
    )
    source_vocab_size: int = 32000
    target_vocab_size: int = 32000
    attention_bias: bool = False
    pad_idx: int = 0


train_args = TrainArugment()
model_args = ModelArugment()
