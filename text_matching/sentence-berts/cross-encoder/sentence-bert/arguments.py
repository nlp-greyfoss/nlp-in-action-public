from dataclasses import dataclass, field
from typing import Optional

import os


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to train corpus"}
    )
    eval_data_path: str = field(default=None, metadata={"help": "Path to eval corpus"})
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text."
        },
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(
                f"cannot find file: {self.train_data_path}, please set a true path"
            )

        if not os.path.exists(self.eval_data_path):
            raise FileNotFoundError(
                f"cannot find file: {self.eval_data_path}, please set a true path"
            )
