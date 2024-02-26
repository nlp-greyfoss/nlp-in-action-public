from dataclasses import dataclass


@dataclass
class TrainArguments:
    batch_size: int = 16
    weight_decay: float = 1e-1
    epochs: int = 50
    warmup_proportion: float = 0.05
    learning_rate: float = 4e-5
    logging_steps = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_wandb: bool = False
    from_remote: bool = True
    dataset_name: str = "doupo-dataset"
    model_name: str = "simple-gpt-doupo"
    tokenizer_name: str = "simple-gpt-doupo"
    owner: str = "greyfoss"
    devices: str = "0"


train_args = TrainArguments()
