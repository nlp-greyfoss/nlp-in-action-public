from dataclasses import dataclass


@dataclass
class TrainArguments:
    batch_size: int = 8
    weight_decay: float = 1e-1
    epochs: int = 50
    warmup_proportion: float = 0.025
    learning_rate: float = 5e-4
    logging_steps = 100
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    use_wandb: bool = False
    from_remote: bool = False
    dataset_name: str = "doupo-dataset2"
    model_name: str = "simple-gpt2-doupo"
    tokenizer_name: str = "simple-gpt2-doupo"
    owner: str = "greyfoss"
    devices: str = "0"


train_args = TrainArguments()
