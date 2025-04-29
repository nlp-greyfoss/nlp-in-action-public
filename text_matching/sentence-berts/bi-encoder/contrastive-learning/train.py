from transformers import set_seed, HfArgumentParser, TrainingArguments

import logging
from pathlib import Path

from datetime import datetime

from modeling import SentenceBert
from trainer import TripletTrainer
from arguments import DataArguments, ModelArguments
from dataset import TripletCollator, TripletDataset
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

os.environ["WANDB_PROJECT"] = "constrastive-learning"


def main():
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    output_dir = f"{training_args.output_dir}/{model_args.model_name_or_path.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    training_args.output_dir = output_dir

    logger.info(f"Training parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

    set_seed(training_args.seed)

    model = SentenceBert(
        model_args.model_name_or_path,
        trust_remote_code=True,
        max_length=data_args.max_length,
    )

    tokenizer = model.tokenizer

    train_dataset = TripletDataset(data_args.train_data_path)
    # eval_dataset = PairDataset(data_args.eval_data_path)

    trainer = TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=TripletCollator(),
        tokenizer=tokenizer,
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
