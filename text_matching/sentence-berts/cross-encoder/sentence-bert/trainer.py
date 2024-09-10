import torch
from transformers.trainer import Trainer

from typing import Optional
import os
import logging

TRAINING_ARGS_NAME = "training_args.bin"


from modeling import SentenceBert

logger = logging.getLogger(__name__)


class CrossTrainer(Trainer):

    def compute_loss(self, model: SentenceBert, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        return model(inputs, labels)["loss"]

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
