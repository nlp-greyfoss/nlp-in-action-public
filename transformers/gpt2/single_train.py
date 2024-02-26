from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from torch.utils.data.dataloader import DataLoader

from torch.optim import AdamW
import torch

from tqdm import tqdm

from log import logger
from utils import EarlyStopper, generate
from config import train_args


from configuration_gpt2 import GPT2Config
from modeling_gpt2 import GPT2LMHeadModel


def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def train(model, train_dataloader, val_dataloader, optimizer, device, scheduler, args):
    max_grad_norm = args.max_grad_norm
    logging_steps = args.logging_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps

    total_loss = 0.0
    logging_loss = 0.0
    best_loss = 10000

    global_steps = 0

    early_stopper = EarlyStopper()

    for epoch in range(args.epochs):
        model.train()
        p_bar = tqdm(train_dataloader, disable=False)
        for step, batch in enumerate(p_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss

            total_loss += loss.item()

            p_bar.set_description(
                f"epoch {epoch + 1:2d} (loss={loss.item():5.3f} | global_steps {global_steps:4d} | lr {scheduler.get_last_lr()[0]:.5f} )"
            )
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_steps += 1

                if logging_steps > 0 and global_steps & logging_steps == 0:
                    train_loss = (total_loss - logging_loss) / (
                        logging_steps * gradient_accumulation_steps
                    )

                    if args.use_wandb:
                        wandb.log(
                            {
                                "global_steps": global_steps,
                                "lr": scheduler.get_lr()[0],
                                "train_loss:": train_loss,
                            }
                        )

                    logging_loss = total_loss

        eval_loss = evalute(model, val_dataloader, device)
        logger.info(
            f"epoch {epoch} | global_steps {global_steps}  | eval loss {eval_loss:.3f}"
        )

        if args.use_wandb:
            wandb.log({"epoch": epoch, "eval_loss:": eval_loss})

        torch.cuda.empty_cache()

        if eval_loss < best_loss:
            best_loss = eval_loss
            logger.info(
                f"Saving model to {args.model_name} with best eval loss {eval_loss:.3f}"
            )
            # save to local disk
            model.save_pretrained(f"{args.model_name}")

        if early_stopper.step(eval_loss):
            print(f"Stop from early stopping.")
            break


@torch.no_grad()
def evalute(model, dataloader, device):
    model.eval()
    p_bar = tqdm(dataloader, desc="iter", disable=False)

    total_loss = 0.0

    for batch in p_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(batch["input_ids"], labels=labels)

        total_loss += outputs.loss.item()

    test_loss = total_loss / len(dataloader)

    return test_loss


if __name__ == "__main__":
    # run train_tokenizer.py to get tokenizer
    if train_args.from_remote:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.owner}/{train_args.tokenizer_name}", use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{train_args.tokenizer_name}", use_fast=True
        )

    if train_args.use_wandb:
        import wandb

        wandb.init(
            project="simple-gpt",
            config=vars(train_args),
        )

    config = GPT2Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel(config)
    model.to(device)

    # run data_process.py to get dataset
    # run data_process.py to get dataset
    if train_args.from_remote:
        tokenized_dataset = load_dataset(
            f"{train_args.owner}/{train_args.dataset_name}"
        )
    else:
        tokenized_dataset = load_from_disk(f"{train_args.dataset_name}")

    tokenized_dataset.set_format("torch")

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["valid"]

    batch_size = int(train_args.batch_size / train_args.gradient_accumulation_steps)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    total_training_steps = int(
        train_args.epochs
        * len(train_dataloader)
        / train_args.gradient_accumulation_steps
    )

    print(f"total train steps={total_training_steps}")

    optimizer = AdamW(
        get_grouped_params(model, weight_decay=train_args.weight_decay),
        lr=train_args.learning_rate,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_args.warmup_proportion * total_training_steps),
        num_training_steps=total_training_steps,
    )

    train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        lr_scheduler,
        train_args,
    )

    model = GPT2LMHeadModel.from_pretrained(f"{train_args.model_name}")
    generated_text = generate(model, tokenizer, device, "萧炎经过不懈地修炼，终于达到了斗帝级别")

    print(f"generated text: {generated_text}")
