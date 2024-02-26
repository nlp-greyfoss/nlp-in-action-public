from log import logger


def generate(model, tokenizer, device, prefix):
    model = model.to(device)
    input_ids = tokenizer.encode(
        prefix, return_tensors="pt", add_special_tokens=False
    ).to(device)
    beam_output = model.generate(
        input_ids,
        max_length=512,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        repetition_penalty=1.25,
    )

    return tokenizer.decode(beam_output[0], skip_special_tokens=True).replace(" ", "")


class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        self.patience = patience
        self.counter = patience
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")

        self.best_value = 0.0 if mode == "max" else float("inf")

        self.mode = mode

    def step(self, value: float) -> bool:
        if self.is_better(value):
            self.best_value = value
            self.counter = self.patience
        else:
            self.counter -= 1

            if self.counter == 0:
                return True

        if self.counter != self.patience:
            logger.info(f"early stop left: {self.counter}")

        return False

    def is_better(self, a: float) -> bool:
        if self.mode == "min":
            return a < self.best_value
        return a > self.best_value
