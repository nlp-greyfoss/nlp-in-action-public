from torch.utils.data import Dataset
import datasets

class TripletDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self.dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index) -> dict[str, str]:
        anchor = self.dataset[index]["query"]
        pos = self.dataset[index]["pos"]
        neg = self.dataset[index]["neg"]
      
        return {"anchor": anchor, "pos": pos, "neg": neg}


class TripletCollator:
    def __call__(self, features) -> dict[str, list[str]]:
        anchor = [feature["anchor"] for feature in features]
        positive = [feature["pos"] for feature in features]
        negative = [feature["neg"] for feature in features]

        return {"anchor": anchor, "positive": positive, "negative": negative}
