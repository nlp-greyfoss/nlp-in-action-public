import jieba
from typing import Tuple
from collections import defaultdict
from tqdm import tqdm
import pickle
from functools import lru_cache


class BPETokenizer:
    """
    For teaching purposes only, we need to use the fast sentencepiece package in practice.
    """
    unk_token = "<unk>"  # unknown
    pad_token = "<pad>"  # padding
    bos_token = "<bos>"  # begin of sentence
    eos_token = "<eos>"  # end of sentence
    eow_token = "Ġ"  # end of word which is </w>

    def __init__(self) -> None:
        self.word_freqs = defaultdict(int)
        self.merges = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.eow_token,
        ]

        for token in special_tokens:
            self._add_token(token)

        self.pad_idx = self.token_to_id.get(self.pad_token)
        self.unk_idx = self.token_to_id.get(self.unk_token)
        self.bos_idx = self.token_to_id.get(self.bos_token)
        self.eos_idx = self.token_to_id.get(self.eos_token)

        self.cache = {}

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _learn_vocab(self, corpus: list[str]) -> None:
        for sentence in corpus:
            sentence = sentence.lower()
            words = [w + self.eow_token for w in jieba.cut(sentence) if w != " "]
            for word in words:
                self.word_freqs[word] += 1

    def _compute_pair_freqs(self, splits: dict[str, list[str]]) -> dict[Tuple, int]:
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, a: str, b: str, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split

        return splits

    def _merge_vocab(self, vocab_size: int, splits: dict[str, list[str]]):
        merges = {}

        with tqdm(total=vocab_size - self.vocab_size) as pbar:
            while self.vocab_size < vocab_size:
                pair_freqs = self._compute_pair_freqs(splits)

                best_pair = None
                max_freq = 0

                for pair, freq in pair_freqs.items():
                    if max_freq < freq:
                        best_pair = pair
                        max_freq = freq

                if best_pair is None:
                    print(f"no best pair found , vocab_size = {self.vocab_size}")
                    break

                splits = self._merge_pair(*best_pair, splits)
                merges[best_pair] = best_pair[0] + best_pair[1]
                self._add_token(best_pair[0] + best_pair[1])
                pbar.update(1)

        return merges

    @classmethod
    def train(
        cls,
        corpus: list[str],
        save_path: str,
        vocab_size: int = 20000,
    ) -> "BPETokenizer":
        tokenizer = cls()
        tokenizer._train(corpus, vocab_size)
        tokenizer.save_model(save_path)

        return tokenizer

    def _train(self, corpus: list[str], vocab_size: int):
        self._learn_vocab(corpus)
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # sort by alphabetically
        characters = sorted(set(sum(splits.values(), [])))

        for c in characters:
            self._add_token(c)

        self.merges = self._merge_vocab(vocab_size, splits)

    
    def _merge_token(self, split, pair, merge):
        key = f"{split}-{pair}-{merge}"
        if key in self.cache:
            return self.cache[key]
        i = 0
        while i < len(split) - 1:
            if split[i] == pair[0] and split[i + 1] == pair[1]:
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        self.cache[key] = split
        return split

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        words = [w + self.eow_token for w in jieba.cut(text) if w != " "]
        splits = [[c for c in word] for word in words]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                splits[idx] = self._merge_token(split, pair, merge)
        return sum(splits, [])

 
    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_idx)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def _convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self._convert_id_to_token(index) for index in token_ids]

    def _convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)

        return self._convert_tokens_to_ids(tokens)

    def clean_up_tokenization(self, out_string: str) -> str:
        out_string = out_string.replace(self.eow_token, " ")

        return out_string

    def decode(self, token_ids: list[int]) -> str:
        tokens = self._convert_ids_to_tokens(token_ids)
        return self.clean_up_tokenization("".join(tokens))

    def save_model(self, file_name: str) -> None:
        with open(file_name, "wb") as f:
            model = {
                "word_freqs": self.word_freqs,
                "merges": self.merges,
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
            }
            pickle.dump(model, f)

        with open(f"{file_name}.vocab", "w", encoding="utf-8") as f:
            f.write("\n".join(self.token_to_id.keys()))

    @classmethod
    def load_model(cls, file_name: str) -> "BPETokenizer":
        bpe = cls()
        with open(file_name, "rb") as file:
            model = pickle.load(file)
            bpe.word_freqs = model["word_freqs"]
            bpe.merges = model["merges"]
            bpe.token_to_id = model["token_to_id"]
            bpe.id_to_token = model["id_to_token"]
        return bpe
