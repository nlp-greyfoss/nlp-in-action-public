from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


class Vocabulary:
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx: dict = None, tokens: list[str] = None) -> None:
        """
        Args:
            token_to_idx (dict, optional): a pre-existing map of tokens to indices. Defaults to None.
            tokens (list[str], optional): a list of unique tokens with no duplicates. Defaults to None.
        """

        assert any(
            [tokens, token_to_idx]
        ), "At least one of these parameters should be set as not None."
        if token_to_idx:
            self._token_to_idx = token_to_idx
        else:
            self._token_to_idx = {}
            if PAD_TOKEN not in tokens:
                tokens = [PAD_TOKEN] + tokens

            for idx, token in enumerate(tokens):
                self._token_to_idx[token] = idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self.unk_index = self._token_to_idx[UNK_TOKEN]
        self.pad_index = self._token_to_idx[PAD_TOKEN]

    @classmethod
    def build(
        cls,
        sentences: list[list[str]],
        min_freq: int = 2,
        reserved_tokens: list[str] = None,
    ) -> "Vocabulary":
        """Construct the Vocabulary from sentences

        Args:
            sentences (list[list[str]]): a list of tokenized sequences
            min_freq (int, optional): the minimum word frequency to be saved. Defaults to 2.
            reserved_tokens (list[str], optional): the reserved tokens to add into the Vocabulary. Defaults to None.

        Returns:
            Vocabulary: a Vocubulary instane
        """

        token_freqs = defaultdict(int)
        for sentence in tqdm(sentences):
            for token in sentence:
                token_freqs[token] += 1

        unique_tokens = (reserved_tokens if reserved_tokens else []) + [UNK_TOKEN]
        unique_tokens += [
            token
            for token, freq in token_freqs.items()
            if freq >= min_freq and token != UNK_TOKEN
        ]
        return cls(tokens=unique_tokens)

    def __len__(self) -> int:
        return len(self._idx_to_token)

    def __iter__(self):
        for idx, token in self._idx_to_token.items():
            yield idx, token

    def __getitem__(self, tokens: list[str] | str) -> list[int] | int:
        """Retrieve the indices associated with the tokens or the index with the single token

        Args:
            tokens (list[str] | str): a list of tokens or single token

        Returns:
            list[int] | int: the indices or the single index
        """
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self.unk_index)
        return [self.__getitem__(token) for token in tokens]

    def lookup_token(self, indices: list[int] | int) -> list[str] | str:
        """Retrive the tokens associated with the indices or the token with the single index

        Args:
            indices (list[int] | int): a list of index or single index

        Returns:
            list[str] | str: the corresponding tokens (or token)
        """

        if not isinstance(indices, (list, tuple)):
            return self._idx_to_token[indices]

        return [self._idx_to_token[index] for index in indices]

    def to_serializable(self) -> dict:
        """Returns a dictionary that can be serialized"""
        return {"token_to_idx": self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents: dict) -> "Vocabulary":
        """Instantiates the Vocabulary from a serialized dictionary


        Args:
            contents (dict): a dictionary generated by `to_serializable`

        Returns:
            Vocabulary: the Vocabulary instance
        """
        return cls(**contents)

    def __repr__(self):
        return f"<Vocabulary(size={len(self)})>"


class TMVectorizer:
    """The Vectorizer which vectorizes the Vocabulary"""

    def __init__(self, vocab: Vocabulary, max_len: int) -> None:
        """
        Args:
            vocab (Vocabulary): maps characters to integers
            max_len (int): the max length of the sequence in the dataset
        """
        self.vocab = vocab
        self.max_len = max_len
        self.padding_index = vocab.pad_index

    def _vectorize(self, indices: list[int], vector_length: int = -1) -> np.ndarray:
        """Vectorize the provided indices

        Args:
            indices (list[int]): a list of integers that represent a sequence
            vector_length (int, optional): an arugment for forcing the length of index vector. Defaults to -1.

        Returns:
            np.ndarray: the vectorized index array
        """

        if vector_length <= 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        if len(indices) > vector_length:
            vector[:] = indices[:vector_length]
        else:
            vector[: len(indices)] = indices
            vector[len(indices) :] = self.padding_index

        return vector

    def _get_indices(self, sentence: list[str]) -> list[int]:
        """Return the vectorized sentence

        Args:
            sentence (list[str]): list of tokens
        Returns:
            indices (list[int]): list of integers representing the sentence
        """
        return [self.vocab[token] for token in sentence]

    def vectorize(
        self, sentence: list[str], use_dataset_max_length: bool = True
    ) -> np.ndarray:
        """
        Return the vectorized sequence

        Args:
            sentence (list[str]): raw sentence from the dataset
            use_dataset_max_length (bool): whether to use the global max vector length
        Returns:
            the vectorized sequence with padding
        """
        vector_length = -1
        if use_dataset_max_length:
            vector_length = self.max_len

        indices = self._get_indices(sentence)
        vector = self._vectorize(indices, vector_length=vector_length)

        return vector

    @classmethod
    def from_serializable(cls, contents: dict) -> "TMVectorizer":
        """Instantiates the TMVectorizer from a serialized dictionary

        Args:
            contents (dict): a dictionary generated by `to_serializable`

        Returns:
            TMVectorizer:
        """
        vocab = Vocabulary.from_serializable(contents["vocab"])
        max_len = contents["max_len"]
        return cls(vocab=vocab, max_len=max_len)

    def to_serializable(self) -> dict:
        """Returns a dictionary that can be serialized

        Returns:
            dict: a dict contains Vocabulary instance and max_len attribute
        """
        return {"vocab": self.vocab.to_serializable(), "max_len": self.max_len}

    def save_vectorizer(self, filepath: str) -> None:
        """Dump this TMVectorizer instance to file

        Args:
            filepath (str): the path to store the file
        """
        with open(filepath, "w") as f:
            json.dump(self.to_serializable(), f)

    @classmethod
    def load_vectorizer(cls, filepath: str) -> "TMVectorizer":
        """Load TMVectorizer from a file

        Args:
            filepath (str): the path stored the file

        Returns:
            TMVectorizer:
        """
        with open(filepath) as f:
            return TMVectorizer.from_serializable(json.load(f))


class TMDataset(Dataset):
    """Dataset for text matching"""

    def __init__(self, text_df: pd.DataFrame, vectorizer: TMVectorizer) -> None:
        """

        Args:
            text_df (pd.DataFrame): a DataFrame which contains the processed data examples
            vectorizer (TMVectorizer): a TMVectorizer instance
        """

        self.text_df = text_df
        self._vectorizer = vectorizer

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        row = self.text_df.iloc[index]

        vector1 = self._vectorizer.vectorize(row.sentence1)
        vector2 = self._vectorizer.vectorize(row.sentence2)

        mask1 = vector1 != self._vectorizer.padding_index
        mask2 = vector2 != self._vectorizer.padding_index

        return (vector1, vector2, mask1, mask2, row.label)

    def get_vectorizer(self) -> TMVectorizer:
        return self._vectorizer

    def __len__(self) -> int:
        return len(self.text_df)