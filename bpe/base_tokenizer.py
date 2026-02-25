from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import regex  # type: ignore[import-untyped]

TokenId = int
TokenPair = tuple[TokenId, TokenId]
WordCountDict = dict[tuple[TokenId, ...], int]
BASE_VOCAB_SIZE = 256
ENCODE_WORD_CACHE_SIZE = 32768

GPT2_REGEX = regex.compile(
    (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| "
        r"?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
)


def apply_merge(
    ids: list[TokenId] | tuple[TokenId, ...],
    best_pair: TokenPair,
    new_id: TokenId,
) -> list[TokenId]:
    """Replace all adjacent `best_pair` occurrences in `ids` with `new_id`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


def get_pair_counts(word_count: WordCountDict) -> dict[TokenPair, int]:
    """Count adjacent token-pair frequencies across weighted words."""
    pair_counts: dict[TokenPair, int] = {}
    for word_ids, count in word_count.items():
        for pair in zip(word_ids, word_ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    return pair_counts


def split_by_special_tokens(text: str, special_tokens: dict[str, TokenId]) -> list[str]:
    """Split text while preserving special tokens as separate chunks."""
    if not special_tokens:
        return [text]

    pattern = r"(" + "|".join(regex.escape(token) for token in special_tokens) + r")"
    return regex.split(pattern, text)


class BaseTokenizer(ABC):
    """Abstract contract shared by all byte-pair encoding tokenizer implementations."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None,
    ) -> None:
        """Store shared config and derived structures for special tokens."""
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        assert vocab_size >= BASE_VOCAB_SIZE

        self.merge_rules: dict[TokenPair, TokenId] = {}  # used in encode
        self.vocab = {id: bytes([id]) for id in range(BASE_VOCAB_SIZE)}  # decode map

        self.special_tokens = special_tokens if special_tokens else {}
        self.inverse_special_tokens: dict[TokenId, bytes] = {}
        for token, id in self.special_tokens.items():
            assert id >= self.vocab_size
            assert id not in self.inverse_special_tokens
            self.inverse_special_tokens[id] = token.encode(encoding="utf-8")

        self.is_trained = False

    @abstractmethod
    def load_corpus(self) -> str:
        """Load training data from the configured file path."""
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """Train tokenizer to the configured vocabulary size."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token ids."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back to text."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist tokenizer state to disk."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseTokenizer:
        """Load a saved tokenizer from disk."""
        raise NotImplementedError
