from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    """Abstract contract shared by all byte-pair encoding tokenizer implementations."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: tuple[str, ...] = (),
    ) -> None:
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    @abstractmethod
    def load_corpus(self) -> str:
        """Load training data from the configured file path."""
        raise NotImplementedError

    @abstractmethod
    def train(self, num_merges: int) -> None:
        """Learn merges and vocabulary from the corpus."""
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
