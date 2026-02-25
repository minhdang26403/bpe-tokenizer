from __future__ import annotations

from pathlib import Path

from .base_tokenizer import BaseTokenizer, TokenId


class OptimizedTokenizer(BaseTokenizer):
    """Optimized byte-pair encoding tokenizer implementation."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
    ) -> None:
        super().__init__(
            file_path=file_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    def load_corpus(self) -> str:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> OptimizedTokenizer:
        raise NotImplementedError
