"""Base BPE tokenizer with shared encoding/decoding and special-token handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import regex  # type: ignore[import-untyped]

TokenId = int
TokenPair = tuple[TokenId, TokenId]
WordCountDict = dict[tuple[TokenId, ...], int]


BASE_VOCAB_SIZE = 256
ENCODE_WORD_CACHE_SIZE = 32768
# GPT-2 style regex: splits on Unicode letters, numbers, whitespace, contractions.
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
GPT2_REGEX = regex.compile(GPT2_PATTERN)


def apply_merge(
    ids: list[TokenId] | tuple[TokenId, ...],
    best_pair: TokenPair,
    new_id: TokenId,
) -> list[TokenId]:
    """Replace all adjacent occurrences of best_pair in ids with new_id.

    Walks through the id sequence once; whenever (ids[i], ids[i+1]) equals
    best_pair, emits new_id and skips both; otherwise emits ids[i] and advances.
    """
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


class BaseTokenizer(ABC):
    """Base tokenizer shared by all byte-pair encoding tokenizer implementations."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None,
    ) -> None:
        """Store shared config and derived structures for special tokens."""
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE}, got {vocab_size}"
            )

        # Key data structures: merge rules for encoding, vocab for decoding.
        self.merge_rules: dict[TokenPair, TokenId] = {}
        self.vocab = {
            id: bytes([id]) for id in range(BASE_VOCAB_SIZE)
        }  # byte-level base vocab

        self.special_tokens = special_tokens if special_tokens else {}

        # Inverse map: token_id -> bytes for decoding special tokens.
        self.inverse_special_tokens: dict[TokenId, bytes] = {}
        for token, token_id in self.special_tokens.items():
            if token_id < self.vocab_size:
                raise ValueError(
                    f"special token id {token_id} must be >= vocab_size "
                    f"({self.vocab_size}); ids in [0, vocab_size) are reserved"
                )
            if token_id in self.inverse_special_tokens:
                raise ValueError(f"duplicate special token id {token_id}")
            self.inverse_special_tokens[token_id] = token.encode("utf-8")

        self.special_tokens_pattern = None
        if self.special_tokens:
            self.special_tokens_pattern = regex.compile(
                r"("
                + "|".join(regex.escape(token) for token in self.special_tokens)
                + r")"
            )

    def split_by_special_tokens(self, text: str) -> list[str]:
        """Split text into chunks, separating on special token boundaries."""
        if not self.special_tokens_pattern:
            return [text]

        return self.special_tokens_pattern.split(text)

    @abstractmethod
    def train(self) -> None:
        """Train tokenizer to the configured vocabulary size."""
        raise NotImplementedError

    @lru_cache(maxsize=ENCODE_WORD_CACHE_SIZE)
    def encode_word(self, word: str) -> list[TokenId]:
        """Encode a single pre-tokenized word using learned merge rules."""
        ids = list(word.encode("utf-8"))  # Start with byte-level ids.
        while len(ids) >= 2:
            best_pair = min(
                (pair for pair in zip(ids, ids[1:]) if pair in self.merge_rules),
                key=lambda p: self.merge_rules[p],
                default=None,
            )

            if not best_pair:
                break

            ids = apply_merge(ids, best_pair, self.merge_rules[best_pair])
        return ids

    def encode_chunk(self, chunk: str) -> list[TokenId]:
        """Encode a chunk of text (no special tokens inside) into token ids."""
        ids: list[TokenId] = []
        words: list[str] = GPT2_REGEX.findall(chunk)
        for word in words:
            word_ids = self.encode_word(word)
            ids.extend(word_ids)

        return ids

    def encode(self, text: str) -> list[TokenId]:
        """Encode text into token ids, preserving explicit special tokens."""
        chunks = self.split_by_special_tokens(text)
        ids: list[TokenId] = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_chunk(chunk))

        return ids

    def decode(self, token_ids: list[TokenId]) -> str:
        """Decode token ids back to a UTF-8 string."""
        chunk_bytes: list[bytes] = []
        for id in token_ids:
            if id in self.vocab:
                chunk_bytes.append(self.vocab[id])
            elif id in self.inverse_special_tokens:
                chunk_bytes.append(self.inverse_special_tokens[id])
            else:
                raise ValueError(f"invalid token id {id}")

        text_bytes = b"".join(chunk_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist tokenizer state to disk."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseTokenizer:
        """Load a saved tokenizer from disk."""
        raise NotImplementedError
