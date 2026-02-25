from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .base_tokenizer import (
    ENCODE_WORD_CACHE_SIZE,
    GPT2_REGEX,
    BaseTokenizer,
    TokenId,
    WordCountDict,
    apply_merge,
    get_pair_counts,
    split_by_special_tokens,
)


class NaiveTokenizer(BaseTokenizer):
    """Baseline byte-pair encoding tokenizer implementation."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
    ) -> None:
        """Initialize tokenizer state with byte-level base vocabulary."""
        super().__init__(
            file_path=file_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    def load_corpus(self) -> str:
        """Read and return training corpus text from disk."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def train(self) -> None:
        """Learn merge rules until vocabulary reaches `self.vocab_size`."""
        text = self.load_corpus()

        word_counts: WordCountDict = {}
        chunks = split_by_special_tokens(text, self.special_tokens)

        for chunk in chunks:
            # A special token already has its token id, so we don't need to train on it
            if chunk in self.special_tokens:
                continue

            words: list[str] = GPT2_REGEX.findall(chunk)
            for word in words:
                word_ids = tuple(word.encode(encoding="utf-8"))
                word_counts[word_ids] = word_counts.get(word_ids, 0) + 1

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts = get_pair_counts(word_counts)
            if not pair_counts:
                break

            # Deterministic tie-break when frequencies are equal.
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            new_word_counts: WordCountDict = {}
            for word_ids, count in word_counts.items():
                new_word_ids = tuple(apply_merge(word_ids, best_pair, new_id))
                new_word_counts[new_word_ids] = (
                    new_word_counts.get(new_word_ids, 0) + count
                )

            word_counts = new_word_counts

        self.is_trained = True

    @lru_cache(maxsize=ENCODE_WORD_CACHE_SIZE)
    def _encode_word(self, word_bytes: bytes) -> list[TokenId]:
        """Encode one word by repeatedly applying the best-ranked merge rule."""
        ids = list(word_bytes)
        while len(ids) >= 2:
            # If the generator is empty, best_pair becomes None
            best_pair = min(
                (pair for pair in zip(ids, ids[1:]) if pair in self.merge_rules),
                key=lambda pair: self.merge_rules[pair],
                default=None,
            )

            # If no more mergeable pairs exist, exit the loop
            if not best_pair:
                break

            ids = apply_merge(ids, best_pair, self.merge_rules[best_pair])

        return ids

    def _encode_chunk(self, chunk: str) -> list[TokenId]:
        """Encode a chunk that does not include special tokens."""
        ids = []
        words: list[str] = GPT2_REGEX.findall(chunk)
        for word in words:
            word_bytes = word.encode(encoding="utf-8")
            word_ids = self._encode_word(word_bytes)
            ids.extend(word_ids)

        return ids

    def encode(self, text: str) -> list[TokenId]:
        """Encode text into token ids, preserving explicit special tokens."""
        assert self.is_trained

        chunks = split_by_special_tokens(text, self.special_tokens)
        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode_chunk(chunk))

        return ids

    def decode(self, token_ids: list[TokenId]) -> str:
        """Decode token ids back to a UTF-8 string."""
        assert self.is_trained

        chunk_bytes = []
        for id in token_ids:
            if id in self.vocab:
                chunk_bytes.append(self.vocab[id])
            elif id in self.inverse_special_tokens:
                chunk_bytes.append(self.inverse_special_tokens[id])
            else:
                raise ValueError(f"invalid token id: {id}")
        text_bytes = b"".join(chunk_bytes)
        text = text_bytes.decode(encoding="utf-8", errors="replace")
        return text

    def save(self, path: str | Path) -> None:
        """Persist tokenizer artifacts (not implemented yet)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> NaiveTokenizer:
        """Load tokenizer artifacts from disk (not implemented yet)."""
        raise NotImplementedError
