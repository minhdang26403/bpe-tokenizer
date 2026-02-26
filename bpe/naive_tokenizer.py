"""Naive BPE tokenizer: baseline implementation with in-memory word counts."""

from __future__ import annotations

from pathlib import Path

from .base_tokenizer import (
    GPT2_REGEX,
    BaseTokenizer,
    TokenId,
    TokenPair,
    WordCountDict,
    apply_merge,
)


def get_pair_counts(word_count: WordCountDict) -> dict[TokenPair, int]:
    """Count adjacent token-pair frequencies across weighted words.

    For each (word_ids, count) in word_count, adds count to every adjacent
    pair in word_ids. Used during training to pick the most frequent pair.
    """
    pair_counts: dict[TokenPair, int] = {}
    for word_ids, count in word_count.items():
        for pair in zip(word_ids, word_ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    return pair_counts


class NaiveTokenizer(BaseTokenizer):
    """Baseline BPE tokenizer using in-memory word counts and full recomputation.

    Each merge iteration rebuilds word counts from scratch. Simple and correct,
    but not memory-efficient for large corpora.
    """

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

    def pretokenize(self) -> WordCountDict:
        """Scan corpus and build weighted counts of (word_ids, count)."""
        text = self.file_path.read_text(encoding="utf-8")
        chunks = self.split_by_special_tokens(text)
        word_counts: WordCountDict = {}
        for chunk in chunks:
            # Special tokens already have fixed ids; skip them in training.
            if chunk in self.special_tokens:
                continue

            words: list[str] = GPT2_REGEX.findall(chunk)
            for word in words:
                word_ids = tuple(word.encode("utf-8"))
                word_counts[word_ids] = word_counts.get(word_ids, 0) + 1

        return word_counts

    def train(self) -> None:
        """Learn merge rules until vocabulary reaches `self.vocab_size`."""
        word_counts = self.pretokenize()

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts = get_pair_counts(word_counts)
            if not pair_counts:
                break

            # Pick highest-frequency pair; tie-break by (count, pair) for determinism.
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            # Apply merge to all words and aggregate counts (merge may coalesce).
            new_word_counts: WordCountDict = {}
            for word_ids, count in word_counts.items():
                new_word_ids = tuple(apply_merge(word_ids, best_pair, new_id))
                new_word_counts[new_word_ids] = (
                    new_word_counts.get(new_word_ids, 0) + count
                )

            word_counts = new_word_counts

    def save(self, path: str | Path) -> None:
        """Persist tokenizer artifacts (not implemented yet)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> NaiveTokenizer:
        """Load tokenizer artifacts from disk (not implemented yet)."""
        raise NotImplementedError
