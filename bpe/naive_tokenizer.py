"""Naive BPE tokenizer: baseline implementation with in-memory word counts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .base_tokenizer import (
    GPT2_REGEX,
    BaseTokenizer,
    TokenId,
    TokenPair,
    WordCountDict,
    apply_merge,
    split_text_by_special_tokens,
)


def get_pair_counts(word_counts: WordCountDict) -> dict[TokenPair, int]:
    """Count adjacent token-pair frequencies across weighted words.

    For each (word_ids, count) in word_counts, adds count to every adjacent
    pair in word_ids. Used during training to pick the most frequent pair.

    Args:
        word_counts: Dict mapping (word_ids, count); count is the corpus
            frequency of that word sequence.

    Returns:
        Dict mapping each adjacent (left_id, right_id) pair to its total
        frequency across all words.
    """
    pair_counts: defaultdict[TokenPair, int] = defaultdict(int)
    for word_ids, count in word_counts.items():
        # Each adjacent pair in word_ids contributes count to its total.
        for pair in zip(word_ids, word_ids[1:]):
            pair_counts[pair] += count

    return pair_counts


class NaiveTokenizer(BaseTokenizer):
    """Baseline BPE tokenizer with in-memory word counts and full recomputation.

    Each merge iteration rebuilds word_counts from scratch. Simple and correct,
    but not memory-efficient for large corpora.
    """

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
    ) -> None:
        """Initialize tokenizer with base vocab and special tokens.

        Args:
            file_path: Path to corpus file for training.
            vocab_size: Target vocabulary size (must be >= 256).
            special_tokens: Optional map from special token strings to ids.

        Returns:
            None.
        """
        super().__init__(
            file_path=file_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    def pretokenize(self) -> WordCountDict:
        """Scan corpus and build weighted counts of word sequences.

        Reads the file, splits on special tokens, and counts each GPT-2-style
        word (as byte-level ids). Special token chunks are skipped.

        Returns:
            Dict mapping (word_ids,) to corpus frequency count.
        """
        text = self.file_path.read_text(encoding="utf-8")
        chunks = split_text_by_special_tokens(
            text,
            self.special_tokens_pattern,
        )
        word_counts: defaultdict[tuple[TokenId, ...], int] = defaultdict(int)

        for chunk in chunks:
            # Special tokens have fixed ids; exclude from training.
            if chunk in self.special_tokens:
                continue

            words: list[str] = GPT2_REGEX.findall(chunk)
            for word in words:
                word_ids = tuple(word.encode("utf-8"))
                word_counts[word_ids] += 1

        return word_counts

    def train(self) -> None:
        """Learn merge rules until vocabulary reaches self.vocab_size.

        Iteratively: pick highest-frequency pair, add merge rule, apply to all
        words, and rebuild counts. Stops when no pairs remain or vocab is full.

        Returns:
            None. Populates self.merge_rules and self.vocab; calls merge_vocab().
        """
        word_counts = self.pretokenize()

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts = get_pair_counts(word_counts)
            if not pair_counts:
                break

            # Highest-frequency pair; ties prefer smaller token ids.
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            # Apply merge to all words; aggregate counts (merge can coalesce).
            new_word_counts: defaultdict[tuple[TokenId, ...], int] = defaultdict(int)
            for word_ids, count in word_counts.items():
                new_word_ids = tuple(apply_merge(word_ids, best_pair, new_id))
                new_word_counts[new_word_ids] += count

            word_counts = new_word_counts

        self.merge_vocab()

    def save(self, path: str | Path) -> None:
        """Persist tokenizer artifacts to disk (not implemented).

        Args:
            path: File or directory path to write artifacts.

        Returns:
            None.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> NaiveTokenizer:
        """Load tokenizer from disk (not implemented).

        Args:
            path: File or directory path containing artifacts.

        Returns:
            NaiveTokenizer instance.
        """
        raise NotImplementedError
