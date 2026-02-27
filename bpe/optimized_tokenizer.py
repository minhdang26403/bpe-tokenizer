"""Optimized BPE tokenizer with parallel pretokenization and in-place merges."""

from __future__ import annotations

import mmap
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import regex  # type: ignore[import-untyped]

from .base_tokenizer import (
    GPT2_PATTERN,
    BaseTokenizer,
    TokenId,
    TokenPair,
    WordCountDict,
)

# Byte-level version of GPT2_REGEX for use with mmap/memoryview.
GPT2_BYTES_REGEX = regex.compile(GPT2_PATTERN.encode("utf-8"))

# Parallel: encoded_words[i] = word (list[TokenId]), word_freqs[i] = its count.
EncodedWordList = list[list[TokenId]]
WordFreqList = list[int]


def pretokenize_worker(
    args: tuple[Path, int, int, set[bytes]],
) -> WordCountDict:
    """Pretokenize a file segment into word counts. Single tuple arg for Pool.map.

    Pool.map passes one argument per call; we pack (path, start, end, special_tokens)
    into a tuple for this reason.

    Args:
        args: (file_path, start_byte, end_byte, special_tokens_bytes). The segment
            to process is mmap[start:end].

    Returns:
        WordCountDict mapping (word_ids,) to frequency in this segment.
    """
    file_path, start, end, special_tokens = args

    # Split on special tokens; capturing group makes separators appear in result.
    split_pattern = None
    if special_tokens:
        alternation = b"|".join([regex.escape(t) for t in special_tokens])
        split_pattern = regex.compile(b"(" + alternation + b")")

    word_counts: defaultdict[tuple[TokenId, ...], int] = defaultdict(int)
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            segment = memoryview(mm)[start:end]
            chunks = split_pattern.split(segment) if split_pattern else [segment]
            for chunk in chunks:
                if chunk in special_tokens:
                    continue
                words = GPT2_BYTES_REGEX.findall(chunk)
                for word_bytes in words:
                    word_ids = tuple(word_bytes)
                    word_counts[word_ids] += 1
            segment.release()

    return word_counts


def merge_word_counts(
    per_segment_word_counts: list[WordCountDict],
) -> tuple[EncodedWordList, WordFreqList]:
    """Merge word counts from multiple segments into parallel word list + freq list.

    Deduplicates words across segments and sums their counts. Returns two
    parallel lists: encoded words (as mutable list[TokenId]) and their freqs.

    Args:
        per_segment_word_counts: List of WordCountDict, one per worker segment.

    Returns:
        (encoded_words, word_freqs) where encoded_words[i] is list[TokenId] and
        word_freqs[i] is its total corpus frequency.
    """
    word_to_idx: dict[tuple[TokenId, ...], int] = {}
    encoded_words: EncodedWordList = []
    word_freqs: WordFreqList = []

    for word_counts in per_segment_word_counts:
        for word_ids, count in word_counts.items():
            if word_ids not in word_to_idx:
                word_to_idx[word_ids] = len(encoded_words)
                encoded_words.append(list(word_ids))
                word_freqs.append(count)
            else:
                idx = word_to_idx[word_ids]
                word_freqs[idx] += count

    return encoded_words, word_freqs


def apply_merge_in_place(
    ids: list[TokenId], best_pair: TokenPair, new_id: TokenId
) -> None:
    """Replace all best_pair occurrences in ids with new_id, mutating in place.

    Uses two pointers: read position i, write position write_idx. Shrinks the
    list at the end.

    Args:
        ids: Mutable list of token ids to merge (modified in place).
        best_pair: The (left, right) token pair to replace.
        new_id: The token id that replaces each best_pair occurrence.

    Returns:
        None.
    """
    i = 0
    write_idx = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
            ids[write_idx] = new_id
            i += 2
        else:
            ids[write_idx] = ids[i]
            i += 1
        write_idx += 1

    # Trim the list to the new length.
    del ids[write_idx:]


class OptimizedTokenizer(BaseTokenizer):
    """Optimized BPE tokenizer with parallel pretokenization and in-place merges."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
        num_workers: int = 4,
    ) -> None:
        """Initialize tokenizer with parallel worker count.

        Args:
            file_path: Path to corpus file for training.
            vocab_size: Target vocabulary size (must be >= 256).
            special_tokens: Optional map from special token strings to ids.
            num_workers: Number of parallel workers for pretokenization.

        Returns:
            None.
        """
        super().__init__(
            file_path=file_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        self.num_workers = num_workers
        # Byte form for regex/mmap; avoids repeated encoding in workers.
        self.encoded_special_tokens = {
            token.encode("utf-8") for token in self.special_tokens
        }

    def get_worker_segment_boundaries(self, num_desired_chunks: int) -> list[int]:
        """Find byte boundaries to split the file for parallel workers.

        Splits at newlines and/or special token boundaries so we never cut
        mid-line or mid-token. Uses newlines when present; adds special
        tokens when configured so chunks stay balanced even when tokens are rare.

        Args:
            num_desired_chunks: Target number of segments (typically num_workers).

        Returns:
            List of byte offsets [b0, b1, ..., bn] where segments are
            (b0,b1), (b1,b2), ..., (b_{n-1}, bn). b0=0, bn=file_size.
        """
        # Match newlines; if special tokens exist, also match them for more splits.
        if not self.encoded_special_tokens:
            pattern = regex.compile(b"\n")
        else:
            escaped = b"|".join([regex.escape(t) for t in self.encoded_special_tokens])
            pattern = regex.compile(b"\n|(?:" + escaped + b")")

        file_size = self.file_path.stat().st_size
        chunk_size = file_size // num_desired_chunks
        boundaries: list[int] = [0]

        with open(self.file_path, "rb") as f:
            # Create a read-only memory map of the entire file
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for i in range(1, num_desired_chunks):
                    target = i * chunk_size

                    # Check if the previous search already jumped past this target
                    if target <= boundaries[-1]:
                        continue

                    # The search handles all the "buffer" and "straddle" logic for us!
                    match = pattern.search(mm, target)
                    if match:
                        boundaries.append(match.end())
                    else:
                        break  # No more delimiters; stop early.

        # Ensure last segment reaches end of file.
        if boundaries[-1] != file_size:
            boundaries.append(file_size)

        return boundaries

    def train(self) -> None:
        """Learn merge rules using parallel pretokenization and in-place merges.

        Splits corpus into segments, pretokenizes in parallel, merges counts,
        then iteratively applies BPE merges in place on the unified word list.

        Returns:
            None. Populates self.merge_rules and self.vocab; calls merge_vocab().
        """
        boundaries = self.get_worker_segment_boundaries(self.num_workers)
        segments = [
            (self.file_path, start, end, self.encoded_special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]

        with Pool(self.num_workers) as p:
            per_segment_word_counts = p.map(pretokenize_worker, segments)

        encoded_words, word_freqs = merge_word_counts(per_segment_word_counts)

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts: defaultdict[TokenPair, int] = defaultdict(int)
            for word_ids, count in zip(encoded_words, word_freqs):
                for pair in zip(word_ids, word_ids[1:]):
                    pair_counts[pair] += count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            for word_ids in encoded_words:
                apply_merge_in_place(word_ids, best_pair, new_id)

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
    def load(cls, path: str | Path) -> OptimizedTokenizer:
        """Load tokenizer from disk (not implemented).

        Args:
            path: File or directory path containing artifacts.

        Returns:
            OptimizedTokenizer instance.
        """
        raise NotImplementedError
