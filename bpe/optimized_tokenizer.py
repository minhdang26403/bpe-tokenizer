"""Optimized BPE tokenizer with parallel pretokenization and in-place merges."""

from __future__ import annotations

import heapq
import mmap
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Mapping

import regex  # type: ignore[import-untyped]

from .base_tokenizer import (
    GPT2_REGEX,
    BaseTokenizer,
    TokenId,
    TokenPair,
    WordCountDict,
    compile_special_tokens_pattern,
    split_text_by_special_tokens,
)

# Parallel: encoded_words[i] = word (list[TokenId]), word_freqs[i] = its count.
EncodedWordList = list[list[TokenId]]
WordFreqList = list[int]
# pair_counts[pair] = total weighted occurrences of that pair in the corpus.
PairCountDict = defaultdict[TokenPair, int]
# pair_map[pair] = set of word indices where that pair appears at least once.
PairMap = defaultdict[TokenPair, set[int]]


def pretokenize_worker(
    args: tuple[Path, int, int, dict[str, TokenId]],
) -> WordCountDict:
    """Pretokenize one mmap text segment into word counts.

    Pool.map passes one argument per call; we pack
    (file_path, start, end, special_tokens) into a tuple for this reason.

    Args:
        args: (file_path, start_byte, end_byte, special_tokens).

    Returns:
        WordCountDict mapping (word_ids,) to frequency within this segment.
    """
    file_path, start, end, special_tokens = args
    special_tokens_pattern = compile_special_tokens_pattern(special_tokens)

    word_counts: defaultdict[tuple[TokenId, ...], int] = defaultdict(int)
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with memoryview(mm)[start:end] as segment:
                # Decode only this segment; no full-file read.
                segment_text = bytes(segment).decode("utf-8")
                chunks = split_text_by_special_tokens(
                    segment_text,
                    special_tokens_pattern=special_tokens_pattern,
                )
                for chunk in chunks:
                    if chunk in special_tokens:
                        continue
                    words: list[str] = GPT2_REGEX.findall(chunk)
                    for word in words:
                        word_ids = tuple(word.encode("utf-8"))
                        word_counts[word_ids] += 1

    return word_counts


def merge_word_counts(
    per_segment_word_counts: list[WordCountDict],
) -> tuple[
    EncodedWordList,
    WordFreqList,
    PairCountDict,
    PairMap,
]:
    """Merge word counts from multiple segments into parallel word list + freq list.

    Deduplicates words across segments and sums their counts. Returns two
    parallel lists: encoded words (as mutable list[TokenId]) and their freqs.

    Args:
        per_segment_word_counts: List of WordCountDict, one per worker segment.

    Returns:
        (encoded_words, word_freqs, pair_counts, pair_map) where:
        - encoded_words[i] is list[TokenId] for unique word i.
        - word_freqs[i] is the corpus frequency of encoded_words[i].
        - pair_counts[pair] is weighted occurrence count of pair across corpus.
        - pair_map[pair] is set of word indices that currently contain pair.
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

    # Initial counts of all pairs in the training text.
    pair_counts: PairCountDict = defaultdict(int)
    # Inverted index: pair -> set of word indices that contain this pair.
    pair_map: PairMap = defaultdict(set)

    for i, (word, count) in enumerate(zip(encoded_words, word_freqs)):
        for pair in zip(word, word[1:]):
            # Count every occurrence of pair in this word, weighted by frequency.
            pair_counts[pair] += count
            # pair_map tracks membership only (whether word i contains pair).
            pair_map[pair].add(i)

    return encoded_words, word_freqs, pair_counts, pair_map


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


def find_best_pair(
    heap: list[tuple[int, TokenPair]],
    pair_counts: Mapping[TokenPair, int],
) -> TokenPair | None:
    """Pop heap until finding a non-stale best pair.

    The heap is lazily updated, so it may contain stale entries
    (outdated frequency or pairs no longer present in pair_counts).

    Args:
        heap: Max-heap emulated with (-count, pair) entries.
        pair_counts: Ground-truth pair frequencies.

    Returns:
        The best valid pair if found, otherwise None.
    """
    best_pair: TokenPair | None = None
    while heap:
        neg_freq, pair = heapq.heappop(heap)
        if pair in pair_counts and pair_counts[pair] == -neg_freq:
            best_pair = pair
            break

    return best_pair


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

    def get_worker_segment_boundaries(self, num_desired_chunks: int) -> list[int]:
        """Find byte boundaries to split the corpus for parallel workers.

        Splits at newlines and/or special-token boundaries to avoid cutting
        special tokens across segment boundaries.

        Args:
            num_desired_chunks: Target number of segments (typically num_workers).

        Returns:
            List of byte offsets [b0, b1, ..., bn] where segments are
            (b0,b1), (b1,b2), ..., (b_{n-1}, bn). b0=0, bn=file_size.
        """
        # Match newlines; if special tokens exist, also match them.
        if not self.special_tokens:
            delimiter_pattern = regex.compile(b"\n")
        else:
            escaped = b"|".join(
                [regex.escape(t.encode("utf-8")) for t in self.special_tokens]
            )
            delimiter_pattern = regex.compile(b"\n|(?:" + escaped + b")")

        file_size = self.file_path.stat().st_size
        chunk_size = file_size // num_desired_chunks
        boundaries: list[int] = [0]

        with open(self.file_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for i in range(1, num_desired_chunks):
                    target = i * chunk_size

                    # Check if the previous search already jumped past this target.
                    if target <= boundaries[-1]:
                        continue

                    match = delimiter_pattern.search(mm, target)
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

        Maintains two synchronized structures for fast updates:
        - pair_counts: weighted occurrence count of each pair.
        - pair_map: word-index membership for each pair.

        Returns:
            None. Populates self.merge_rules and self.vocab; calls merge_vocab().
        """
        boundaries = self.get_worker_segment_boundaries(self.num_workers)
        segments = [
            (self.file_path, start, end, self.special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]

        with Pool(self.num_workers) as p:
            per_segment_word_counts = p.map(pretokenize_worker, segments)

        encoded_words, word_freqs, pair_counts, pair_map = merge_word_counts(
            per_segment_word_counts
        )
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = find_best_pair(heap, pair_counts)
            if best_pair is None:
                raise RuntimeError(
                    "Inconsistent heap state: pair_counts is non-empty but "
                    "no valid heap entry was found. pair_counts is the ground truth "
                    "and heap is lazily updated; this indicates missing/incorrect "
                    f"heap pushes. pair_counts_size={len(pair_counts)}, "
                    f"remaining_heap_entries={len(heap)}"
                )

            # Pick highest-frequency pair (deterministic tie-break by pair value).
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            # Only words containing best_pair can change after this merge.
            affected_indices = tuple(pair_map[best_pair])

            for idx in affected_indices:
                word = encoded_words[idx]
                freq = word_freqs[idx]

                # Pair counts before the merge.
                old_pair_counts = Counter(zip(word, word[1:]))
                # Apply the merge in place.
                apply_merge_in_place(word, best_pair, new_id)
                # Pair counts after the merge.
                new_pair_counts = Counter(zip(word, word[1:]))

                # Get all pairs before and after the merge.
                all_changed_pairs = old_pair_counts.keys() | new_pair_counts.keys()
                for pair in all_changed_pairs:
                    old_count, new_count = old_pair_counts[pair], new_pair_counts[pair]
                    if old_count == new_count:
                        continue

                    if old_count == 0 and new_count > 0:
                        # Pair is new.
                        pair_map[pair].add(idx)
                    elif old_count > 0 and new_count == 0:
                        # Pair is gone.
                        pair_map[pair].remove(idx)
                    else:
                        # Pair is still there, but count has changed.
                        pass

                    pair_counts[pair] += (new_count - old_count) * freq
                    new_count = pair_counts[pair]
                    if new_count > 0:
                        heapq.heappush(heap, (-new_count, pair))
                    else:
                        del pair_counts[pair]
                        pair_map.pop(pair)

        self.merge_vocab()
