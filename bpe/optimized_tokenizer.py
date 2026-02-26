from __future__ import annotations

import mmap
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

GPT2_BYTES_REGEX = regex.compile(GPT2_PATTERN.encode("utf-8"))

WordList = list[list[TokenId]]
WordCountList = list[int]


def pretokenize_worker(
    args: tuple[Path, int, int, set[bytes]],
) -> WordCountDict:
    """Pretokenize a file segment into word counts. Single tuple arg for Pool.map.

    Pool.map passes one argument per call; we pack (path, start, end, special_tokens)
    into a tuple for this reason.
    """
    file_path, start, end, special_tokens = args

    # Split on special tokens; capturing group makes separators appear in result.
    split_pattern = None
    if special_tokens:
        alternation = b"|".join(regex.escape(t) for t in special_tokens)
        split_pattern = regex.compile(b"(" + alternation + b")")

    word_counts: WordCountDict = {}
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # This creates a copy. We can use memoryview to avoid one copy, but we need
            # ensure that its scope is inside the mmap object
            segment = memoryview(mm)[start:end]
            chunks = split_pattern.split(segment) if split_pattern else [segment]
            for chunk in chunks:
                if chunk in special_tokens:
                    continue
                words = GPT2_BYTES_REGEX.findall(chunk)
                for word_bytes in words:
                    word_ids = tuple(word_bytes)
                    word_counts[word_ids] = word_counts.get(word_ids, 0) + 1
            segment.release()

    return word_counts


def merge_word_counts(
    word_counts_list: list[WordCountDict],
) -> tuple[WordList, WordCountList]:

    word_to_idx = {}
    word_list: WordList = []
    word_freq_list: WordCountList = []

    for word_counts in word_counts_list:
        for word_ids, count in word_counts.items():
            if word_ids not in word_to_idx:
                word_to_idx[word_ids] = len(word_list)
                word_list.append(list(word_ids))
                word_freq_list.append(count)
            else:
                idx = word_to_idx[word_ids]
                word_freq_list[idx] += count

    return word_list, word_freq_list


def apply_merge_in_place(
    word_ids: list[TokenId], best_pair: TokenPair, new_id: TokenId
):
    i = 0
    write_idx = 0
    while i < len(word_ids):
        if (
            i < len(word_ids) - 1
            and word_ids[i] == best_pair[0]
            and word_ids[i + 1] == best_pair[1]
        ):
            word_ids[write_idx] = new_id
            i += 2
        else:
            word_ids[write_idx] = word_ids[i]
            i += 1
        write_idx += 1

    del word_ids[write_idx:]


class OptimizedTokenizer(BaseTokenizer):
    """Optimized byte-pair encoding tokenizer implementation."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__(
            file_path=file_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        self.num_workers = num_workers

        self.encoded_special_tokens = {
            token.encode("utf-8") for token in self.special_tokens
        }

    def get_worker_segment_boundaries(
        self, num_desired_chunks: int
    ) -> list[int]:
        """Find byte boundaries to split the file for parallel workers.

        Splits at newlines and/or special token boundaries so we never cut
        mid-line or mid-token. Uses newlines when present; adds special
        tokens when configured so chunks stay balanced even when tokens are rare.
        """
        # Match newlines; if special tokens exist, also match them for more splits.
        if not self.encoded_special_tokens:
            pattern = regex.compile(b"\n")
        else:
            escaped = b"|".join(
                regex.escape(t) for t in self.encoded_special_tokens
            )
            pattern = regex.compile(b"\n|(?:" + escaped + b")")

        file_size = self.file_path.stat().st_size
        chunk_size = file_size // num_desired_chunks
        boundaries = [0]

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
                        break

        if boundaries[-1] != file_size:
            boundaries.append(file_size)

        return boundaries

    def train(self) -> None:
        boundaries = self.get_worker_segment_boundaries(self.num_workers)

        segments = [
            (self.file_path, start, end, self.encoded_special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]

        with Pool(self.num_workers) as p:
            word_counts_list = p.map(pretokenize_worker, segments)

        word_list, word_freq_list = merge_word_counts(word_counts_list)

        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts: dict[TokenPair, int] = {}
            for word, count in zip(word_list, word_freq_list):
                for pair in zip(word, word[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merge_rules[best_pair] = new_id

            for word in word_list:
                apply_merge_in_place(word, best_pair, new_id)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> OptimizedTokenizer:
        raise NotImplementedError
