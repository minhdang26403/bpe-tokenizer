from __future__ import annotations

import mmap
from multiprocessing import Pool
from pathlib import Path

import regex  # type: ignore[import-untyped]

from .base_tokenizer import (
    GPT2_REGEX,
    BaseTokenizer,
    TokenId,
    TokenPair,
    WordCountDict,
    apply_merge,
    split_by_special_tokens,
)

GPT_2_BYTE_REGEX = regex.compile(
    rb"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

WordList = list[list[TokenId]]
WordCountList = list[int]


def _pretokenizer_worker(
    args: tuple[Path, int, int, set[bytes]],
) -> WordCountDict:
    file_path, start, end, special_tokens = args

    special_pattern = None
    if special_tokens:
        pattern_bytes = b"(" + b"|".join(regex.escape(t) for t in special_tokens) + b")"
        special_pattern = regex.compile(pattern_bytes)

    word_counts: WordCountDict = {}
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # This creates a copy. We can use memoryview to avoid one copy, but we need
            # ensure that its scope is inside the mmap object
            segment = mm[start:end]
            chunks = special_pattern.split(segment) if special_pattern else [segment]
            for chunk in chunks:
                if chunk in special_tokens:
                    continue
                words = GPT_2_BYTE_REGEX.findall(chunk)
                for word_bytes in words:
                    word_ids = tuple(word_bytes)
                    word_counts[word_ids] = word_counts.get(word_ids, 0) + 1

    return word_counts


def _merge_word_counts(
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


def _apply_merge_in_place(word: list[int], best_pair: TokenPair, new_id: TokenId):
    id1, id2 = best_pair
    i = 0
    write_idx = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == id1 and word[i + 1] == id2:
            word[write_idx] = new_id
            i += 2
        else:
            word[write_idx] = word[i]
            i += 1
        write_idx += 1

    del word[write_idx:]


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

    def load_corpus(self) -> str:
        raise NotImplementedError

    def _find_chunk_boundaries(self, num_desired_chunks) -> list[int]:
        if not self.encoded_special_tokens:
            pattern = regex.compile(b"\n")
        else:
            pattern = regex.compile(
                b"|".join(regex.escape(token) for token in self.encoded_special_tokens)
            )

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
        boundaries = self._find_chunk_boundaries(self.num_workers)

        segments = [
            (self.file_path, start, end, self.encoded_special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]

        with Pool(self.num_workers) as p:
            word_counts_list = p.map(_pretokenizer_worker, segments)

        word_list, word_freq_list = _merge_word_counts(word_counts_list)

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
                _apply_merge_in_place(word, best_pair, new_id)

        self.is_trained = True

    def _encode_word(self, word_bytes: bytes) -> list[TokenId]:
        ids = list(word_bytes)
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

    def _encode_chunk(self, chunk: str) -> list[TokenId]:
        ids = []
        words: list[str] = GPT2_REGEX.findall(chunk)

        for word in words:
            word_bytes = word.encode("utf-8")
            word_ids = self._encode_word(word_bytes)
            ids.extend(word_ids)

        return ids

    def encode(self, text: str) -> list[TokenId]:
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

        try:
            text_bytes = b"".join(
                self.vocab[id] if id in self.vocab else self.inverse_special_tokens[id]
                for id in token_ids
            )
        except KeyError as e:
            raise ValueError(f"invalid token id: {e.args[0]}")

        return text_bytes.decode(encoding="utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> OptimizedTokenizer:
        raise NotImplementedError
