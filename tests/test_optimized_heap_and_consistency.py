from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from bpe.naive_tokenizer import NaiveTokenizer
from bpe.optimized_tokenizer import OptimizedTokenizer, find_best_pair


def test_find_best_pair_skips_stale_entries() -> None:
    """Heap lookup should ignore stale entries and return a valid best pair."""
    target_pair = (1, 2)
    pair_counts = defaultdict(int, {target_pair: 5, (3, 4): 4})
    heap = [
        (-9, target_pair),  # stale, outdated count
        (-5, target_pair),  # current valid count
        (-4, (3, 4)),
    ]
    assert find_best_pair(heap, pair_counts) == target_pair


def test_find_best_pair_returns_none_when_only_stale_entries() -> None:
    """Return None when heap has no valid entry for current pair_counts."""
    pair_counts = defaultdict(int, {(1, 2): 3})
    heap = [(-5, (1, 2)), (-2, (7, 8))]
    assert find_best_pair(heap, pair_counts) is None


def test_optimized_matches_naive_on_unicode_and_special_tokens(tmp_path: Path) -> None:
    """Naive and optimized tokenizers should remain behaviorally consistent."""
    corpus = (
        "hello world\n"
        "abc你好123\n"
        "안녕하세요 세계\n"
        "<|endoftext|> should be skipped in training\n"
        "emoji 😀 test\n"
    )
    corpus_path = tmp_path / "train.txt"
    corpus_path.write_text(corpus, encoding="utf-8")

    special_tokens = {"<|endoftext|>": 50256}
    vocab_size = 300
    naive = NaiveTokenizer(
        file_path=corpus_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    optimized = OptimizedTokenizer(
        file_path=corpus_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=4,
    )

    naive.train()
    optimized.train()

    probe = "abc你好123 <|endoftext|> 안녕하세요 😀"
    naive_ids = naive.encode(probe)
    optimized_ids = optimized.encode(probe)
    assert optimized_ids == naive_ids
    assert naive.decode(naive_ids) == probe
    assert optimized.decode(optimized_ids) == probe


def test_decode_invalid_token_id_raises(trained_tokenizer) -> None:
    """Decoding unknown ids should raise ValueError for all implementations."""
    with pytest.raises(ValueError):
        trained_tokenizer.decode([10**9])
