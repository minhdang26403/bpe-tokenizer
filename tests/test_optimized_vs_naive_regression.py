from __future__ import annotations

from pathlib import Path

from bpe.naive_tokenizer import NaiveTokenizer
from bpe.optimized_tokenizer import OptimizedTokenizer


def test_optimized_matches_naive_on_repeated_pair_corpus(tmp_path: Path) -> None:
    """Regression corpus that previously exposed naive/optimized divergence."""
    corpus = (
        ("aaaaaa aaaaaa aaaaaa\n" * 200)
        + ("abababab abab abab\n" * 100)
        + ("abc你好123 abc你好123\n" * 50)
    )
    corpus_path = tmp_path / "train.txt"
    corpus_path.write_text(corpus, encoding="utf-8")

    special_tokens = {"<|endoftext|>": 50256}
    vocab_size = 280

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

    probe_text = "aaaaaa abab abc你好123 aaaaaa"
    assert optimized.encode(probe_text) == naive.encode(probe_text)

    naive_rule_by_id = {rule_id: pair for pair, rule_id in naive.merge_rules.items()}
    optimized_rule_by_id = {
        rule_id: pair for pair, rule_id in optimized.merge_rules.items()
    }
    max_rule_id = min(len(naive.vocab), len(optimized.vocab))
    for rule_id in range(256, max_rule_id):
        assert optimized_rule_by_id.get(rule_id) == naive_rule_by_id.get(rule_id)
