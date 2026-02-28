from __future__ import annotations

from pathlib import Path

from bpe.base_tokenizer import GPT2_REGEX
from bpe.optimized_tokenizer import pretokenize_worker


def test_str_regex_handles_non_ascii_words() -> None:
    """Current str-based GPT-2 regex behavior for Unicode text."""
    samples = [
        "你好世界",
        "안녕하세요 세계",
        "emoji 😀 test",
    ]
    for text in samples:
        assert GPT2_REGEX.findall(text)


def test_optimized_worker_matches_str_regex_mixed_script(tmp_path: Path) -> None:
    """Optimized worker should now follow str-regex mixed-script boundaries."""
    text = "abc你好123"
    corpus_path = tmp_path / "train.txt"
    corpus_path.write_text(text, encoding="utf-8")

    str_tokens = GPT2_REGEX.findall(text)
    assert str_tokens == ["abc你好", "123"]

    word_counts = pretokenize_worker((corpus_path, 0, corpus_path.stat().st_size, {}))
    worker_tokens = [bytes(ids).decode("utf-8") for ids in word_counts.keys()]

    # Worker pretokenization should exactly match str regex token boundaries.
    assert worker_tokens == str_tokens
