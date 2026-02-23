from __future__ import annotations

import pytest

from bpe import BaseTokenizer


def test_encode_decode_identity(trained_tokenizer: BaseTokenizer) -> None:
    text = "BPE should reconstruct this line exactly."
    try:
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
    except NotImplementedError as exc:
        pytest.skip(f"{type(trained_tokenizer).__name__} is not implemented yet: {exc}")

    assert decoded == text


def test_special_token_roundtrip(trained_tokenizer: BaseTokenizer) -> None:
    special = "<|endoftext|>"
    text = f"hello {special} world"

    try:
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
    except NotImplementedError as exc:
        pytest.skip(f"{type(trained_tokenizer).__name__} is not implemented yet: {exc}")

    assert special in text
    assert decoded == text
