from __future__ import annotations

import pytest

from bpe import BaseTokenizer


def _debug_print_tokens(tokenizer: BaseTokenizer, token_ids: list[int]) -> None:
    """Print token ids and their byte mapping for debugging (`pytest -s`)."""
    print(f"encoded: {token_ids}")
    for token_id in token_ids:
        if token_id in tokenizer.vocab:
            token_bytes = tokenizer.vocab[token_id]
        elif token_id in tokenizer.inverse_special_tokens:
            token_bytes = tokenizer.inverse_special_tokens[token_id]
        else:
            token_bytes = b"<unknown>"
        print(f"{token_id} -> {token_bytes!r}")


def test_encode_decode_identity(trained_tokenizer: BaseTokenizer) -> None:
    text = (
        "On a rainy evening, the old library smelled of paper and dust, "
        "and every lamp along the aisle flickered softly."
    )
    try:
        encoded = trained_tokenizer.encode(text)
        _debug_print_tokens(trained_tokenizer, encoded)
        decoded = trained_tokenizer.decode(encoded)
    except NotImplementedError as exc:
        pytest.skip(f"{type(trained_tokenizer).__name__} is not implemented yet: {exc}")

    assert decoded == text


def test_special_token_roundtrip(trained_tokenizer: BaseTokenizer) -> None:
    special = "<|endoftext|>"
    text = (
        f"The spacecraft log ended here {special} "
        "and resumed later with a short systems report."
    )

    try:
        encoded = trained_tokenizer.encode(text)
        _debug_print_tokens(trained_tokenizer, encoded)
        decoded = trained_tokenizer.decode(encoded)
    except NotImplementedError as exc:
        pytest.skip(f"{type(trained_tokenizer).__name__} is not implemented yet: {exc}")

    assert special in text
    assert decoded == text
