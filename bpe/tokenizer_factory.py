from __future__ import annotations

from pathlib import Path
from typing import Type

from .base_tokenizer import BaseTokenizer
from .naive_tokenizer import NaiveTokenizer
from .optimized_tokenizer import OptimizedTokenizer

DEFAULT_SPECIAL_TOKENS: dict[str, int] = {"<|endoftext|>": 50256}

TOKENIZER_REGISTRY: dict[str, Type[BaseTokenizer]] = {
    "naive": NaiveTokenizer,
    "optimized": OptimizedTokenizer,
}


def get_tokenizer_class(name: str) -> Type[BaseTokenizer]:
    """Resolve a tokenizer implementation name to its class."""
    try:
        return TOKENIZER_REGISTRY[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(TOKENIZER_REGISTRY))
        raise ValueError(f"Unknown tokenizer '{name}'. Allowed values: {allowed}") from exc


def create_tokenizer(
    implementation: str,
    file_path: str | Path,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> BaseTokenizer:
    """Create tokenizer instance with shared defaults."""
    tokenizer_cls = get_tokenizer_class(implementation)
    return tokenizer_cls(
        file_path=Path(file_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens or DEFAULT_SPECIAL_TOKENS,
    )
