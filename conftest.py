from __future__ import annotations

from pathlib import Path
from typing import Type

import pytest

from bpe import BaseTokenizer
from bpe.tokenizer_factory import TOKENIZER_REGISTRY, create_tokenizer


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tokenizer",
        action="store",
        default="all",
        choices=["naive", "optimized", "all"],
        help="Select tokenizer implementation(s) to test.",
    )
    parser.addoption(
        "--vocab-size",
        action="store",
        type=int,
        default=512,
        help="Vocabulary size passed to the tokenizer constructor.",
    )
    parser.addoption(
        "--corpus-path",
        action="store",
        default=None,
        help=(
            "Path to training corpus file. If omitted, a tiny temporary corpus is used."
        ),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "tokenizer_cls" not in metafunc.fixturenames:
        return

    selected = metafunc.config.getoption("--tokenizer")
    if selected == "all":
        names = ["naive", "optimized"]
    else:
        names = [selected]

    classes = [TOKENIZER_REGISTRY[name] for name in names]
    metafunc.parametrize("tokenizer_cls", classes, ids=names, scope="session")


@pytest.fixture(scope="session")
def corpus_file(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    configured_path = request.config.getoption("--corpus-path")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.exists():
            raise pytest.UsageError(f"--corpus-path does not exist: {path}")
        if not path.is_file():
            raise pytest.UsageError(f"--corpus-path must point to a file: {path}")
        return path

    corpus = (
        "low lower newest widest\n"
        "BPE tokenizer educational sample text\n"
        "special tokens should survive round trip\n"
    )
    path = tmp_path_factory.mktemp("corpus") / "train.txt"
    path.write_text(corpus, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def trained_tokenizer(
    tokenizer_cls: Type[BaseTokenizer],
    corpus_file: Path,
    request: pytest.FixtureRequest,
) -> BaseTokenizer:
    vocab_size = request.config.getoption("--vocab-size")
    implementation = (
        "naive" if tokenizer_cls is TOKENIZER_REGISTRY["naive"] else "optimized"
    )
    tokenizer = create_tokenizer(
        implementation=implementation,
        file_path=corpus_file,
        vocab_size=vocab_size,
    )
    try:
        tokenizer.train()
    except NotImplementedError as exc:
        pytest.skip(f"{tokenizer_cls.__name__} is not implemented yet: {exc}")
    return tokenizer
