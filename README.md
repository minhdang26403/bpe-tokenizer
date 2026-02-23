# BPE Tokenizer (Educational)

Minimal Python project for implementing and optimizing a Byte-Pair Encoding (BPE) tokenizer.

## Project layout

- `bpe/base_tokenizer.py`: abstract tokenizer contract
- `bpe/naive_tokenizer.py`: naive implementation (you fill in)
- `bpe/optimized_tokenizer.py`: optimized implementation (you fill in)
- `tests/test_tokenizer_contract.py`: shared tests for all implementations
- `conftest.py`: pytest CLI options + tokenizer selection fixture
- `data/`: place corpora/text files for experiments

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

You can select implementation(s) from CLI:

```bash
pytest -q --tokenizer naive
pytest -q --tokenizer optimized
pytest -q --tokenizer all
```

You can also choose vocabulary size and corpus path:

```bash
pytest -q --vocab-size 200
pytest -q --corpus-path data/my_corpus.txt
pytest -q --tokenizer optimized --vocab-size 500 --corpus-path data/wiki.txt
```

And select a specific test by name:

```bash
pytest -q -k identity
pytest -q -k special --tokenizer optimized
```

By default, tests train each tokenizer once per test session and reuse that trained instance across tests for speed. If `--corpus-path` is not passed, pytest uses a tiny temporary corpus.
