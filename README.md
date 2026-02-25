# BPE Tokenizer

Minimal Python project for implementing and optimizing a Byte-Pair Encoding (BPE) tokenizer.

## Current status

- `NaiveTokenizer` has a working baseline implementation for `train`, `encode`, and `decode`.
- `OptimizedTokenizer` is currently a skeleton for future optimization work.
- Test harness uses pytest with CLI flags to select tokenizer, vocab size, and corpus path.

## Project layout

- `bpe/base_tokenizer.py`: abstract interface + shared types, constants, and helper functions.
- `bpe/naive_tokenizer.py`: baseline BPE implementation.
- `bpe/optimized_tokenizer.py`: optimized implementation skeleton.
- `bpe/tokenizer_factory.py`: shared tokenizer registry + constructor helper.
- `bpe/__init__.py`: public exports.
- `tests/test_tokenizer.py`: tokenizer tests.
- `conftest.py`: pytest CLI options and fixtures.
- `benchmarks/benchmark_train.py`: training-time benchmark CLI.
- `data/`: corpus files for training experiments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Tokenizer interface

Both tokenizers share the same constructor/API:

```python
tokenizer = NaiveTokenizer(
    file_path="data/corpus.en",
    vocab_size=500,
    special_tokens={"<|endoftext|>": 500},
)

tokenizer.train()
ids = tokenizer.encode("hello <|endoftext|> world")
text = tokenizer.decode(ids)
```

### Notes

- `special_tokens` is `dict[str, int] | None`, not a tuple/list.
- `train()` uses the configured `vocab_size` from the constructor.
- `encode()` / `decode()` expect the tokenizer to be trained first.

## Shared helpers and constants

From `bpe/base_tokenizer.py`:

- `TokenId`, `TokenPair`, `WordCountDict`
- `BYTE_VOCAB_SIZE`, `UTF8_ENCODING`, `ENCODE_WORD_CACHE_SIZE`
- `GPT2_REGEX`
- `apply_merge(...)`, `get_pair_counts(...)`, `split_by_special_tokens(...)`

These are intended for reuse by both naive and optimized implementations.

## Tests

Run all tests:

```bash
pytest -q
```

Run by implementation:

```bash
pytest -q --tokenizer naive
pytest -q --tokenizer optimized
pytest -q --tokenizer all
```

Configure vocab size and corpus path:

```bash
pytest -q --vocab-size 200
pytest -q --corpus-path data/corpus.en
pytest -q --tokenizer naive --vocab-size 500 --corpus-path data/the-verdict.txt
```

Run specific tests:

```bash
pytest -q -k identity
pytest -q -k special --tokenizer naive
```

## Benchmark training time

Benchmark `train()` for a chosen implementation:

```bash
python benchmarks/benchmark_train.py \
  --tokenizer naive \
  --corpus-path data/corpus.en \
  --vocab-size 1000 \
  --runs 3
```

Switch implementation:

```bash
python benchmarks/benchmark_train.py \
  --tokenizer optimized \
  --corpus-path data/corpus.en \
  --vocab-size 1000
```
