# BPE Tokenizer (Naive + Optimized)

Educational Byte-Pair Encoding (BPE) tokenizer project with two implementations:

- `NaiveTokenizer`: straightforward baseline, easy to read.
- `OptimizedTokenizer`: faster training via parallel pretokenization + incremental pair updates.

The two implementations share the same public API (`train`, `encode`, `decode`) and are tested for behavioral consistency.

## Features

- Byte-level base vocabulary (`0..255`)
- GPT-2 style regex pretokenization
- Support for explicit special tokens (default: `<|endoftext|>`)
- Deterministic merge rule learning
- Parallel training path in optimized implementation
- Benchmark CLI for training-time comparison

## Project layout

- `bpe/base_tokenizer.py`
  - Shared abstractions/types/constants
  - Shared helpers such as special-token splitting
  - Shared `encode`/`decode` logic
- `bpe/naive_tokenizer.py`
  - Baseline BPE training loop (recomputes pair counts each merge)
- `bpe/optimized_tokenizer.py`
  - Parallel pretokenization
  - Incremental `pair_counts` + `pair_map` updates
  - Lazy max-heap selection of best pair
- `bpe/tokenizer_factory.py`
  - Registry and `create_tokenizer(...)` helper
- `tests/`
  - Unit and regression tests (details below)
- `benchmarks/benchmark_train.py`
  - CLI benchmark for `train()` runtime
- `data/`
  - Sample corpora for experiments/benchmarking

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick usage

```python
from bpe.naive_tokenizer import NaiveTokenizer

tokenizer = NaiveTokenizer(
    file_path="data/corpus.en",
    vocab_size=512,
    special_tokens={"<|endoftext|>": 50256},
)

tokenizer.train()
ids = tokenizer.encode("hello <|endoftext|> world")
text = tokenizer.decode(ids)
print(ids)
print(text)
```

You can swap `NaiveTokenizer` with `OptimizedTokenizer` without changing call sites.

## Unit tests

### Test commands

Run all tests:

```bash
pytest -v
```

Run only one implementation:

```bash
pytest -v --tokenizer naive
pytest -v --tokenizer optimized
```

Run with a larger corpus and custom vocab size:

```bash
pytest -v \
  --tokenizer optimized \
  --vocab-size 2048 \
  --corpus-path data/tinystories_sample_5M.txt
```

Run one test by name:

```bash
pytest -v -k "test_encode_decode_identity"
```

Show debug `print(...)` output from tests:

```bash
pytest -v -s tests/test_tokenizer.py
```

### Important test fixture behavior

If `--corpus-path` is omitted, `conftest.py` creates a tiny temporary corpus.
This keeps default tests fast, but runtime does not reflect large-corpus training performance.

### What each test file covers

- `tests/test_tokenizer.py`
  - End-to-end round-trip identity (`encode` -> `decode`)
  - Special token preservation in round-trip
  - Debug token-id printing for inspection with `-s`

- `tests/test_pretokenization_behavior.py`
  - Unicode/non-ASCII pretokenization sanity checks
  - Mixed-script behavior check in optimized worker pretokenization

- `tests/test_optimized_vs_naive_regression.py`
  - Regression corpus that previously exposed merge-rule divergence
  - Validates optimized and naive produce matching encodings and merge-rule order

- `tests/test_optimized_heap_and_consistency.py`
  - `find_best_pair(...)` stale-heap handling
  - Additional naive-vs-optimized consistency case (Unicode + special tokens)
  - Error-path test: invalid token id in `decode` raises `ValueError`

## Benchmarking

Benchmark training time with the provided CLI:

```bash
python benchmarks/benchmark_train.py \
  --tokenizer naive \
  --corpus-path data/tinystories_sample_5M.txt \
  --vocab-size 1024 \
  --runs 3
```

Optimized version:

```bash
python benchmarks/benchmark_train.py \
  --tokenizer optimized \
  --corpus-path data/tinystories_sample_5M.txt \
  --vocab-size 1024 \
  --runs 3
```

Output includes per-run times, best time, and average time.

## Notes on deterministic behavior

- Tie-breaking is deterministic in both implementations.
- Heap entries in optimized training are lazily updated; stale entries are filtered.
- Tests include explicit consistency checks between `NaiveTokenizer` and `OptimizedTokenizer`.
