# BPE Tokenizer (Educational)

Minimal Python project for implementing and optimizing a Byte-Pair Encoding (BPE) tokenizer.

## Project layout

- `src/bpe.py`: BPE implementation (training, naive encode, cached encode, decode)
- `tests/test_bpe.py`: unit tests with `unittest`
- `benchmarks/benchmark_bpe.py`: quick benchmark for encode speedup
- `data/`: place corpora/text files for experiments

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run benchmark

```bash
python benchmarks/benchmark_bpe.py
```

Use the benchmark as a baseline, then optimize internals in `src/bpe.py` and compare speedups.
