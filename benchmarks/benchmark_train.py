from __future__ import annotations

import argparse
import time
from pathlib import Path

from bpe.tokenizer_factory import TOKENIZER_REGISTRY, create_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark BPE tokenizer training time."
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        choices=sorted(TOKENIZER_REGISTRY.keys()),
        help="Tokenizer implementation to benchmark.",
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        type=Path,
        help="Path to training corpus text file.",
    )
    parser.add_argument(
        "--vocab-size",
        required=True,
        type=int,
        help="Target vocabulary size for training.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3).",
    )
    return parser.parse_args()


def benchmark_training(
    implementation: str,
    corpus_path: Path,
    vocab_size: int,
    runs: int,
) -> list[float]:
    """Run `train()` multiple times and return elapsed seconds per run."""
    durations: list[float] = []
    for _ in range(runs):
        tokenizer = create_tokenizer(
            implementation=implementation,
            file_path=corpus_path,
            vocab_size=vocab_size,
        )
        start = time.perf_counter()
        tokenizer.train()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    """CLI entrypoint for training benchmark."""
    args = parse_args()

    corpus_path = args.corpus_path.expanduser().resolve()
    if not corpus_path.exists() or not corpus_path.is_file():
        raise FileNotFoundError(f"Invalid --corpus-path: {corpus_path}")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    durations = benchmark_training(
        implementation=args.tokenizer,
        corpus_path=corpus_path,
        vocab_size=args.vocab_size,
        runs=args.runs,
    )
    best = min(durations)
    avg = sum(durations) / len(durations)

    print("Benchmark: tokenizer training")
    print(f"Implementation: {args.tokenizer}")
    print(f"Corpus path: {corpus_path}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Runs: {args.runs}")
    print("Run times (s): " + ", ".join(f"{d:.6f}" for d in durations))
    print(f"Best time (s): {best:.6f}")
    print(f"Avg time (s):  {avg:.6f}")


if __name__ == "__main__":
    main()
