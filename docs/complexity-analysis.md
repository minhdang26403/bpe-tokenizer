# Naive Tokenizer Complexity Analysis

This note summarizes the asymptotic time complexity of the `NaiveTokenizer`
training pipeline.

## Symbols

- `N`: number of bytes in the training corpus
- `U`: number of unique words after pretokenization (`word_counts` keys)
- `L̄`: average token-id length of a unique word
- `S = Σ len(word_i)` across unique words, so `S ≈ U * L̄`
- `P`: number of distinct adjacent pairs currently in `pair_counts`
- `M`: number of merge iterations actually executed

## Step-by-step cost

### 1) Pretokenization

In `NaiveTokenizer.pretokenize(...)`, the corpus is read and scanned once, then
tokens are counted.

- Cost: approximately `O(N)`

### 2) Build pair counts

In `get_pair_counts(...)`, we iterate adjacent pairs for every unique word.

- Cost: `O(S)` (equivalently `O(U * L̄)`)

### 3) Find best pair

In `train(...)`, selecting the best pair via `max(pair_counts, ...)` scans all
current pairs.

- Cost: `O(P)`
- Since `P <= S`, this is upper-bounded by `O(S)`

### 4) Rebuild merged word counts

For each unique word, `apply_merge(...)` is linear in word length, and we rebuild
the dictionary.

- Cost: `O(S)` (equivalently `O(U * L̄)`)

## Total training complexity

Per merge iteration:

- `O(S + P + S) = O(S + P)`, and with `P <= S`, this becomes `O(S)`

Across `M` merges:

- `O(M * S)` (equivalently `O(M * U * L̄)`)

Including pretokenization:

- `O(N + M * S)`

In most practical settings where training iterations dominate, this is often
reported as:

- `O(M * U * L̄)`

---

# Optimized Tokenizer Complexity Analysis

This section summarizes the asymptotic complexity of `OptimizedTokenizer`.

## Additional symbols

- `W`: number of pretokenization workers
- `Q`: number of distinct adjacent pairs (`len(pair_counts)`)
- `S_t`: total token-id length of words affected in merge step `t`
  - `S_t = Σ len(word_i)` for `i in affected_indices`

(`N`, `U`, `L̄`, `S`, `M` are the same as above.)

## Step-by-step cost

### 1) Find worker segment boundaries

`get_worker_segment_boundaries(...)` probes around estimated chunk boundaries and
searches forward to the next delimiter (newline or special token).

- Typical case (delimiter-dense corpora): near `O(W * c)` for small constant `c`
- Worst-like case (sparse delimiters): searches can scan long ranges, approaching
  `O(N)` total work

In practice, this is usually treated as approximately linear scan cost.

### 2) Parallel pretokenization

Workers process disjoint file segments with mmap + decode + regex tokenization.

- Total work: `O(N)`
- Ideal wall-clock: `O(N / W)` plus multiprocessing overhead and load imbalance

### 3) Merge per-segment counts and build initial structures

`merge_word_counts(...)` does:

- dedup/aggregate words: `O(U)` (average-case dict operations)
- build `pair_counts` + `pair_map` by scanning adjacent pairs of all unique words:
  `O(S)`

So total: `O(U + S) = O(S)`.

### 4) Build heap

The heap is built from current pair counts:

- heapify: `O(Q)`

### 5) Per-merge update loop

For each merge step:

1. Find valid best pair from lazy heap: amortized near `O(log Q)` per accepted
   entry (stale pops are filtered).
2. Update only affected words (`affected_indices`):
   - for one affected word of length `l`:
     - old/new `Counter(zip(...))` + in-place merge: `O(l)`
     - changed-pair updates may push to heap (`log Q` each), up to `O(l)` pushes
     - dominant per-word term: `O(l log Q)`
   - summing over affected words:
     - `O(S_t log Q)`

Per merge step: approximately `O(S_t log Q)`.

## Total training complexity (optimized)

Overall:

- preprocessing: `O(N)` total work (parallel wall-clock near `O(N / W)`)
- structure build: `O(S + Q)`
- merge loop: `Σ_{t=1..M} O(S_t log Q)`

So a useful summary is:

- `O(N + S + Q + Σ_t S_t log Q)`

Worst case (if most words are affected every merge, `S_t ≈ S`):

- `O(N + M * S * log Q)`

Typical case is much better when `S_t << S` for most steps.
