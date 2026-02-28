"""Base BPE tokenizer with shared encoding/decoding and special-token handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import regex  # type: ignore[import-untyped]

# Type aliases for tokenizer internals.
TokenId = int
TokenPair = tuple[TokenId, TokenId]  # (left_id, right_id) for merge rules.
WordCountDict = dict[tuple[TokenId, ...], int]  # word_ids -> count.

# First 256 ids are reserved for byte-level tokens (0x00-0xFF).
BASE_VOCAB_SIZE = 256
# GPT-2 style regex: splits on Unicode letters, numbers, whitespace, contractions.
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
GPT2_REGEX = regex.compile(GPT2_PATTERN)


def compile_special_tokens_pattern(
    special_tokens: dict[str, TokenId],
) -> regex.Pattern[str] | None:
    """Compile a regex that captures special token boundaries.

    Args:
        special_tokens: Mapping from special token text to token id.

    Returns:
        A compiled regex that captures any special token, or None when no
        special tokens are configured.
    """
    if not special_tokens:
        return None

    return regex.compile(
        r"(" + "|".join([regex.escape(token) for token in special_tokens]) + r")"
    )


def split_text_by_special_tokens(
    text: str,
    special_tokens_pattern: regex.Pattern[str] | None,
) -> list[str]:
    """Split text into chunks around special token boundaries.

    Args:
        text: Input string that may contain special tokens.
        special_tokens_pattern: Compiled pattern for special tokens.

    Returns:
        List of chunks (alternating text and matched special tokens). If no
        pattern is provided, returns [text].
    """
    if not special_tokens_pattern:
        return [text]

    return special_tokens_pattern.split(text)


def apply_merge(
    ids: list[TokenId] | tuple[TokenId, ...],
    best_pair: TokenPair,
    new_id: TokenId,
) -> list[TokenId]:
    """Replace all adjacent occurrences of best_pair in ids with new_id.

    Walks through the id sequence once. Whenever (ids[i], ids[i+1]) equals
    best_pair, emits new_id and skips both positions; otherwise emits ids[i]
    and advances by one.

    Args:
        ids: Sequence of token ids (list or tuple) to process.
        best_pair: The (left, right) token pair to merge.
        new_id: The token id that replaces each occurrence of best_pair.

    Returns:
        A new list of token ids with all best_pair occurrences replaced.
    """
    new_ids: list[TokenId] = []
    i = 0
    while i < len(ids):
        # Check if current position starts a best_pair match.
        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


class BaseTokenizer(ABC):
    """Base tokenizer shared by all byte-pair encoding tokenizer implementations."""

    def __init__(
        self,
        file_path: str | Path,
        vocab_size: int,
        special_tokens: dict[str, TokenId] | None,
    ) -> None:
        """Initialize shared config and data structures for encoding/decoding.

        Args:
            file_path: Path to the corpus file used for training.
            vocab_size: Target vocabulary size (>= 256; first 256 are byte codes).
            special_tokens: Optional map from special token strings to their ids.
                Special token ids must be >= vocab_size and unique.

        Raises:
            ValueError: If vocab_size < 256 or special token ids are invalid.
        """
        self.file_path = Path(file_path)
        self.vocab_size = vocab_size
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= {BASE_VOCAB_SIZE}, got {vocab_size}"
            )

        # Merge rules: (left_id, right_id) -> merged_id; filled during train().
        self.merge_rules: dict[TokenPair, TokenId] = {}
        # Base vocab: id 0-255 map to single-byte tokens (byte-level BPE).
        self.vocab = {id: bytes([id]) for id in range(BASE_VOCAB_SIZE)}

        self.special_tokens = special_tokens if special_tokens else {}

        # Inverse map for decode: token_id -> bytes for special tokens only.
        self.inverse_special_tokens: dict[TokenId, bytes] = {}
        for token, token_id in self.special_tokens.items():
            if token_id < self.vocab_size:
                raise ValueError(
                    f"special token id {token_id} must be >= vocab_size "
                    f"({self.vocab_size}); ids in [0, vocab_size) are reserved"
                )
            if token_id in self.inverse_special_tokens:
                raise ValueError(f"duplicate special token id {token_id}")
            self.inverse_special_tokens[token_id] = token.encode("utf-8")

        # Regex to split text on special token boundaries (capturing group).
        self.special_tokens_pattern = compile_special_tokens_pattern(
            self.special_tokens
        )

        # FIFO cache for encode_word results; improves repeated encode calls.
        self.cache: dict[str, tuple[TokenId, ...]] = {}
        self.max_cache_size = 32768

    def merge_vocab(self) -> None:
        """Build a unified vocab for fast decode lookup.

        Merges the regular vocab (byte tokens + learned merges) with the special
        token inverse map. Call this after train() and before decode().

        Returns:
            None. Updates self.unified_vocab in place.
        """
        self.unified_vocab = {**self.vocab, **self.inverse_special_tokens}

    @abstractmethod
    def train(self) -> None:
        """Train tokenizer to the configured vocabulary size.

        Subclasses must populate self.merge_rules and self.vocab.
        """
        raise NotImplementedError

    def encode_word(self, word: str) -> tuple[TokenId, ...]:
        """Encode a single pre-tokenized word into token ids using merge rules.

        Starts with byte-level ids and greedily applies merges in order of
        rule precedence until no more apply.

        Args:
            word: A single "word" from GPT-2 style pre-tokenization (letters,
                numbers, punctuation, etc.). Must be valid UTF-8.

        Returns:
            Tuple of token ids representing the encoded word (cached).
        """
        if word in self.cache:
            return self.cache[word]

        # Start with byte-level ids (0-255).
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # Pick the earliest applicable merge (by rule order).
            best_pair = min(
                (pair for pair in zip(ids, ids[1:]) if pair in self.merge_rules),
                key=lambda p: self.merge_rules[p],
                default=None,
            )

            if not best_pair:
                break

            ids = apply_merge(ids, best_pair, self.merge_rules[best_pair])

        # FIFO eviction when cache is full.
        if len(self.cache) >= self.max_cache_size:
            # Pop the first key (oldest) - Python dicts preserve order!
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        word_ids = tuple(ids)
        # Store in cache for future calls.
        self.cache[word] = word_ids
        return word_ids

    def encode_chunk(self, chunk: str) -> list[TokenId]:
        """Encode a chunk of text (no special tokens inside) into token ids.

        Args:
            chunk: Text segment that does not contain any special token strings.

        Returns:
            List of token ids for the entire chunk.
        """
        ids: list[TokenId] = []
        # GPT-2 style: split into words (letters, numbers, punctuation, etc.).
        words: list[str] = GPT2_REGEX.findall(chunk)
        for word in words:
            word_ids = self.encode_word(word)
            ids.extend(word_ids)

        return ids

    def encode(self, text: str) -> list[TokenId]:
        """Encode text into token ids, preserving explicit special tokens.

        Splits on special token boundaries first; special token chunks are
        emitted as their fixed ids; other chunks are encoded normally.

        Args:
            text: Input string (may contain special token substrings).

        Returns:
            List of token ids for the full sequence.
        """
        chunks = split_text_by_special_tokens(
            text,
            self.special_tokens_pattern,
        )
        ids: list[TokenId] = []
        for chunk in chunks:
            # Chunk is either a special token (emit id) or normal text (encode).
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_chunk(chunk))

        return ids

    def decode(self, token_ids: list[TokenId]) -> str:
        """Decode token ids back to a UTF-8 string.

        Args:
            token_ids: Sequence of token ids from encode().

        Returns:
            Decoded UTF-8 string.

        Raises:
            ValueError: If any token id is unknown (not in unified_vocab).
                Call merge_vocab() after train() before using decode().
        """
        try:
            text_bytes = b"".join([self.unified_vocab[id] for id in token_ids])
        except KeyError as e:
            raise ValueError(f"invalid token id {e.args[0]}")

        text = text_bytes.decode("utf-8", errors="replace")
        return text
