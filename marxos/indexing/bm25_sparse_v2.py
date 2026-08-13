"""Deterministic, corpus-fitted BM25 sparse vectors for the v2 index.

The fitted encoder is deliberately self-contained: it needs no downloaded
tokenizer and persists every statistic required to reproduce query vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


FORMAT_VERSION = "bm25-sparse-v2"
TOKENIZER_VERSION = "marxos-zh-ngram-v1"
_MAX_TERM_ID = (1 << 32) - 2


@dataclass(frozen=True)
class BM25Config:
    """Versioned BM25 and tokenizer configuration."""

    k1: float = 1.2
    b: float = 0.75
    tokenizer_version: str = TOKENIZER_VERSION
    chinese_ngram_sizes: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")
        if not self.chinese_ngram_sizes or any(size <= 0 for size in self.chinese_ngram_sizes):
            raise ValueError("chinese_ngram_sizes must contain positive integers")


@dataclass(frozen=True)
class BM25Stats:
    """Immutable fitted corpus statistics."""

    document_count: int
    average_document_length: float
    document_frequencies: tuple[tuple[str, int], ...]


def tokenize_bm25_v2(text: str, config: BM25Config | None = None) -> tuple[str, ...]:
    """Tokenize Latin words and overlapping Chinese n-grams deterministically."""

    active = config or BM25Config()
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    tokens: list[str] = re.findall(r"[a-z0-9]+", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(run) <= min(active.chinese_ngram_sizes):
            tokens.append(run)
            continue
        for size in active.chinese_ngram_sizes:
            tokens.extend(run[index:index + size] for index in range(len(run) - size + 1))
    return tuple(tokens)


class BM25SparseEncoderV2:
    """A fitted BM25 encoder producing Milvus-compatible sparse mappings."""

    def __init__(
        self,
        *,
        config: BM25Config,
        stats: BM25Stats,
        vocabulary: Sequence[tuple[str, int]],
    ) -> None:
        self.config = config
        self.stats = stats
        ordered_vocabulary = tuple(sorted((str(term), int(term_id)) for term, term_id in vocabulary))
        self._vocabulary = ordered_vocabulary
        self._term_ids: Mapping[str, int] = MappingProxyType(dict(ordered_vocabulary))
        self._document_frequencies: Mapping[str, int] = MappingProxyType(dict(stats.document_frequencies))

    @classmethod
    def fit(
        cls,
        documents: Iterable[str],
        *,
        config: BM25Config | None = None,
    ) -> "BM25SparseEncoderV2":
        active = config or BM25Config()
        tokenized = [tokenize_bm25_v2(text, active) for text in documents]
        frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            frequencies.update(set(tokens))
        used_ids: set[int] = set()
        vocabulary_items: list[tuple[str, int]] = []
        for term in sorted(frequencies):
            term_id = cls._stable_term_id(term)
            while term_id in used_ids:
                term_id = (term_id % _MAX_TERM_ID) + 1
            used_ids.add(term_id)
            vocabulary_items.append((term, term_id))
        vocabulary = tuple(vocabulary_items)
        document_count = len(tokenized)
        total_length = sum(len(tokens) for tokens in tokenized)
        stats = BM25Stats(
            document_count=document_count,
            average_document_length=(total_length / document_count if document_count else 0.0),
            document_frequencies=tuple(sorted(frequencies.items())),
        )
        return cls(config=active, stats=stats, vocabulary=vocabulary)

    @property
    def vocabulary(self) -> tuple[tuple[str, int], ...]:
        return self._vocabulary

    def term_id(self, term: str) -> int | None:
        return self._term_ids.get(term)

    def embed_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self._embed_document(text) for text in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        counts = Counter(tokenize_bm25_v2(text, self.config))
        return {
            self._term_ids[term]: self._idf(term) * (1.0 + math.log(float(count)))
            for term, count in sorted(counts.items())
            if term in self._term_ids
        }

    def _embed_document(self, text: str) -> dict[int, float]:
        tokens = tokenize_bm25_v2(text, self.config)
        if not tokens or not self.stats.average_document_length:
            return {}
        counts = Counter(tokens)
        length_ratio = len(tokens) / self.stats.average_document_length
        normalization = self.config.k1 * (1.0 - self.config.b + self.config.b * length_ratio)
        return {
            self._term_ids[term]: self._idf(term)
            * (count * (self.config.k1 + 1.0) / (count + normalization))
            for term, count in sorted(counts.items())
            if term in self._term_ids
        }

    def _idf(self, term: str) -> float:
        frequency = self._document_frequencies[term]
        numerator = self.stats.document_count - frequency + 0.5
        return math.log1p(numerator / (frequency + 0.5))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        payload = self._payload_without_checksum()
        payload["checksum"] = self._checksum(payload)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25SparseEncoderV2":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = payload.pop("checksum", None)
        if not checksum or checksum != cls._checksum(payload):
            raise ValueError("BM25 manifest checksum mismatch")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported BM25 format: {payload.get('format_version')}")
        config_payload = dict(payload["config"])
        config_payload["chinese_ngram_sizes"] = tuple(config_payload["chinese_ngram_sizes"])
        stats_payload = payload["stats"]
        return cls(
            config=BM25Config(**config_payload),
            stats=BM25Stats(
                document_count=int(stats_payload["document_count"]),
                average_document_length=float(stats_payload["average_document_length"]),
                document_frequencies=tuple(
                    (str(term), int(frequency))
                    for term, frequency in stats_payload["document_frequencies"]
                ),
            ),
            vocabulary=tuple((str(term), int(term_id)) for term, term_id in payload["vocabulary"]),
        )

    def _payload_without_checksum(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "format_version": FORMAT_VERSION,
            "stats": {
                "average_document_length": self.stats.average_document_length,
                "document_count": self.stats.document_count,
                "document_frequencies": self.stats.document_frequencies,
            },
            "vocabulary": self._vocabulary,
        }

    @staticmethod
    def _checksum(payload: Mapping[str, object]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _stable_term_id(term: str) -> int:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
        return (int.from_bytes(digest, "big") % _MAX_TERM_ID) + 1


__all__ = [
    "BM25Config",
    "BM25SparseEncoderV2",
    "BM25Stats",
    "FORMAT_VERSION",
    "TOKENIZER_VERSION",
    "tokenize_bm25_v2",
]
