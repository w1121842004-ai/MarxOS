#!/usr/bin/env python3
"""Compare v2 semantic-child character windows with offline BM25 retrieval.

The evaluator builds postings only for terms used by the evaluation questions.
This produces the same scores as materializing full BM25 sparse vectors while
keeping memory proportional to the fixed evaluation set instead of vocabulary.
It does not load Milvus, embedding models, private caches, or network clients.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos.indexing.bm25_sparse_v2 import BM25Config, TOKENIZER_VERSION, tokenize_bm25_v2


DEFAULT_RECORDS = ROOT_DIR / "data/artifacts/corpus_v2/paragraph_records_enriched_v2.jsonl"
DEFAULT_EVALUATION = ROOT_DIR / "eval_dataset_v2.json"
DEFAULT_OUTPUT = ROOT_DIR / "data/artifacts/corpus_v2/bm25_chunk_eval_v2.json"
DEFAULT_CONFIGS = ((180, 40), (256, 48), (320, 64))
REPORT_VERSION = "bm25-chunk-evaluation/v2"


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    @property
    def config_id(self) -> str:
        return f"chars-{self.chunk_size}-overlap-{self.chunk_overlap}"


@dataclass(frozen=True)
class RetrievalChunk:
    text: str
    paragraph_id: str
    work_id: str
    source: str
    page: int | None
    char_start: int
    char_end: int


def _chunk_spans(text_length: int, config: ChunkConfig) -> Iterator[tuple[int, int]]:
    step = config.chunk_size - config.chunk_overlap
    start = 0
    while start < text_length:
        end = min(start + config.chunk_size, text_length)
        yield start, end
        if end == text_length:
            return
        start += step


def iter_chunks(records: Iterable[dict], config: ChunkConfig) -> Iterator[RetrievalChunk]:
    """Yield traceable character chunks without mutating source records."""

    for record in records:
        text = str(record.get("paragraph_text") or "").strip()
        if not text or record.get("retrievable") is False:
            continue
        for start, end in _chunk_spans(len(text), config):
            yield RetrievalChunk(
                text=text[start:end],
                paragraph_id=str(record.get("paragraph_id") or ""),
                work_id=str(record.get("work_id") or ""),
                source=str(record.get("source") or ""),
                page=_optional_int(record.get("pdf_page_start")),
                char_start=start,
                char_end=end,
            )


def load_retrievable_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if record.get("retrievable") is False or not str(record.get("paragraph_text") or "").strip():
                continue
            records.append(record)
    if not records:
        raise ValueError(f"no retrievable paragraph records in {path}")
    return records


def load_evaluation_cases(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("evaluation dataset must be a non-empty JSON array")
    cases: list[dict] = []
    for index, raw_case in enumerate(payload):
        if not isinstance(raw_case, dict):
            raise ValueError(f"evaluation case {index} must be an object")
        case = dict(raw_case)
        if not str(case.get("question") or "").strip():
            raise ValueError(f"evaluation case {index} is missing question")
        if not str(case.get("expected_work_id") or "").strip():
            raise ValueError(f"evaluation case {index} is missing expected_work_id")
        cases.append(case)
    return cases


def _build_query_term_index(
    chunks: Sequence[RetrievalChunk],
    cases: Sequence[dict],
    bm25_config: BM25Config,
) -> tuple[list[int], dict[str, int], dict[str, list[tuple[int, int]]]]:
    query_terms = {
        term
        for case in cases
        for term in tokenize_bm25_v2(str(case["question"]), bm25_config)
    }
    lengths: list[int] = []
    document_frequencies: Counter[str] = Counter()
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for document_id, chunk in enumerate(chunks):
        tokens = tokenize_bm25_v2(chunk.text, bm25_config)
        lengths.append(len(tokens))
        counts = Counter(term for term in tokens if term in query_terms)
        document_frequencies.update(counts.keys())
        for term, frequency in counts.items():
            postings[term].append((document_id, frequency))
    return lengths, dict(document_frequencies), dict(postings)


def _rank_case(
    question: str,
    *,
    chunks: Sequence[RetrievalChunk],
    lengths: Sequence[int],
    document_frequencies: dict[str, int],
    postings: dict[str, list[tuple[int, int]]],
    bm25_config: BM25Config,
    limit: int = 8,
) -> list[tuple[int, float]]:
    document_count = len(chunks)
    average_length = sum(lengths) / document_count if document_count else 0.0
    if not average_length:
        return []
    scores: dict[int, float] = defaultdict(float)
    query_counts = Counter(tokenize_bm25_v2(question, bm25_config))
    for term, query_frequency in query_counts.items():
        document_frequency = document_frequencies.get(term)
        if not document_frequency:
            continue
        idf = math.log1p((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        query_weight = idf * (1.0 + math.log(float(query_frequency)))
        for document_id, term_frequency in postings[term]:
            length_ratio = lengths[document_id] / average_length
            normalization = bm25_config.k1 * (
                1.0 - bm25_config.b + bm25_config.b * length_ratio
            )
            document_weight = idf * (
                term_frequency * (bm25_config.k1 + 1.0)
                / (term_frequency + normalization)
            )
            scores[document_id] += query_weight * document_weight
    return heapq.nlargest(limit, scores.items(), key=lambda item: (item[1], -item[0]))


def evaluate_chunk_config(
    records: Sequence[dict],
    cases: Sequence[dict],
    config: ChunkConfig,
    *,
    bm25_config: BM25Config | None = None,
) -> dict:
    active_bm25 = bm25_config or BM25Config()
    started = time.monotonic()
    chunks = list(iter_chunks(records, config))
    lengths, document_frequencies, postings = _build_query_term_index(chunks, cases, active_bm25)
    available_work_ids = {chunk.work_id for chunk in chunks if chunk.work_id}
    results: list[dict] = []
    reciprocal_rank_sum = 0.0
    recall_counts = {1: 0, 5: 0, 8: 0}
    hard_negative_hits = 0
    hard_negative_evaluable = 0
    evaluable_count = 0

    for case in cases:
        expected = str(case["expected_work_id"])
        evaluable = expected in available_work_ids
        ranked = _rank_case(
            str(case["question"]),
            chunks=chunks,
            lengths=lengths,
            document_frequencies=document_frequencies,
            postings=postings,
            bm25_config=active_bm25,
        )
        ranked_work_ids = [chunks[document_id].work_id for document_id, _ in ranked]
        expected_rank = next(
            (rank for rank, work_id in enumerate(ranked_work_ids, start=1) if work_id == expected),
            None,
        )
        hard_negatives = {str(value) for value in case.get("hard_negative", [])}
        hard_negative_hit = bool(hard_negatives.intersection(ranked_work_ids[:8]))
        if evaluable:
            evaluable_count += 1
            for cutoff in recall_counts:
                recall_counts[cutoff] += int(expected in ranked_work_ids[:cutoff])
            if expected_rank is not None and expected_rank <= 8:
                reciprocal_rank_sum += 1.0 / expected_rank
            if hard_negatives:
                hard_negative_evaluable += 1
                hard_negative_hits += int(hard_negative_hit)
        results.append({
            "id": case.get("id"),
            "expected_work_id": expected,
            "evaluable": evaluable,
            "expected_rank_at_8": expected_rank,
            "hard_negative_hit_at_8": hard_negative_hit,
            "top_work_ids": ranked_work_ids,
            "top_scores": [round(score, 8) for _, score in ranked],
        })

    metric_denominator = evaluable_count or 1
    return {
        "config_id": config.config_id,
        "parameters": {**asdict(config), "unit": "unicode_codepoint"},
        "bm25": {
            "k1": active_bm25.k1,
            "b": active_bm25.b,
            "tokenizer_version": active_bm25.tokenizer_version,
            "chinese_ngram_sizes": list(active_bm25.chinese_ngram_sizes),
            "implementation": "query-term-only exact sparse dot product",
        },
        "corpus": {
            "retrievable_paragraph_count": len(records),
            "chunk_count": len(chunks),
            "available_work_id_count": len(available_work_ids),
            "query_term_count": len(postings),
        },
        "evaluation": {
            "case_count": len(cases),
            "evaluable_case_count": evaluable_count,
            "hard_negative_case_count": hard_negative_evaluable,
        },
        "metrics": {
            "evaluable_coverage": evaluable_count / len(cases) if cases else 0.0,
            "recall_at_1": recall_counts[1] / metric_denominator,
            "recall_at_5": recall_counts[5] / metric_denominator,
            "recall_at_8": recall_counts[8] / metric_denominator,
            "mrr_at_8": reciprocal_rank_sum / metric_denominator,
            "hard_negative_hit_count_at_8": hard_negative_hits,
            "hard_negative_hit_rate_at_8": (
                hard_negative_hits / hard_negative_evaluable if hard_negative_evaluable else 0.0
            ),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cases": results,
    }


def _select_recommended(reports: Sequence[dict]) -> str:
    best = max(
        reports,
        key=lambda report: (
            report["metrics"]["recall_at_8"],
            report["metrics"]["mrr_at_8"],
            report["metrics"]["recall_at_1"],
            -report["parameters"]["chunk_size"],
        ),
    )
    return str(best["config_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_config(value: str) -> ChunkConfig:
    try:
        size, overlap = value.split(":", maxsplit=1)
        return ChunkConfig(int(size), int(overlap))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("config must use SIZE:OVERLAP") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        dest="configs",
        help="repeatable SIZE:OVERLAP; defaults to 180:40, 256:48, 320:64",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = args.configs or [ChunkConfig(*values) for values in DEFAULT_CONFIGS]
    records = load_retrievable_records(args.records)
    cases = load_evaluation_cases(args.evaluation)
    reports = []
    for config in configs:
        print(f"evaluating {config.config_id} ...", flush=True)
        report = evaluate_chunk_config(records, cases, config)
        reports.append(report)
        print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True), flush=True)
    payload = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline": True,
        "inputs": {
            "records": str(args.records),
            "records_sha256": _sha256(args.records),
            "evaluation": str(args.evaluation),
            "evaluation_sha256": _sha256(args.evaluation),
        },
        "tokenizer_version": TOKENIZER_VERSION,
        "recommended_config_id": _select_recommended(reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.output}", flush=True)
    print(f"recommended: {payload['recommended_config_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
