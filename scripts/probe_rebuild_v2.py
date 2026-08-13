#!/usr/bin/env python3
"""Select a deterministic corpus-v2 probe and compare chunk layouts offline.

The tool deliberately has no embedding, vector-store, or network dependencies.  It
only reads paragraph JSONL and emits a versioned report that can later be enriched
with retrieval queries and relevance judgements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


REPORT_VERSION = "rebuild-probe/v1"
CHUNK_REPORT_VERSION = "chunk-structure/v1"
RETRIEVAL_EVAL_VERSION = "retrieval-eval-candidates/v1"
DEFAULT_CHUNKS = ((180, 40), (256, 48), (320, 64))


def record_identity(record: dict) -> str:
    identity = record.get("record_id") or record.get("paragraph_id")
    if not identity:
        raise ValueError("eligible paragraph is missing stable identity (record_id/paragraph_id)")
    return str(identity)


def length_bucket(text: str) -> str:
    size = len(str(text or ""))
    if size < 120:
        return "short"
    if size <= 600:
        return "medium"
    return "long"


def is_cross_page(record: dict) -> bool:
    if record.get("cross_page") is True:
        return True
    page_span = record.get("page_span") or record.get("source_page_ids") or []
    if isinstance(page_span, (list, tuple)) and len(page_span) > 1:
        return True
    start = record.get("pdf_page_start")
    end = record.get("pdf_page_end")
    return start is not None and end is not None and start != end


def _eligible(record: dict) -> bool:
    return (
        record.get("retrievable") is not False
        and str(record.get("page_type") or "body") == "body"
        and bool(str(record.get("paragraph_text") or ""))
    )


def _stable_rank(seed: int, record: dict) -> str:
    value = f"{seed}\0{record_identity(record)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_probe_records(records: Iterable[dict], sample_size: int, seed: int = 20260812) -> list[dict]:
    """Select body paragraphs while spreading coverage over key corpus dimensions."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    eligible = [dict(record) for record in records if _eligible(record)]
    for record in eligible:
        record_identity(record)
    eligible.sort(key=lambda row: (_stable_rank(seed, row), record_identity(row)))
    target = min(sample_size, len(eligible))

    selected: list[dict] = []
    remaining = eligible[:]
    counts: dict[str, Counter] = {
        "source": Counter(),
        "length": Counter(),
        "cross_page": Counter(),
    }
    while remaining and len(selected) < target:
        def priority(row: dict) -> tuple:
            dimensions = (
                ("source", str(row.get("source") or "unknown")),
                ("length", length_bucket(row.get("paragraph_text", ""))),
                ("cross_page", str(is_cross_page(row)).lower()),
            )
            # Prefer candidates that add uncovered values, then the least represented
            # combination. Stable hash is the final deterministic tie-breaker.
            uncovered = sum(counts[name][value] == 0 for name, value in dimensions)
            represented = sum(counts[name][value] for name, value in dimensions)
            return (-uncovered, represented, _stable_rank(seed, row))

        chosen = min(remaining, key=priority)
        remaining.remove(chosen)
        selected.append(chosen)
        counts["source"][str(chosen.get("source") or "unknown")] += 1
        counts["length"][length_bucket(chosen.get("paragraph_text", ""))] += 1
        counts["cross_page"][str(is_cross_page(chosen)).lower()] += 1
    return selected


def _chunk_spans(text_length: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk overlap must be >= 0 and smaller than chunk size")
    if text_length == 0:
        return []
    step = chunk_size - chunk_overlap
    spans = []
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        spans.append((start, end))
        if end == text_length:
            break
        start += step
    return spans


def evaluate_chunk_config(records: Iterable[dict], chunk_size: int, chunk_overlap: int) -> dict:
    rows = list(records)
    chunks = []
    total_source_chars = 0
    indexed_chars = 0
    covered_chars = 0
    orphan_offsets = 0
    for record in rows:
        identity = record_identity(record)
        text = str(record.get("paragraph_text") or "")
        total_source_chars += len(text)
        spans = _chunk_spans(len(text), chunk_size, chunk_overlap)
        covered = set()
        for index, (start, end) in enumerate(spans):
            chunk_text = text[start:end]
            invalid = start < 0 or end > len(text) or start >= end or chunk_text != text[start:end]
            orphan_offsets += int(invalid)
            covered.update(range(start, end))
            indexed_chars += len(chunk_text)
            chunks.append({
                "chunk_id": f"{identity}#c{index:04d}",
                "parent_record_id": identity,
                "source": record.get("source"),
                "article": record.get("article"),
                "page_span": record.get("page_span") or [],
                "cross_page": is_cross_page(record),
                "length_bucket": length_bucket(text),
                "chunk_index": index,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
                "text_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            })
        covered_chars += len(covered)

    duplicated_chars = max(0, indexed_chars - covered_chars)
    config_id = f"chars-{chunk_size}-overlap-{chunk_overlap}"
    return {
        "schema_version": CHUNK_REPORT_VERSION,
        "config_id": config_id,
        "parameters": {"unit": "unicode_codepoint", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "summary": {
            "record_count": len(rows),
            "chunk_count": len(chunks),
            "source_character_count": total_source_chars,
            "covered_character_count": covered_chars,
            "indexed_character_count": indexed_chars,
            "duplicated_character_count": duplicated_chars,
            "coverage_ratio": covered_chars / total_source_chars if total_source_chars else 1.0,
            "storage_multiplier": indexed_chars / total_source_chars if total_source_chars else 0.0,
            "orphan_offset_count": orphan_offsets,
        },
        "chunks": chunks,
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(value)
    return records


def _parse_chunk(value: str) -> tuple[int, int]:
    try:
        size_text, overlap_text = value.split(":", 1)
        size, overlap = int(size_text), int(overlap_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("chunk must use SIZE:OVERLAP, for example 256:48") from error
    try:
        _chunk_spans(1, size, overlap)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return size, overlap


def build_report(input_path: Path, sample_size: int, seed: int, configs: list[tuple[int, int]]) -> dict:
    raw_bytes = input_path.read_bytes()
    records = load_jsonl(input_path)
    selected = select_probe_records(records, sample_size=sample_size, seed=seed)
    chunk_reports = [evaluate_chunk_config(selected, size, overlap) for size, overlap in configs]
    selected_ids = [record_identity(record) for record in selected]
    return {
        "schema_version": REPORT_VERSION,
        "input": {
            "path": str(input_path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "record_count": len(records),
        },
        "selection": {
            "strategy": "deterministic-multidimension-greedy/v1",
            "seed": seed,
            "requested_count": sample_size,
            "selected_count": len(selected),
            "record_ids": selected_ids,
            "sources": dict(sorted(Counter(str(row.get("source") or "unknown") for row in selected).items())),
            "length_buckets": dict(sorted(Counter(length_bucket(row.get("paragraph_text", "")) for row in selected).items())),
            "cross_page": dict(sorted(Counter(str(is_cross_page(row)).lower() for row in selected).items())),
        },
        "chunk_configurations": chunk_reports,
        "retrieval_eval": {
            "schema_version": RETRIEVAL_EVAL_VERSION,
            "probe_record_ids": selected_ids,
            "candidate_sets": [
                {"config_id": report["config_id"], "chunk_ids": [row["chunk_id"] for row in report["chunks"]]}
                for report in chunk_reports
            ],
            "queries": [],
            "judgements": [],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="corpus-v2 paragraph JSONL")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--chunk", action="append", type=_parse_chunk, dest="chunks")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.input, args.sample_size, args.seed, args.chunks or list(DEFAULT_CHUNKS))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
