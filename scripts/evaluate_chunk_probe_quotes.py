#!/usr/bin/env python3
"""Evaluate quote preservation across offline chunk-probe configurations.

This is a structural evaluation: it measures whether a source/page candidate
preserves an expected quotation. It deliberately performs no retrieval and has
no model, network, Milvus, or embedding dependencies.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Iterable


REPORT_VERSION = "chunk-quote-eval/v1"


def normalize_quote_text(value: object) -> str:
    """Normalize typography while retaining letters, numbers, and Han text."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalize_source(value: object) -> str:
    return Path(str(value or "").strip()).name.casefold()


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _page_range(chunk: dict) -> tuple[int, int] | None:
    start = _as_int(chunk.get("citation_page_start") or chunk.get("citation_page"))
    end = _as_int(chunk.get("citation_page_end") or start)
    if start is not None:
        return min(start, end or start), max(start, end or start)
    # chunk-structure/v1 originally only persisted page_span. Retain a clearly
    # reported compatibility fallback so old probes remain evaluable.
    pages = [_as_int(value) for value in (chunk.get("page_span") or [])]
    pages = [page for page in pages if page is not None]
    return (min(pages), max(pages)) if pages else None


def _structural_match(chunk: dict, source: str, page: int) -> bool:
    if normalize_source(chunk.get("source")) != source:
        return False
    page_range = _page_range(chunk)
    return bool(page_range and page_range[0] <= page <= page_range[1])


def _quote_coverage(quote: str, candidate: str) -> float:
    if not quote:
        return 0.0
    if quote in candidate:
        return 1.0
    match = difflib.SequenceMatcher(None, quote, candidate, autojunk=False).find_longest_match()
    return match.size / len(quote)


def _expected_citations(dataset: Iterable[dict]) -> list[dict]:
    citations = []
    for case in dataset:
        case_id = str(case.get("id") or "")
        for index, citation in enumerate(case.get("expected_citations") or []):
            source = normalize_source(citation.get("source"))
            page = _as_int(citation.get("citation_page"))
            quote = normalize_quote_text(citation.get("quote"))
            if source and page is not None and quote:
                citations.append({
                    "case_id": case_id,
                    "citation_index": index,
                    "source": source,
                    "citation_page": page,
                    "quote": quote,
                })
    return citations


def _evaluate_config(config: dict, citations: list[dict]) -> dict:
    chunks = list(config.get("chunks") or [])
    results = []
    for citation in citations:
        candidates = [
            chunk for chunk in chunks
            if _structural_match(chunk, citation["source"], citation["citation_page"])
        ]
        scored = [
            (_quote_coverage(citation["quote"], normalize_quote_text(chunk.get("text"))), chunk)
            for chunk in candidates
        ]
        best_coverage, best_chunk = max(scored, key=lambda item: item[0], default=(0.0, None))
        results.append({
            "case_id": citation["case_id"],
            "citation_index": citation["citation_index"],
            "source": citation["source"],
            "citation_page": citation["citation_page"],
            "evaluable": bool(candidates),
            "candidate_count": len(candidates),
            "exact_quote_containment": bool(candidates) and best_coverage == 1.0,
            "best_coverage": round(best_coverage, 6),
            "best_chunk_id": best_chunk.get("chunk_id") if best_chunk else None,
        })

    evaluable = [row for row in results if row["evaluable"]]
    exact = [row for row in evaluable if row["exact_quote_containment"]]
    partial = [row for row in evaluable if 0 < row["best_coverage"] < 1]
    failures = [row for row in evaluable if not row["exact_quote_containment"]]
    unevaluable = [row for row in results if not row["evaluable"]]
    return {
        "config_id": str(config.get("config_id") or "unknown"),
        "parameters": dict(config.get("parameters") or {}),
        "summary": {
            "citation_count": len(citations),
            "evaluable_cases": len(evaluable),
            "unevaluable_cases": len(unevaluable),
            "exact_quote_containment_count": len(exact),
            "exact_quote_containment_rate": len(exact) / len(evaluable) if evaluable else None,
            "partial_coverage_count": len(partial),
            "mean_partial_coverage": (
                sum(row["best_coverage"] for row in partial) / len(partial) if partial else 0.0
            ),
        },
        "failures": failures,
        "unevaluable": unevaluable,
    }


def evaluate_quote_probe(probe: dict, dataset: list[dict]) -> dict:
    configs = probe.get("chunk_configurations")
    if not isinstance(configs, list) or not configs:
        raise ValueError("chunk probe must contain non-empty chunk_configurations")
    if not isinstance(dataset, list):
        raise ValueError("evaluation dataset must be a JSON array")
    citations = _expected_citations(dataset)
    return {
        "schema_version": REPORT_VERSION,
        "method": {
            "candidate_filter": "normalized source + inclusive citation page",
            "page_span_compatibility_fallback": True,
            "text_normalization": "NFKC + casefold + alphanumeric-only",
            "partial_coverage": "longest contiguous normalized quote match / quote length",
        },
        "dataset": {
            "case_count": len(dataset),
            "valid_expected_citation_count": len(citations),
        },
        "configurations": [_evaluate_config(config, citations) for config in configs],
    }


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-probe", required=True, type=Path)
    parser.add_argument("--eval-dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_quote_probe(_read_json(args.chunk_probe), _read_json(args.eval_dataset))
    except (OSError, ValueError, json.JSONDecodeError) as error:
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
