#!/usr/bin/env python3
"""Build immutable semantic-child RetrievalRecord v2 JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marxos.data.retrieval_records_v2 import build_semantic_child_records  # noqa: E402


def _records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="enriched ParagraphRecord v2 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new semantic-child JSONL path")
    parser.add_argument("--summary", type=Path, required=True, help="new build summary JSON path")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    return parser


def _assert_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path} already exists")


def build(args: argparse.Namespace) -> dict:
    if not args.input.is_file():
        raise FileNotFoundError(f"input does not exist: {args.input}")
    if args.output == args.summary:
        raise ValueError("output and summary paths must differ")
    _assert_new_file(args.output)
    _assert_new_file(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    input_count = 0
    excluded_count = 0
    output_count = 0
    output_hash = hashlib.sha256()

    def counted_records():
        nonlocal input_count, excluded_count
        for record in _records(args.input):
            input_count += 1
            text = str(record.get("paragraph_text") or record.get("text") or "")
            if record.get("retrievable") is False or not text:
                excluded_count += 1
            yield record

    try:
        with args.output.open("x", encoding="utf-8") as output:
            for record in build_semantic_child_records(
                counted_records(),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ):
                encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                output.write(encoded.decode("utf-8"))
                output_hash.update(encoded)
                output_count += 1
    except Exception:
        args.output.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": "retrieval-build-summary/v2",
        "input": str(args.input),
        "output": str(args.output),
        "paragraph_records_read": input_count,
        "paragraph_records_excluded": excluded_count,
        "semantic_child_records_written": output_count,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "output_sha256": output_hash.hexdigest(),
    }
    try:
        with args.summary.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    return summary


def main() -> int:
    try:
        summary = build(_parser().parse_args())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
