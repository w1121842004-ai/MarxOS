from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "config/rebuild_v2.json"
MIN_FREE_BYTES = 10 * 1024**3


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    artifact_dir: Path,
    artifacts: dict[str, Path],
    *,
    build_id: str,
    parent_build_id: str,
) -> dict[str, Any]:
    entries = {}
    for name, path in sorted(artifacts.items()):
        row_count = None
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
        entries[name] = {
            "path": str(path.relative_to(artifact_dir)),
            "bytes": path.stat().st_size,
            "row_count": row_count,
            "sha256": _sha256_file(path),
        }
    return {
        "schema_version": "marxos-artifact-manifest/v2",
        "build_id": build_id,
        "parent_build_id": parent_build_id,
        "status": "building",
        "artifacts": entries,
    }


def build_page_records(page_cache: Path, sources: list[str], output: Path) -> dict[str, Any]:
    """Freeze cleaned page cache into an immutable, line-oriented v2 snapshot."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    source_counts: Counter[str] = Counter()
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for source in sources:
            source_dir = page_cache / source.removesuffix(".pdf")
            page_paths = sorted(
                source_dir.glob("page_*.json"),
                key=lambda path: int(path.stem.removeprefix("page_")),
            )
            for page_path in page_paths:
                payload = json.loads(page_path.read_text(encoding="utf-8"))
                page_number = int(page_path.stem.removeprefix("page_"))
                raw_text = str(payload.get("raw_text") or payload.get("cleaned_text") or "")
                normalized_text = str(payload.get("cleaned_text") or "")
                record = {
                    "record_version": "page-record/v2",
                    "page_id": f"{source}#pdf{page_number}",
                    "source": source,
                    "pdf_page": page_number,
                    "raw_text": raw_text,
                    "raw_text_sha256": _sha256_text(raw_text),
                    "normalized_text": normalized_text,
                    "normalized_text_sha256": _sha256_text(normalized_text),
                    "text_source": payload.get("text_source") or "legacy_ocr",
                    "page_type": payload.get("page_type") or "unknown",
                    "page_type_source": "page_cache",
                    "page_number_candidates": payload.get("page_number_candidates") or [],
                    "cleaning_reasons": payload.get("reasons") or [],
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                records += 1
                source_counts[source] += 1
    return {"records": records, "sources": dict(sorted(source_counts.items())), "path": str(output)}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _issue(code: str, path: Path, message: str) -> dict[str, str]:
    return {"code": code, "path": str(path), "message": message}


def preflight_report(config: dict[str, Any], root: Path = ROOT, free_bytes: int | None = None) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if config.get("schema_version") != "marxos-rebuild/v2":
        issues.append(_issue("CONFIG_VERSION_INVALID", root, "expected marxos-rebuild/v2"))

    inputs = config.get("inputs") or {}
    outputs = config.get("outputs") or {}
    page_cache = _resolve(root, str(inputs.get("page_cache") or ""))
    for key in ("article_map", "work_catalog", "page_map"):
        path = _resolve(root, str(inputs.get(key) or ""))
        if not path.is_file():
            issues.append(_issue("INPUT_MISSING", path, f"missing required input: {key}"))

    for source in (config.get("scope") or {}).get("sources") or []:
        source_dir = page_cache / str(source).removesuffix(".pdf")
        if not source_dir.is_dir() or not any(source_dir.glob("page_*.json")):
            issues.append(_issue("SOURCE_CACHE_MISSING", source_dir, f"no cached pages for {source}"))

    overwrite = bool((config.get("contracts") or {}).get("overwrite_existing"))
    if not overwrite:
        for key in ("artifact_dir", "milvus_uri"):
            path = _resolve(root, str(outputs.get(key) or ""))
            if path.exists():
                issues.append(_issue("OUTPUT_EXISTS", path, f"refusing to overwrite {key}"))

    available = free_bytes if free_bytes is not None else shutil.disk_usage(root).free
    if available < MIN_FREE_BYTES:
        issues.append(_issue("DISK_SPACE_LOW", root, f"requires at least {MIN_FREE_BYTES} free bytes"))

    counts = Counter(issue["code"] for issue in issues)
    return {
        "schema_version": "marxos-rebuild-preflight/v1",
        "ready": not issues,
        "summary": {
            "errors": len(issues),
            "issues_by_code": dict(sorted(counts.items())),
            "free_bytes": available,
            "source_count": len((config.get("scope") or {}).get("sources") or []),
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate the non-destructive MarxOS corpus v2 rebuild.")
    parser.add_argument("stage", choices=["preflight", "pages", "paragraphs", "enrich", "manifest"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.stage == "enrich":
            from marxos.data.bibliography_v2 import BibliographyIndex

            input_path = _resolve(ROOT, config["outputs"]["paragraph_records"])
            output_path = _resolve(ROOT, config["outputs"]["enriched_paragraph_records"])
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite immutable enriched artifact: {output_path}")
            work_catalog = json.loads(_resolve(ROOT, config["inputs"]["work_catalog"]).read_text(encoding="utf-8"))
            article_map = json.loads(_resolve(ROOT, config["inputs"]["article_map"]).read_text(encoding="utf-8"))
            bibliography = BibliographyIndex(work_catalog, article_map)
            counts: Counter[str] = Counter()
            with input_path.open("r", encoding="utf-8") as source_handle, output_path.open("x", encoding="utf-8", newline="\n") as target:
                for line in source_handle:
                    if not line.strip():
                        continue
                    record = bibliography.enrich(json.loads(line))
                    target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["records"] += 1
                    counts["work_id"] += bool(record.get("work_id"))
                    counts["article_id"] += bool(record.get("article_id"))
                    counts["edition_id"] += bool(record.get("edition_id"))
            report = {
                "schema_version": "marxos-rebuild-stage/v1",
                "stage": "enrich",
                "summary": {**dict(counts), "path": str(output_path)},
            }
        elif args.stage == "manifest":
            artifact_dir = _resolve(ROOT, config["outputs"]["artifact_dir"])
            manifest_path = _resolve(ROOT, config["outputs"]["manifest"])
            if manifest_path.exists():
                raise FileExistsError(f"refusing to overwrite immutable manifest: {manifest_path}")
            artifacts = {
                name: _resolve(ROOT, config["outputs"][name])
                for name in ("page_records", "paragraph_records", "enriched_paragraph_records", "audit_report")
            }
            missing = [str(path) for path in artifacts.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"manifest inputs missing: {missing}")
            build_id = "corpus-v2-" + _sha256_file(artifacts["paragraph_records"])[:16]
            manifest = build_artifact_manifest(
                artifact_dir,
                artifacts,
                build_id=build_id,
                parent_build_id="milvus_bgem3_stable",
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report = {"schema_version": "marxos-rebuild-stage/v1", "stage": "manifest", "summary": manifest}
        elif args.stage == "paragraphs":
            from marxos.data.document_contract import audit_document_records
            from rag.paragraph_cache import read_paragraph_cache, write_paragraph_cache

            output = _resolve(ROOT, config["outputs"]["paragraph_records"])
            audit_path = _resolve(ROOT, config["outputs"]["audit_report"])
            if output.exists() or audit_path.exists():
                raise FileExistsError(f"refusing to overwrite immutable paragraph artifacts under {output.parent}")
            summary = write_paragraph_cache(
                output,
                sources=config["scope"]["sources"],
                ocr_cache_dir=_resolve(ROOT, config["inputs"]["page_cache"]),
            )
            audit = audit_document_records(read_paragraph_cache(output))
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report = {
                "schema_version": "marxos-rebuild-stage/v1",
                "stage": "paragraphs",
                "ready": audit["summary"]["passed"],
                "summary": {**summary, "audit": audit["summary"]},
            }
        elif args.stage == "pages":
            page_cache = _resolve(ROOT, config["inputs"]["page_cache"])
            output = _resolve(ROOT, config["outputs"]["page_records"])
            if output.exists():
                raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
            report = {
                "schema_version": "marxos-rebuild-stage/v1",
                "stage": "pages",
                "summary": build_page_records(page_cache, config["scope"]["sources"], output),
            }
        else:
            report = preflight_report(config)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"schema_version": "marxos-rebuild-preflight/v1", "ready": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.stage in {"pages", "paragraphs", "enrich", "manifest"}:
        return 0 if report.get("ready", True) else 1
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
