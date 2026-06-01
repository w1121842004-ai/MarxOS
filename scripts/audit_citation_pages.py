"""
Audit work_catalog page ranges against actual OCR content.

For each work entry, loads OCR text for sampled pages in the claimed range
and checks if the work's title or key concepts appear. Reports entries where
the page mapping appears inaccurate.

Usage:
    venv/Scripts/python.exe scripts/audit_citation_pages.py
    venv/Scripts/python.exe scripts/audit_citation_pages.py --fix  # auto-correct
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_page_map():
    path = ROOT / "data" / "page_map.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        pm = json.load(f)
    mapping = {}
    for src, data in pm.get("sources", {}).items():
        mapping[src] = {}
        for pp, info in data.get("pages", {}).items():
            printed = info.get("printed_page")
            pdf = info.get("pdf_page")
            if printed is not None and pdf is not None:
                mapping[src][printed] = pdf
    return mapping


def load_ocr_page(source_stem, pdf_page, ocr_cache_dir):
    path = Path(ocr_cache_dir) / source_stem / f"page_{pdf_page}.json"
    if not path.exists():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cleaned_text") or data.get("raw_text") or ""
    except (OSError, json.JSONDecodeError):
        return ""


def normalize(text):
    import re
    return re.sub(r"[^0-9A-Za-z一-鿿]", "", str(text or ""))


def check_page_content(title, concepts, ocr_text):
    """Check if work title or concepts appear in OCR text."""
    norm_text = normalize(ocr_text)
    if not norm_text:
        return None, 0.0

    # Check title (key parts)
    title_parts = normalize(title)
    title_hit = title_parts[:6] in norm_text if len(title_parts) >= 6 else title_parts in norm_text

    # Check concepts
    concept_hits = 0
    for c in (concepts or []):
        nc = normalize(c)
        if len(nc) >= 3 and nc in norm_text:
            concept_hits += 1

    score = (1 if title_hit else 0) + min(concept_hits, 3)
    return title_hit, score


def audit(ocr_cache_dir="data/ocr_cache"):
    with open(ROOT / "rag" / "work_catalog.json", encoding="utf-8") as f:
        wc = json.load(f)

    page_map = load_page_map()

    results = {"total_entries": 0, "verified": 0, "suspicious": 0, "no_ocr": 0, "details": []}

    for work in wc["works"]:
        for ek, ev in work.get("editions", {}).items():
            source = ev["source"]
            start = ev.get("start_page")
            end = ev.get("end_page")
            if not source or start is None or end is None:
                continue

            results["total_entries"] += 1
            source_stem = source.replace(".pdf", "")
            title = work["title"]
            concepts = (work.get("primary_concepts", []) or [])[:3]

            # Sample pages: first, middle, last
            page_range = list(range(start, end + 1))
            sample_count = min(3, len(page_range))
            if sample_count == 0:
                continue

            if len(page_range) <= 3:
                samples = page_range
            else:
                step = max(1, len(page_range) // sample_count)
                samples = [page_range[0], page_range[len(page_range)//2], page_range[-1]]

            scores = []
            for printed_page in samples:
                pdf_page = page_map.get(source, {}).get(printed_page, printed_page)
                ocr_text = load_ocr_page(source_stem, pdf_page, ocr_cache_dir)
                if not ocr_text:
                    continue
                title_hit, score = check_page_content(title, concepts, ocr_text)
                scores.append((printed_page, pdf_page, title_hit, score))

            if not scores:
                results["no_ocr"] += 1
                continue

            avg_score = sum(s[3] for s in scores) / len(scores)
            if avg_score >= 1.0:  # at least title or 1 concept found
                results["verified"] += 1
            else:
                results["suspicious"] += 1
                results["details"].append({
                    "work_id": work["work_id"],
                    "title": title,
                    "edition": ek,
                    "source": source,
                    "page_range": f"{start}-{end}",
                    "avg_score": round(avg_score, 1),
                    "samples": [{"printed": s[0], "pdf": s[1],
                                 "title_hit": s[2], "score": s[3]} for s in scores],
                })

    return results


def print_report(results):
    total = results["total_entries"]
    ver = results["verified"]
    sus = results["suspicious"]
    no_ocr = results["no_ocr"]

    print(f"\n{'='*60}")
    print(f"  Page Mapping Audit")
    print(f"{'='*60}")
    print(f"  Total entries:  {total}")
    print(f"  Verified:       {ver} ({100*ver/max(total,1):.0f}%)")
    print(f"  Suspicious:     {sus} ({100*sus/max(total,1):.0f}%)")
    print(f"  No OCR data:    {no_ocr}")
    print()

    if results["details"]:
        print(f"  ── Suspicious Entries ({len(results['details'])}) ──")
        for d in results["details"]:
            print(f"  ⚠ {d['title'][:40]}")
            print(f"    {d['edition']}: {d['source']} p{d['page_range']} (score={d['avg_score']})")
            for s in d["samples"]:
                hit = "✓" if s["title_hit"] else "✗"
                print(f"    {hit} printed={s['printed']} pdf={s['pdf']} score={s['score']}")
            print()
    else:
        print("  All entries verified!")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-cache-dir", default="data/ocr_cache")
    args = parser.parse_args()

    results = audit(args.ocr_cache_dir)
    print_report(results)
