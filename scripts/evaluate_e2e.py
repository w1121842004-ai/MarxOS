"""
End-to-end evaluation pipeline for MarxOS retrieval quality.

Runs all questions from eval_dataset_v2.json through the metadata matching
and constraint resolution pipeline. No LLM calls — measures retrieval accuracy.

Metrics:
  - work_match_rate: % of questions correctly matched to expected work_id
  - source_accuracy: % where expected source is in constraint sources
  - page_accuracy: % where expected source has page_range constraint
  - overall: average of all metrics

Usage:
    venv/Scripts/python.exe scripts/evaluate_e2e.py              # full report
    venv/Scripts/python.exe scripts/evaluate_e2e.py --ci         # CI mode (exit code)
    venv/Scripts/python.exe scripts/evaluate_e2e.py --discipline philosophy  # filter
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from marxos.work_catalog import WorkCatalog


def load_dataset(path=None):
    path = path or (ROOT / "eval_dataset_v2.json")
    if not os.path.exists(path):
        # Fall back to v1 if v2 doesn't exist
        path = ROOT / "eval_dataset.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ctx():
    """Minimal ctx for constraint resolution."""
    from app import (
        TOPIC_CATALOG, WORK_TITLE_ALIASES,
        CONCEPT_CANONICAL_CLASSIC_IDS, CONCEPT_PREFERRED_MARKERS,
        CONCEPT_PREFERRED_SOURCES, OCR_CACHE_DIR,
        RERANK_DEBUG_ENV, CLASSIC_SAYING_QUOTE_SEEDS, CLASSIC_SAYING_QUERY_SEEDS,
        normalize_topic_title, normalize_for_match, clean_article_title,
        clean_text, find_toc_entries, extract_bibliographic_title,
        locator_entries_for_query, enrich_core_classic_entries,
        active_concept_terms, core_classic_by_id, metadata_citation_page,
        as_int, work_catalog_entries_for_query,
    )
    from rag.core_classics import classic_entries_for_query

    return {
        "TOPIC_CATALOG": TOPIC_CATALOG,
        "WORK_TITLE_ALIASES": WORK_TITLE_ALIASES,
        "CONCEPT_CANONICAL_CLASSIC_IDS": CONCEPT_CANONICAL_CLASSIC_IDS,
        "CONCEPT_PREFERRED_MARKERS": CONCEPT_PREFERRED_MARKERS,
        "CONCEPT_PREFERRED_SOURCES": CONCEPT_PREFERRED_SOURCES,
        "OCR_CACHE_DIR": OCR_CACHE_DIR,
        "RERANK_DEBUG_ENV": RERANK_DEBUG_ENV,
        "CLASSIC_SAYING_QUOTE_SEEDS": CLASSIC_SAYING_QUOTE_SEEDS,
        "CLASSIC_SAYING_QUERY_SEEDS": CLASSIC_SAYING_QUERY_SEEDS,
        "normalize_topic_title": normalize_topic_title,
        "normalize_for_match": normalize_for_match,
        "clean_article_title": clean_article_title,
        "clean_text": clean_text,
        "find_toc_entries": find_toc_entries,
        "extract_bibliographic_title": extract_bibliographic_title,
        "locator_entries_for_query": locator_entries_for_query,
        "classic_entries_for_query": classic_entries_for_query,
        "enrich_core_classic_entries": enrich_core_classic_entries,
        "active_concept_terms": active_concept_terms,
        "core_classic_by_id": core_classic_by_id,
        "metadata_citation_page": metadata_citation_page,
        "as_int": as_int,
        "work_catalog_entries_for_query": work_catalog_entries_for_query,
        "book_locator_constraints": lambda q: {},
        "re": __import__("re"),
    }


def evaluate(questions, ctx, catalog, discipline_filter=None):
    """Run evaluation on all questions."""
    results = {
        "total": 0,
        "work_matched": 0,
        "work_correct": 0,
        "source_correct": 0,
        "page_correct": 0,
        "by_discipline": defaultdict(lambda: {"total": 0, "work_correct": 0,
                                               "source_correct": 0, "page_correct": 0}),
        "by_type": defaultdict(lambda: {"total": 0, "work_correct": 0}),
        "by_difficulty": defaultdict(lambda: {"total": 0, "work_correct": 0}),
        "failures": [],
    }

    from retrieval.constraints import constraints_from_query

    for q in questions:
        # Filter
        if discipline_filter and q.get("discipline") != discipline_filter:
            continue

        query = q["question"]
        expected_wid = q.get("expected_work_id", "")
        expected_source = q.get("expected_source", "")
        disc = q.get("discipline", "unknown")
        qtype = q.get("question_type", "unknown")
        diff = q.get("difficulty", "medium")

        # If no expected_source but we have work_id, infer from catalog
        if not expected_source and expected_wid:
            work = catalog.lookup_by_id(expected_wid)
            if work:
                entries = catalog.get_entries(work)
                if entries:
                    expected_source = entries[0]["source"]

        results["total"] += 1
        results["by_discipline"][disc]["total"] += 1
        results["by_type"][qtype]["total"] += 1
        results["by_difficulty"][diff]["total"] += 1

        # Get constraints
        constraints = constraints_from_query(query, ctx)

        # 1. Work match check
        work = catalog.match_query(query)
        if work:
            results["work_matched"] += 1
            if work["work_id"] == expected_wid:
                results["work_correct"] += 1
                results["by_discipline"][disc]["work_correct"] += 1
                results["by_type"][qtype]["work_correct"] += 1
                results["by_difficulty"][diff]["work_correct"] += 1

        # 2. Source accuracy
        constraint_sources = constraints.get("sources", set())
        if expected_source and expected_source in constraint_sources:
            results["source_correct"] += 1
            results["by_discipline"][disc]["source_correct"] += 1

        # 3. Page accuracy
        page_ranges = constraints.get("page_ranges", {})
        if expected_source and expected_source in page_ranges:
            results["page_correct"] += 1
            results["by_discipline"][disc]["page_correct"] += 1

        # 4. Citation accuracy — for quote_lookup, verify page actually has the quote
        if qtype == "quote_lookup" and constraints.get("entries"):
            results.setdefault("citation_total", 0)
            results.setdefault("citation_correct", 0)
            results["citation_total"] += 1
            try:
                from rag.exact_quote_lookup import (exact_quote_lookup, extract_query_quote,
                                                        normalize_quote, fuzzy_quote_match)
                quote = extract_query_quote(query)
                norm_q = normalize_quote(quote)
                exact_docs = exact_quote_lookup(query, ctx.get("OCR_CACHE_DIR", "data/ocr_cache"),
                                                limit=1, constraints=constraints)
                if exact_docs:
                    page_text = exact_docs[0].page_content or ""
                    norm_text = normalize_quote(page_text)
                    if norm_q and len(norm_q) >= 5:
                        # Exact check first, then fuzzy (tolerates OCR noise in constrained pages)
                        if norm_q in norm_text:
                            results["citation_correct"] += 1
                        elif exact_docs[0].metadata.get("lookup_scope") == "work_catalog_fuzzy":
                            matched, _ = fuzzy_quote_match(norm_q, norm_text)
                            if matched:
                                results["citation_correct"] += 1
            except Exception:
                pass  # citation check is best-effort

        # Track failures
        if work is None or work["work_id"] != expected_wid:
            results["failures"].append({
                "id": q["id"],
                "question": query[:80],
                "expected_work_id": expected_wid,
                "expected_source": expected_source,
                "got_work_id": work["work_id"] if work else None,
                "constraint_sources": list(constraint_sources) if constraint_sources else [],
                "discipline": disc,
                "difficulty": diff,
            })

    return results


def print_report(results):
    total = results["total"]
    if total == 0:
        print("No questions to evaluate.")
        return

    wm = results["work_matched"]
    wc = results["work_correct"]
    sc = results["source_correct"]
    pc = results["page_correct"]

    wmr = 100 * wm / total
    wcr = 100 * wc / total
    scr = 100 * sc / total
    pcr = 100 * pc / total

    print(f"\n{'='*70}")
    print(f"  MarxOS E2E Evaluation Report")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {total}")
    print()
    print(f"  {'Metric':<35} {'Score':>8} {'Rate':>8}")
    print(f"  {'-'*51}")
    print(f"  {'Work matched (any)':<35} {wm:>8} {wmr:>7.1f}%")
    print(f"  {'Work correct (precision)':<35} {wc:>8} {wcr:>7.1f}%")
    print(f"  {'Source in constraints':<35} {sc:>8} {scr:>7.1f}%")
    print(f"  {'Page range constraint':<35} {pc:>8} {pcr:>7.1f}%")

    overall = (wcr + scr + pcr) / 3
    print(f"  {'-'*51}")
    print(f"  {'OVERALL SCORE':<35} {'':>8} {overall:>7.1f}%")

    # Citation accuracy (quote_lookup only)
    ct = results.get("citation_total", 0)
    cc = results.get("citation_correct", 0)
    if ct > 0:
        cr = 100 * cc / ct
        print(f"  {'Citation accuracy (quote pg hit)':<35} {cc:>8} {cr:>7.1f}%")
    print()

    # By discipline
    print(f"  ── By Discipline ──")
    all_discs = ["philosophy", "political_economy", "scientific_socialism",
                 "party_labor", "peasant_land", "national_colonial",
                 "state_revolution_military", "history_religion_culture"]
    disc_labels = {
        "philosophy": "哲学", "political_economy": "政治经济学",
        "scientific_socialism": "科学社会主义", "party_labor": "党建与工人运动",
        "peasant_land": "农民与土地", "national_colonial": "民族与殖民",
        "state_revolution_military": "国家革命军事", "history_religion_culture": "历史宗教文化",
    }
    for disc in all_discs:
        d = results["by_discipline"][disc]
        if d["total"] > 0:
            wr = 100 * d["work_correct"] / d["total"]
            sr = 100 * d["source_correct"] / d["total"]
            label = disc_labels.get(disc, disc)
            print(f"  {label:<12}: {d['work_correct']:>3}/{d['total']:<3} work={wr:.0f}%  source={sr:.0f}%")

    print()
    print(f"  ── By Difficulty ──")
    for diff in ["easy", "medium", "hard"]:
        d = results["by_difficulty"][diff]
        if d["total"] > 0:
            wr = 100 * d["work_correct"] / d["total"]
            print(f"  {diff:<10}: {d['work_correct']:>3}/{d['total']:<3} = {wr:.0f}%")

    print()
    print(f"  ── By Type ──")
    for qtype in ["quote_lookup", "concept_explain", "analysis", "bibliographic"]:
        d = results["by_type"][qtype]
        if d["total"] > 0:
            wr = 100 * d["work_correct"] / d["total"]
            print(f"  {qtype:<20}: {d['work_correct']:>3}/{d['total']:<3} = {wr:.0f}%")

    # Failures
    failures = results["failures"]
    if failures:
        print(f"\n  ── Failures ({len(failures)}) ──")
        for f in failures[:10]:
            print(f"  [{f['id']}] {f['question'][:60]}")
            print(f"    expected: {f['expected_work_id']} @ {f['expected_source']}")
            print(f"    got:      {f['got_work_id']}")
            print(f"    sources:  {f['constraint_sources']}")
            print()

    print(f"{'='*70}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="MarxOS E2E Evaluation")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 if score below threshold")
    parser.add_argument("--threshold", type=float, default=85.0, help="Minimum work_correct % for CI (default: 85)")
    parser.add_argument("--citation-threshold", type=float, default=70.0, help="Minimum citation accuracy % for CI (default: 70)")
    parser.add_argument("--discipline", type=str, default=None, help="Filter by discipline")
    parser.add_argument("--dataset", type=str, default=None, help="Path to eval dataset")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    catalog = WorkCatalog()
    ctx = build_ctx()
    questions = load_dataset(args.dataset)

    start = time.time()
    results = evaluate(questions, ctx, catalog, args.discipline)
    elapsed = time.time() - start

    if args.json:
        print(json.dumps({
            "total": results["total"],
            "work_correct": results["work_correct"],
            "source_correct": results["source_correct"],
            "page_correct": results["page_correct"],
            "overall": round((100*results["work_correct"]/max(results["total"],1) +
                              100*results["source_correct"]/max(results["total"],1) +
                              100*results["page_correct"]/max(results["total"],1)) / 3, 1),
            "elapsed_s": round(elapsed, 1),
        }, ensure_ascii=False))
    else:
        overall = print_report(results)
        print(f"\n  Completed in {elapsed:.1f}s")

    if args.ci:
        wcr = 100 * results["work_correct"] / max(results["total"], 1)
        ct = results.get("citation_total", 0)
        cc = results.get("citation_correct", 0)
        cr = 100 * cc / ct if ct > 0 else 100

        failed = False
        if wcr < args.threshold:
            print(f"\nCI FAIL: work_correct={wcr:.1f}% < threshold={args.threshold}%")
            failed = True
        if ct > 0 and cr < args.citation_threshold:
            print(f"CI FAIL: citation_accuracy={cr:.1f}% < citation_threshold={args.citation_threshold}%")
            failed = True

        if failed:
            sys.exit(1)
        print(f"\nCI PASS: work_correct={wcr:.1f}% >= {args.threshold}%, citation_accuracy={cr:.1f}% >= {args.citation_threshold}%")


if __name__ == "__main__":
    main()
