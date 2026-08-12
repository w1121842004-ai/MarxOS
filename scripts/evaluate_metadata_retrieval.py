"""
Evaluate retrieval quality improvements from work_catalog metadata constraints.

Measures:
  - Work match rate: how often a query is correctly mapped to its work
  - Source precision@1: top result comes from the correct source (PDF)
  - Page accuracy@1: top result is within the correct page range
  - Constraints coverage: what % of queries get metadata constraints

Usage:
    venv/Scripts/python.exe scripts/evaluate_metadata_retrieval.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from marxos.work_catalog import WorkCatalog, _normalize
from marxos.runtime import RuntimeState
from retrieval.constraints import constraints_from_query


# ── Test Queries ─────────────────────────────────────────────────

# Each entry: (query, expected_work_id, expected_source)
# These test queries span all three disciplines
TEST_QUERIES = [
    # ── 哲学 ──
    ("费尔巴哈的实践观", "theses-feuerbach", "mes01.pdf"),
    ("关于费尔巴哈的提纲第十一条", "theses-feuerbach", "mes01.pdf"),
    ("德意志意识形态中的历史唯物主义", "german-ideology", "mes01.pdf"),
    ("马克思论人的本质", "theses-feuerbach", "mes01.pdf"),
    ("《黑格尔法哲学批判》导言中宗教批判的观点", "critique-hegel-law-intro", "mes01.pdf"),
    ("1844年手稿中的异化劳动理论", "economic-philosophic-manuscripts-1844", "mea01.pdf"),
    ("恩格斯自然辩证法的主要内容", "dialectics-nature", "mea09.pdf"),
    ("费尔巴哈和德国古典哲学的终结", "ludwig-feuerbach", "mea04.pdf"),
    ("路德维希·费尔巴哈论", "ludwig-feuerbach", "mea04.pdf"),
    ("对黑格尔辩证法的批判", "economic-philosophic-manuscripts-1844", "mea01.pdf"),

    # ── 政治经济学 ──
    ("资本论中商品拜物教的理论", "capital-vol1", "mea05.pdf"),
    ("剩余价值是怎么产生的", "capital-vol1", "mea05.pdf"),
    ("工资价格和利润的关系", "value-price-profit", "mea03.pdf"),
    ("《政治经济学批判》序言的历史唯物主义", "preface-critique-political-economy", "mea02.pdf"),
    ("雇佣劳动与资本的主要内容", "wage-labour-capital", "mes01.pdf"),
    ("资本积累的一般规律", "capital-vol1", "mea05.pdf"),
    ("原始积累的秘密", "capital-vol1", "mea05.pdf"),
    ("国民经济学批判大纲", "outlines-critique-political-economy", "mea01.pdf"),
    ("资本主义生产以前的各种所有制形式", "grundrisse-selections", "mea08.pdf"),
    ("生产劳动和非生产劳动的区分", "manuscripts-1861-1863-selections", "mea08.pdf"),

    # ── 科学社会主义 ──
    ("共产党宣言的阶级斗争理论", "communist-manifesto", "mes01.pdf"),
    ("无产阶级专政这个概念", "class-struggles-france", "mea02.pdf"),
    ("路易波拿巴的雾月十八日国家理论", "eighteenth-brumaire", "mes01.pdf"),
    ("法兰西内战中的巴黎公社经验", "civil-war-france", "mea03.pdf"),
    ("哥达纲领批判中的按需分配", "critique-gotha-programme", "mea03.pdf"),
    ("社会主义从空想到科学", "socialism-utopian-scientific", "mea03.pdf"),
    ("家庭私有制和国家的起源", "origin-family-private-property-state", "mea04.pdf"),
    ("法德农民问题中的合作社思想", "peasant-question-france-germany", "mea04.pdf"),
    ("反杜林论的主要内容", "anti-duhring", "mea09.pdf"),
    ("恩格斯论权威", "on-authority", "mes03.pdf"),

    # ── 中国/殖民主义 ──
    ("马克思论鸦片贸易", "opium-trade", "mea02.pdf"),
    ("不列颠在印度的统治", "british-rule-india", "mes01.pdf"),
    ("中国革命和欧洲革命的关系", "revolution-china-europe", "mea02.pdf"),
    ("新的对华战争", "new-war-china", "mea02.pdf"),
]


def build_retrieval_ctx():
    """Minimal ctx for constraint resolution (matches app.py structure)."""
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
        "re": __import__("re"),
    }


def evaluate():
    catalog = WorkCatalog()
    ctx = build_retrieval_ctx()

    results = {
        "total_queries": len(TEST_QUERIES),
        "work_matched": 0,
        "work_matched_correct": 0,
        "constraints_applied": 0,
        "source_in_constraints": 0,
        "page_in_constraints": 0,
        "details": [],
    }

    for query, expected_work_id, expected_source in TEST_QUERIES:
        detail = {
            "query": query,
            "expected_work_id": expected_work_id,
            "expected_source": expected_source,
        }

        # Test 1: WorkCatalog match accuracy
        work = catalog.match_query(query)
        detail["work_matched"] = work is not None
        if work:
            detail["matched_work_id"] = work["work_id"]
            detail["work_correct"] = (work["work_id"] == expected_work_id)
            results["work_matched"] += 1
            if work["work_id"] == expected_work_id:
                results["work_matched_correct"] += 1
        else:
            detail["matched_work_id"] = None
            detail["work_correct"] = False

        # Test 2: Constraints from query (with work_catalog)
        constraints = constraints_from_query(query, ctx)
        detail["has_constraints"] = bool(constraints.get("sources"))
        if constraints.get("sources"):
            results["constraints_applied"] += 1
            detail["constraint_sources"] = list(constraints["sources"])
            detail["source_in_constraints"] = expected_source in constraints["sources"]
            if detail["source_in_constraints"]:
                results["source_in_constraints"] += 1

            # Check if expected page range exists
            page_ranges = constraints.get("page_ranges", {})
            detail["page_ranges"] = {k: v for k, v in page_ranges.items()}
            detail["page_in_constraints"] = expected_source in page_ranges
            if detail["page_in_constraints"]:
                results["page_in_constraints"] += 1

        results["details"].append(detail)

    return results


def print_results(results):
    total = results["total_queries"]
    print(f"\n{'='*70}")
    print(f"  WorkCatalog Metadata Retrieval Evaluation")
    print(f"{'='*70}")
    print(f"  Queries: {total}")
    print()

    # Work match
    wm = results["work_matched"]
    wmc = results["work_matched_correct"]
    print(f"  ── Work Match ──")
    print(f"  Query → Work matched:     {wm}/{total} ({100*wm/total:.0f}%)")
    print(f"  Match correct (precision): {wmc}/{wm} ({100*wmc/max(wm,1):.0f}%)")
    print()

    # Constraints
    ca = results["constraints_applied"]
    sic = results["source_in_constraints"]
    pic = results["page_in_constraints"]
    print(f"  ── Constraints ──")
    print(f"  Constraints applied:       {ca}/{total} ({100*ca/total:.0f}%)")
    print(f"  Correct source in range:   {sic}/{ca} ({100*sic/max(ca,1):.0f}%)")
    print(f"  Correct page constraint:   {pic}/{ca} ({100*pic/max(ca,1):.0f}%)")
    print()

    # Details for failures
    failures = [d for d in results["details"]
                if not d["work_correct"] or not d.get("source_in_constraints", False)]
    if failures:
        print(f"  ── Failures ({len(failures)}) ──")
        for d in failures:
            print(f"  ✗ {d['query']}")
            print(f"    expected: {d['expected_work_id']} @ {d['expected_source']}")
            print(f"    got:      {d.get('matched_work_id', 'NONE')}")
            print(f"    sources in constraints: {d.get('constraint_sources', [])}")
            print()

    # By discipline
    print(f"  ── By Discipline ──")
    catalog = WorkCatalog()
    for disc, label in [("philosophy", "哲学"), ("political_economy", "政治经济学"),
                         ("scientific_socialism", "科学社会主义")]:
        disc_queries = [(q, ew, es) for q, ew, es in TEST_QUERIES
                        if disc in catalog.lookup_by_id(ew).get("discipline", [])]
        disc_details = [d for d in results["details"]
                        if d["expected_work_id"] in {ew for _, ew, _ in disc_queries}]
        correct = sum(1 for d in disc_details if d["work_correct"])
        print(f"  {label}: {correct}/{len(disc_details)} correct ({100*correct/max(len(disc_details),1):.0f}%)")

    print()
    print(f"{'='*70}")
    print(f"  Overall Score: {results['work_matched_correct']}/{total}")
    print(f"  = {100*results['work_matched_correct']/total:.1f}% query-to-work accuracy")
    print(f"{'='*70}")


if __name__ == "__main__":
    start = time.time()
    results = evaluate()
    elapsed = time.time() - start
    print_results(results)
    print(f"\nEvaluation completed in {elapsed:.1f}s")
