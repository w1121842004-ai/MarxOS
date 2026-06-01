"""
End-to-end answer quality evaluation with real DeepSeek calls.

Runs sampled questions through the full run_query() pipeline, measures:
  - Answer generation success rate
  - Citation format audit pass rate
  - Citation content verification (hallucination detection)
  - Answer length and citation density
  - Per-discipline and per-type breakdown

Usage:
    venv/Scripts/python.exe scripts/evaluate_answer_quality.py           # 40 questions
    venv/Scripts/python.exe scripts/evaluate_answer_quality.py --all    # all 200
    venv/Scripts/python.exe scripts/evaluate_answer_quality.py --count 10  # 10 random
"""
import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MARXOS_DEV_MODE", "1")  # suppress some noise


def load_dataset():
    path = ROOT / "eval_dataset_v2.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_questions(dataset, count=40):
    """Sample questions evenly across 8 disciplines."""
    by_disc = defaultdict(list)
    for q in dataset:
        by_disc[q.get("discipline", "unknown")].append(q)

    sampled = []
    per_disc = max(1, count // len(by_disc))
    for disc, qs in by_disc.items():
        random.shuffle(qs)
        sampled.extend(qs[:per_disc])

    random.shuffle(sampled)
    return sampled[:count]


def evaluate_one(query, question_id, expected_wid):
    """Run one question through the full pipeline."""
    from app import run_query

    start = time.time()
    try:
        result = run_query(query)
    except Exception as e:
        return {"id": question_id, "query": query, "error": str(e), "elapsed_s": time.time() - start}

    elapsed = time.time() - start

    if isinstance(result, str):
        return {
            "id": question_id, "query": query, "expected_work_id": expected_wid,
            "answer": result, "answer_length": len(result),
            "route": "local", "elapsed_s": elapsed,
        }

    audit = result.get("citation_audit", {})
    verify = audit.get("content_verification", {})
    evidence = result.get("evidence", [])

    return {
        "id": question_id,
        "query": query,
        "expected_work_id": expected_wid,
        "intent": result.get("intent", "?"),
        "answer": result.get("answer", "")[:500],
        "answer_length": len(result.get("answer", "")),
        "route": "llm",
        "citation_count": audit.get("evidence_count", 0),
        "citation_audit_ok": audit.get("ok", False),
        "citation_issues": len(audit.get("issues", [])),
        "content_verified": verify.get("total", 0) if verify else 0,
        "content_verified_ok": verify.get("verified", 0) if verify else 0,
        "content_partial": verify.get("partial", 0) if verify else 0,
        "content_hallucinated": verify.get("hallucinated", 0) if verify else 0,
        "content_unverifiable": verify.get("unverifiable", 0) if verify else 0,
        "crag_score": (audit.get("crag_report") or {}).get("score", 0),
        "crag_recovery_used": audit.get("crag_recovery_used", False),
        "elapsed_s": round(elapsed, 1),
    }


def print_report(results):
    total = len(results)
    errors = [r for r in results if "error" in r]
    local = [r for r in results if r.get("route") == "local"]
    llm = [r for r in results if r.get("route") == "llm"]

    print(f"\n{'='*70}")
    print(f"  MarxOS E2E Answer Quality Evaluation")
    print(f"{'='*70}")
    print(f"  Questions: {total}")
    print(f"  LLM routed: {len(llm)}  |  Local routed: {len(local)}  |  Errors: {len(errors)}")
    print()

    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"    [{e['id']}] {e['query'][:60]} — {e['error']}")
        print()

    if not llm:
        print("  No LLM-routed answers to evaluate.")
        return

    # Citation metrics
    audited = [r for r in llm if r.get("citation_count", 0) > 0]
    audit_ok = sum(1 for r in audited if r.get("citation_audit_ok"))
    verified = [r for r in audited if r.get("content_verified", 0) > 0]
    total_verified = sum(r.get("content_verified", 0) for r in verified)
    total_ok = sum(r.get("content_verified_ok", 0) for r in verified)
    total_partial = sum(r.get("content_partial", 0) for r in verified)
    total_hallucinated = sum(r.get("content_hallucinated", 0) for r in verified)

    avg_length = sum(r["answer_length"] for r in llm) / len(llm)
    avg_citations = sum(r.get("citation_count", 0) for r in llm) / len(llm)
    avg_time = sum(r["elapsed_s"] for r in llm) / len(llm)

    print(f"  ── Answer Quality ──")
    print(f"  Avg answer length:    {avg_length:.0f} chars")
    print(f"  Avg citations/answer: {avg_citations:.1f}")
    print(f"  Avg time/query:       {avg_time:.1f}s")
    print()

    print(f"  ── Citation Audit ──")
    print(f"  Answers with citations: {len(audited)}/{len(llm)}")
    print(f"  Format audit pass:     {audit_ok}/{len(audited)} ({100*audit_ok/max(len(audited),1):.0f}%)")
    print()

    if verified:
        print(f"  ── Content Verification (Citation Verifier) ──")
        print(f"  Citations verified:    {total_verified}")
        print(f"  Verified (exact):      {total_ok} ({100*total_ok/max(total_verified,1):.0f}%)")
        print(f"  Partial (paraphrase):  {total_partial} ({100*total_partial/max(total_verified,1):.0f}%)")
        print(f"  Hallucinated:          {total_hallucinated} ({100*total_hallucinated/max(total_verified,1):.0f}%)")
        print()

    # By discipline
    print(f"  ── By Discipline ──")
    disc_labels = {
        "philosophy": "哲学", "political_economy": "政治经济学",
        "scientific_socialism": "科学社会主义", "party_labor": "党建工人运动",
        "peasant_land": "农民与土地", "national_colonial": "民族与殖民",
        "state_revolution_military": "国家革命军事", "history_religion_culture": "历史宗教文化",
    }
    by_disc = defaultdict(list)
    for r in llm:
        disc = "unknown"
        for q in load_dataset():
            if q["id"] == r["id"]:
                disc = q.get("discipline", "unknown")
                break
        by_disc[disc].append(r)

    for disc in sorted(by_disc):
        rs = by_disc[disc]
        ok_count = sum(1 for r in rs if r.get("citation_audit_ok"))
        hall = sum(r.get("content_hallucinated", 0) for r in rs)
        label = disc_labels.get(disc, disc)
        print(f"  {label:<12}: {len(rs)} answers, audit_ok={ok_count}/{len(rs)}, hallucinated={hall}")

    # CRAG
    crag_used = sum(1 for r in llm if r.get("crag_recovery_used"))
    print(f"\n  CRAG recovery used: {crag_used}/{len(llm)}")

    print(f"\n  Total time: {sum(r['elapsed_s'] for r in results):.0f}s")
    print(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40, help="Questions to evaluate (default: 40)")
    parser.add_argument("--all", action="store_true", help="Evaluate all 200 questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = load_dataset()

    if args.all:
        questions = dataset
    else:
        questions = sample_questions(dataset, args.count)

    print(f"Evaluating {len(questions)} questions with real DeepSeek calls...")
    print(f"(Estimated time: ~{len(questions) * 3}s)")

    results = []
    for i, q in enumerate(questions):
        query = q["question"]
        qid = q["id"]
        wid = q.get("expected_work_id", "")
        print(f"  [{i+1}/{len(questions)}] Q{qid}: {query[:60]}...", end=" ", flush=True)
        r = evaluate_one(query, qid, wid)
        route = r.get("route", "error")
        cit = r.get("citation_count", 0)
        hall = r.get("content_hallucinated", 0)
        print(f"{route} cit={cit} hall={hall} {r['elapsed_s']}s")
        results.append(r)

    print_report(results)

    # Save detailed results
    out = ROOT / "logs" / "answer_quality_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out}")


if __name__ == "__main__":
    main()
