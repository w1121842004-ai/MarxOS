"""
Evaluate LLM deep analysis quality with novel questions across 8 disciplines.

Measures: citation accuracy, hallucination rate, CRAG behavior, answer structure.
Each question triggers the full LLM pipeline (no local shortcuts).

Usage:
    venv/Scripts/python.exe scripts/evaluate_deep_analysis.py
    venv/Scripts/python.exe scripts/evaluate_deep_analysis.py --count 20
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
os.environ.setdefault("MARXOS_DEV_MODE", "1")

# ── 50 deep analysis questions across 8 disciplines ──────────────

QUESTIONS = [
    # === 哲学 (7) ===
    ("从马克思主义实践观出发，分析当代技术社会中人与技术的关系异化", "philosophy"),
    ("运用历史唯物主义方法，分析数字资本主义时代的生产关系变革", "philosophy"),
    ("马克思的意识形态理论对理解当代社交媒体信息茧房有何启示", "philosophy"),
    ("从辩证法的角度，分析全球化进程中的同一性与差异性张力", "philosophy"),
    ("运用马克思的异化理论，分析现代消费社会中人的物化现象", "philosophy"),
    ("从《关于费尔巴哈的提纲》出发，论述马克思主义哲学的革命性变革", "philosophy"),
    ("分析人工智能发展对历史唯物主义'生产力决定生产关系'命题的挑战与印证", "philosophy"),

    # === 政治经济学 (8) ===
    ("从马克思的剩余价值理论出发，分析数字平台经济的价值提取机制", "political_economy"),
    ("运用《资本论》的资本积累理论，分析当代全球财富不平等的历史根源", "political_economy"),
    ("马克思的危机理论对理解2008年全球金融危机和当代经济动荡的启示", "political_economy"),
    ("分析金融化时代虚拟资本对实体经济的影响——基于马克思的生息资本理论", "political_economy"),
    ("从劳动价值论出发，分析零工经济中劳动力商品化的新形式", "political_economy"),
    ("马克思的地租理论对理解当代城市住房危机和房地产金融化的意义", "political_economy"),
    ("分析当代资本主义从福特制向后福特制转型中劳动过程的变迁", "political_economy"),
    ("从《资本论》原始积累理论出发，分析当代数据资本原始积累的特征", "political_economy"),

    # === 科学社会主义 (6) ===
    ("从《共产党宣言》出发，分析当代全球阶级结构的变化与阶级斗争新形式", "scientific_socialism"),
    ("运用马克思主义国家学说，分析数字时代国家治理的转型与困境", "scientific_socialism"),
    ("从《哥达纲领批判》出发，论述共产主义社会两个阶段理论对当代社会主义实践的启示", "scientific_socialism"),
    ("分析马克思主义对当代新自由主义意识形态的批判", "scientific_socialism"),
    ("从《社会主义从空想到科学的发展》出发，论述科学社会主义在21世纪的理论活力", "scientific_socialism"),
    ("马克思主义政党理论视角下当代左翼政治运动的困境与出路", "scientific_socialism"),

    # === 党建与工人运动 (5) ===
    ("从第一国际的历史经验出发，分析当代全球工人运动面临的挑战与机遇", "party_labor"),
    ("运用马克思主义政党理论，分析数字时代工人阶级组织形式的变迁", "party_labor"),
    ("马克思和恩格斯关于党的纪律的思想对当代无产阶级政党的启示", "party_labor"),
    ("从工人运动史出发，分析平台经济时代劳动者集体行动的新形式", "party_labor"),
    ("马克思主义工会理论在当代跨国资本流动条件下的适用性分析", "party_labor"),

    # === 农民与土地 (5) ===
    ("从马克思的地租理论和恩格斯农民问题思想出发，分析当代全球土地掠夺现象", "peasant_land"),
    ("运用马克思主义土地问题理论，分析发展中国家城市化进程中的农民失地问题", "peasant_land"),
    ("恩格斯《法德农民问题》对理解当代农业资本主义化和农民分化的启示", "peasant_land"),
    ("从马克思主义视角分析全球粮食主权运动的理论基础和实践路径", "peasant_land"),
    ("当代资本主义农业中资本积累与生态危机的马克思主义分析", "peasant_land"),

    # === 民族与殖民 (6) ===
    ("从马克思的殖民主义理论出发，分析当代新殖民主义的经济控制机制", "national_colonial"),
    ("运用马克思主义民族理论，分析全球化背景下民族国家主权的变迁", "national_colonial"),
    ("马克思对印度殖民地的分析与当代全球南方的依附性发展", "national_colonial"),
    ("从马克思主义视角分析'一带一路'倡议中的国际经济关系", "national_colonial"),
    ("当代数字殖民主义——马克思主义殖民理论在数字时代的新发展", "national_colonial"),
    ("马克思主义民族自决权理论对理解当代分离主义运动的启示", "national_colonial"),

    # === 国家、革命与军事 (7) ===
    ("从马克思对国家本质的分析出发，论述当代资本主义国家的职能转型", "state_revolution_military"),
    ("运用马克思主义革命理论，分析21世纪社会运动的特点与革命可能性", "state_revolution_military"),
    ("从《法兰西内战》出发，分析巴黎公社原则对当代民主治理的启示", "state_revolution_military"),
    ("马克思主义战争与和平理论视角下的当代地缘政治冲突分析", "state_revolution_military"),
    ("从恩格斯军事思想出发，分析现代战争中技术与人的关系变迁", "state_revolution_military"),
    ("分析当代资本主义国家的福利制度转型——基于马克思主义国家理论", "state_revolution_military"),
    ("马克思主义视域下数字监控国家与公民自由的辩证关系", "state_revolution_military"),

    # === 历史、宗教与文化 (6) ===
    ("从马克思的宗教批判理论出发，分析当代消费主义作为一种世俗宗教的特征", "history_religion_culture"),
    ("运用马克思主义历史分析方法，论述资本主义发展不平衡规律的历史表现", "history_religion_culture"),
    ("从恩格斯的《家庭、私有制和国家的起源》出发，分析当代家庭形态的变迁", "history_religion_culture"),
    ("马克思主义文化理论视角下数字文化生产的商品化与异化", "history_religion_culture"),
    ("从《路德维希·费尔巴哈论》出发，分析马克思主义哲学对当代哲学的超越", "history_religion_culture"),
    ("分析当代后现代主义思潮与马克思主义历史唯物主义的对话与张力", "history_religion_culture"),
]


def evaluate_deep_analysis(count=50, seed=42):
    from app import run_query, LAST_CITATION_AUDIT, LAST_EVIDENCE

    random.seed(seed)
    questions = random.sample(QUESTIONS, min(count, len(QUESTIONS)))

    print(f"Evaluating {len(questions)} deep analysis questions with full LLM pipeline...")
    print(f"(Estimated time: ~{len(questions) * 5}s)")

    results = []
    for i, (query, discipline) in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {query[:60]}...", end=" ", flush=True)
        start = time.time()
        try:
            answer = run_query(query)
        except Exception as e:
            results.append({"query": query, "discipline": discipline, "error": str(e)})
            print(f"ERROR: {e}")
            continue

        elapsed = time.time() - start
        audit = LAST_CITATION_AUDIT or {}
        evidence = LAST_EVIDENCE or []
        verify = audit.get("content_verification", {})

        entry = {
            "query": query[:80],
            "discipline": discipline,
            "answer_length": len(answer) if answer else 0,
            "citation_count": audit.get("evidence_count", 0),
            "citation_audit_ok": audit.get("ok", False),
            "citation_issues": len(audit.get("issues", [])),
            "content_verified": verify.get("total", 0) if verify else 0,
            "content_verified_ok": verify.get("verified", 0) if verify else 0,
            "content_partial": verify.get("partial", 0) if verify else 0,
            "content_hallucinated": verify.get("hallucinated", 0) if verify else 0,
            "crag_used": audit.get("crag_recovery_used", False),
            "crag_score": (audit.get("crag_report") or {}).get("score", 0),
            "elapsed_s": round(elapsed, 1),
        }

        # Quick assessment
        has_structure = 0
        if answer:
            ans = str(answer)
            if "# " in ans or "## " in ans:
                has_structure += 1  # has markdown headings
            if "引言" in ans[:500] or "导言" in ans[:500]:
                has_structure += 1  # has introduction
            if "结论" in ans[-500:] or "总结" in ans[-500:]:
                has_structure += 1  # has conclusion
        entry["structure_score"] = has_structure  # 0-3

        print(f"len={entry['answer_length']} cit={entry['citation_count']} "
              f"audit={'ok' if entry['citation_audit_ok'] else 'fail'} "
              f"hall={entry['content_hallucinated']} struct={has_structure}/3 "
              f"{elapsed:.1f}s")

        results.append(entry)

    return results


def print_report(results):
    errors = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]

    print(f"\n{'='*70}")
    print(f"  MarxOS Deep Analysis Quality Evaluation")
    print(f"{'='*70}")
    print(f"  Questions: {len(results)}  |  Errors: {len(errors)}")
    print()

    if not valid:
        print("  No valid results.")
        return

    # Aggregate metrics
    avg_len = sum(r["answer_length"] for r in valid) / len(valid)
    avg_cit = sum(r["citation_count"] for r in valid) / len(valid)
    avg_time = sum(r["elapsed_s"] for r in valid) / len(valid)

    cit_ok = sum(1 for r in valid if r["citation_audit_ok"])
    has_cit = sum(1 for r in valid if r["citation_count"] > 0)

    total_ver = sum(r.get("content_verified", 0) for r in valid)
    total_ok = sum(r.get("content_verified_ok", 0) for r in valid)
    total_partial = sum(r.get("content_partial", 0) for r in valid)
    total_hall = sum(r.get("content_hallucinated", 0) for r in valid)
    total_unver = sum(r.get("content_unverifiable", 0) for r in valid) if "content_unverifiable" in valid[0] else 0

    crag = sum(1 for r in valid if r.get("crag_used"))
    struct_avg = sum(r["structure_score"] for r in valid) / len(valid)
    struct_perfect = sum(1 for r in valid if r["structure_score"] >= 2)

    print(f"  ── Answer Quality ──")
    print(f"  Avg answer length:     {avg_len:.0f} chars")
    print(f"  Avg citations/answer:  {avg_cit:.1f}")
    print(f"  Avg structure score:   {struct_avg:.1f}/3")
    print(f"  Structure ≥2/3:        {struct_perfect}/{len(valid)} ({100*struct_perfect/len(valid):.0f}%)")
    print(f"  Avg time/query:        {avg_time:.1f}s")
    print()

    print(f"  ── Citation Quality ──")
    print(f"  Answers with citations: {has_cit}/{len(valid)}")
    print(f"  Citation audit pass:   {cit_ok}/{has_cit} ({100*cit_ok/max(has_cit,1):.0f}%)")
    print()

    if total_ver > 0:
        total = total_ok + total_partial + total_hall + total_unver
        print(f"  ── Content Verification ──")
        print(f"  Citations verified:     {total_ver}")
        print(f"  Verified (exact):       {total_ok} ({100*total_ok/max(total,1):.0f}%)")
        print(f"  Partial (paraphrase):   {total_partial} ({100*total_partial/max(total,1):.0f}%)")
        print(f"  Hallucinated:           {total_hall} ({100*total_hall/max(total,1):.0f}%)")
        if total_unver:
            print(f"  Unverifiable (no OCR):  {total_unver}")
        print()

    print(f"  ── Additional ──")
    print(f"  CRAG recovery used:    {crag}/{len(valid)} ({100*crag/len(valid):.0f}%)")
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
    for r in valid:
        by_disc[r.get("discipline", "unknown")].append(r)

    for disc in sorted(by_disc):
        rs = by_disc[disc]
        avg_l = sum(r["answer_length"] for r in rs) / len(rs)
        cit_ok_count = sum(1 for r in rs if r["citation_audit_ok"])
        hall_count = sum(r.get("content_hallucinated", 0) for r in rs)
        label = disc_labels.get(disc, disc)
        print(f"  {label:<12}: {len(rs)}qs, avg_len={avg_l:.0f}, cit_ok={cit_ok_count}/{len(rs)}, hall={hall_count}")

    print(f"\n  Total time: {sum(r['elapsed_s'] for r in results):.0f}s")
    print(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Questions to evaluate (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = evaluate_deep_analysis(args.count, args.seed)
    print_report(results)

    out = ROOT / "logs" / "deep_analysis_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out}")


if __name__ == "__main__":
    main()
