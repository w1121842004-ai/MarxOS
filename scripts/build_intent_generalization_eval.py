#!/usr/bin/env python3
"""Build an out-of-training-distribution intent generalization eval set.

The output is a 400-item test-only set whose queries are checked for exact
non-overlap against the existing 10k synthetic set, the 2k AI-assisted set,
and the two legacy 400 calibration files.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_ai_assisted_intent_dataset import (
    CATALOG_CONCEPT_BLACKLIST,
    CURATED_CONCEPTS,
    DISCIPLINES,
    FALLBACK_QUOTES,
    FALLBACK_TITLES,
    INTENTS,
    LABEL_GUIDE,
    ROOT,
    collect_entities,
    load_work_catalog,
)

DEFAULT_OUTPUT_DIR = ROOT / "data" / "intent_generalization_400"
SEED = 20260621 + 400

HOLDOUT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "bibliographic_lookup": [
        {"q": "要追到原典，{concept}最该从哪几篇马克思文本查起？", "reason": "询问主题对应原典入口，目标是文献定位。", "hard": True},
        {"q": "别解释内容，帮我找{title}在全集里的位置。", "reason": "明确排除解释，要求全集位置。"},
        {"q": "我只需要{alias}的版本/卷次信息。", "reason": "要求篇目版本和卷次信息。"},
        {"q": "关于{concept}的经典表述大概散见在哪些著作？", "reason": "询问主题论述分布于哪些著作。", "hard": True},
        {"q": "{title}这篇在目录里怎么检索？", "reason": "询问篇目检索入口。"},
        {"q": "如果我要引用{concept}，先定位哪部原著比较稳？", "reason": "为引用寻找原著来源，属于文献定位。", "hard": True},
        {"q": "请把{alias}对应到马恩全集的具体文本来源。", "reason": "要求把篇名对应到全集文本来源。"},
        {"q": "{concept}不是解释题，我想知道它在哪些原典里出现。", "reason": "用户明确强调定位原典而非解释概念。", "hard": True},
    ],
    "quote_lookup": [
        {"q": "这句“{quote}”能帮我查到原始出处吗？", "reason": "具体引文出处核验。"},
        {"q": "我不确定“{quote}”是不是原文，帮我核一下。", "reason": "核验具体引文是否原文。"},
        {"q": "{quote}——这句话应该标哪部书哪一页？", "reason": "具体引文页码与书目来源。"},
        {"q": "有人引用“{quote}”，我想看它前后文。", "reason": "要求具体引文上下文。"},
        {"q": "帮我查{quote}有没有更准确的译文出处。", "reason": "围绕具体句子的译文和出处核验。", "hard": True},
        {"q": "这段话是不是出自{title}：“{quote}”", "reason": "核验引文与著作对应关系。"},
        {"q": "{quote}这句话别概括，直接找出处。", "reason": "具体句子出处检索。", "hard": True},
        {"q": "“{quote}”常被引用，它的页码能确定吗？", "reason": "具体引文页码核验。"},
    ],
    "concept_explain": [
        {"q": "刚入门的话，{concept}可以怎么解释？", "reason": "入门式概念解释。"},
        {"q": "{concept}这个术语的意思先讲清楚。", "reason": "要求解释术语含义。"},
        {"q": "不用展开文本，先说{concept}的定义。", "reason": "明确要求定义而非文本分析。", "hard": True},
        {"q": "在马克思主义语境中，{concept}指什么？", "reason": "解释概念在理论语境中的含义。"},
        {"q": "我总分不清{concept}，能用例子说明吗？", "reason": "要求通过例子解释概念。"},
        {"q": "{title}里出现{concept}时，应按什么含义理解？", "reason": "虽有文本名，核心仍是概念释义。", "hard": True},
        {"q": "{concept}的反面或误解通常是什么？", "reason": "通过误解澄清概念含义。"},
        {"q": "请把{concept}解释成非专业读者能懂的话。", "reason": "通俗概念解释。"},
    ],
    "comparison": [
        {"q": "{concept_a}和{concept_b}听起来接近，实际差在哪里？", "reason": "比较两个概念差异。"},
        {"q": "不要分别讲，直接比较{title_a}与{title_b}。", "reason": "明确要求两文本比较。"},
        {"q": "{concept}在马克思和恩格斯那里侧重点一样吗？", "reason": "比较两位作者对同一概念的侧重。", "hard": True},
        {"q": "{title_a}与{title_b}都涉及{concept}，能否对照一下？", "reason": "两文本围绕同一主题对照。"},
        {"q": "{concept_a}、{concept_b}哪个更接近马克思的原意？", "reason": "要求两个概念关系和差异判断。", "hard": True},
        {"q": "早期文本里的{concept}和成熟政治经济学里的{concept}有变化吗？", "reason": "隐含阶段/语境比较。", "hard": True},
        {"q": "{title_a}和{title_b}的批判对象有什么不同？", "reason": "比较两部文本的批判对象。"},
        {"q": "请用表格思路区分{concept_a}、{concept_b}和{concept}。", "reason": "多概念区分。"},
    ],
    "deep_analysis": [
        {"q": "把{concept}放到当代平台劳动里，能形成什么分析框架？", "reason": "理论应用于当代平台劳动并要求框架。"},
        {"q": "如果以{concept}写论文，问题意识和章节怎么设计？", "reason": "论文设计任务。"},
        {"q": "结合{title}和今天的资本主义危机做一个综合分析。", "reason": "经典文本与当代危机综合分析。"},
        {"q": "请从多个原典串起{concept}的理论演变。", "reason": "跨文本理论发展综合。"},
        {"q": "用马克思主义解释数字资本主义中的{concept}，要怎么展开？", "reason": "当代数字资本主义应用分析。"},
        {"q": "我想做{concept}研究综述，请给出主题结构。", "reason": "研究综述结构设计。"},
        {"q": "围绕{concept}与现实治理，写一段学术分析思路。", "reason": "理论结合现实治理。"},
        {"q": "{title}对理解当代社会矛盾有什么启发？", "reason": "经典文本的当代启示分析。", "hard": True},
    ],
    "theory_analysis": [
        {"q": "{title}的论证链条是什么？", "reason": "分析具体文本论证结构。"},
        {"q": "马克思为什么要批判{concept}的表面理解？", "reason": "理论批判逻辑分析。"},
        {"q": "{concept}在马克思理论体系里承担什么功能？", "reason": "分析概念理论功能。"},
        {"q": "请分析{alias}的核心理论贡献。", "reason": "分析篇目理论贡献。"},
        {"q": "{title}中关于{concept}的论述重点是什么？", "reason": "分析文本中的理论重点。"},
        {"q": "为什么说{concept}不是单纯经验描述？", "reason": "理论命题辨析。", "hard": True},
        {"q": "{title}的批判锋芒主要指向哪里？", "reason": "分析文本批判对象。"},
        {"q": "如何理解马克思对{concept}的历史性说明？", "reason": "分析理论概念的历史性。"},
    ],
    "rag_answer": [
        {"q": "我想从零开始读马克思，先怎么安排？", "reason": "学习路径建议。"},
        {"q": "马克思主义为什么会分成几个组成部分来讲？", "reason": "普通背景问答。"},
        {"q": "能给我一个原典阅读的入门路线吗？", "reason": "泛学习建议，不定位具体篇目。"},
        {"q": "读{title}之前先了解哪些历史背景比较好？", "reason": "阅读背景建议。", "hard": True},
        {"q": "马克思和恩格斯合作的大致脉络是什么？", "reason": "普通背景介绍。"},
        {"q": "本科课堂上讲{concept}，可以先抛出哪些问题？", "reason": "课堂讨论设计，不是概念定义。", "hard": True},
        {"q": "我需要一份马克思主义经典阅读清单。", "reason": "阅读资源整理。"},
        {"q": "解释一下为什么很多社会科学都引用马克思。", "reason": "宽泛背景问答。"},
    ],
}


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def forbidden_queries() -> set[str]:
    paths = [
        ROOT / "eval_dataset_v2.json",
        ROOT / "eval_dataset_me_200.json",
        ROOT / "data" / "intent_dataset_10000" / "intent_dataset_10000.json",
        ROOT / "data" / "intent_dataset_ai_assisted_2000" / "intent_dataset_ai_assisted_2000.json",
    ]
    queries: set[str] = set()
    for path in paths:
        for item in load_json_records(path):
            query = item.get("query") or item.get("question")
            if query:
                queries.add(str(query))
    return queries


def choose_values(rng: random.Random, entities: dict[str, Any]) -> dict[str, str]:
    work = rng.choice(entities["works"]) if entities["works"] else {}
    work_titles = work.get("titles") or entities["titles"]
    work_aliases = work.get("aliases") or entities["aliases"]
    work_quotes = work.get("quotes") or entities["quotes"]
    concept_pool = list(dict.fromkeys(CURATED_CONCEPTS))
    if rng.random() < 0.15:
        concept_pool = [
            c for c in entities["concepts"]
            if c not in CATALOG_CONCEPT_BLACKLIST and 2 <= len(c) <= 12
        ] or concept_pool
    concept_a, concept_b = rng.sample(concept_pool, 2)
    title_a, title_b = rng.sample(entities["titles"] + FALLBACK_TITLES, 2)
    return {
        "title": rng.choice(work_titles or FALLBACK_TITLES),
        "alias": rng.choice(work_aliases or work_titles or FALLBACK_TITLES),
        "quote": rng.choice(work_quotes or FALLBACK_QUOTES),
        "concept": rng.choice(concept_pool),
        "concept_a": concept_a,
        "concept_b": concept_b,
        "title_a": title_a,
        "title_b": title_b,
    }


def target_counts(total: int) -> dict[str, int]:
    base = total // len(INTENTS)
    remainder = total % len(INTENTS)
    return {intent: base + (1 if i < remainder else 0) for i, intent in enumerate(INTENTS)}


def build_records(total: int, seed: int, catalog_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if total != 400:
        raise ValueError("This builder enforces total=400.")
    rng = random.Random(seed)
    entities = collect_entities(load_work_catalog(catalog_path))
    forbidden = forbidden_queries()
    seen = set(forbidden)
    records: list[dict[str, Any]] = []
    rec_id = 1

    for intent, count in target_counts(total).items():
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 500:
                raise RuntimeError(f"Could not generate enough unique {intent} records")
            template = rng.choice(HOLDOUT_TEMPLATES[intent])
            values = choose_values(rng, entities)
            query = template["q"].format(**values)
            if query in seen:
                continue
            seen.add(query)
            hard = bool(template.get("hard"))
            confidence = round(rng.uniform(0.72, 0.84) if hard else rng.uniform(0.86, 0.97), 2)
            records.append({
                "id": f"gen_intent_{rec_id:04d}",
                "query": query,
                "intent": intent,
                "question_type": intent,
                "split": "test",
                "source": "ai_assisted_generalization_eval_v1",
                "confidence": confidence,
                "needs_human_review": hard and confidence < 0.8,
                "boundary_case": hard,
                "label_reason": template["reason"],
                "difficulty": "hard" if hard else rng.choice(["easy", "medium", "hard"]),
                "discipline": rng.choice(DISCIPLINES),
                "entities": values,
            })
            rec_id += 1
            made += 1
    rng.shuffle(records)
    return records, forbidden


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def summarize(records: list[dict[str, Any]], forbidden: set[str]) -> dict[str, Any]:
    queries = {r["query"] for r in records}
    return {
        "total": len(records),
        "split": "test",
        "intents": dict(Counter(r["intent"] for r in records)),
        "boundary_cases": sum(1 for r in records if r["boundary_case"]),
        "needs_human_review": sum(1 for r in records if r["needs_human_review"]),
        "avg_confidence": round(sum(r["confidence"] for r in records) / len(records), 4),
        "exact_overlap_with_forbidden": len(queries & forbidden),
        "forbidden_query_pool_size": len(forbidden),
        "source": "ai_assisted_generalization_eval_v1",
        "note": "Held-out AI-assisted generalization eval; exact queries do not overlap with current train/validation/test/calibration datasets.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 400-item held-out intent generalization eval")
    parser.add_argument("--total", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=ROOT / "rag" / "work_catalog.json")
    args = parser.parse_args()

    records, forbidden = build_records(args.total, args.seed, args.catalog)
    write_json(args.output_dir / "intent_generalization_400.json", records)
    write_jsonl(args.output_dir / "intent_generalization_400.jsonl", records)
    write_json(args.output_dir / "labeling_guide.json", LABEL_GUIDE)
    write_json(args.output_dir / "summary.json", summarize(records, forbidden))
    print(json.dumps(summarize(records, forbidden), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
