#!/usr/bin/env python3
"""Build a large synthetic intent-classification dataset.

The generated data is designed for pretraining / distillation of the MarxOS
intent router.  It is not a replacement for human labels: keep its source field
when mixing it with manually labeled evaluation data.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "intent_dataset_10000"
SEED = 20260621

INTENTS = [
    "bibliographic_lookup",
    "quote_lookup",
    "concept_explain",
    "comparison",
    "deep_analysis",
    "theory_analysis",
    "rag_answer",
]

DISCIPLINES = [
    "philosophy",
    "political_economy",
    "scientific_socialism",
    "history",
    "letters",
]

DIFFICULTIES = ["easy", "medium", "hard"]

FALLBACK_TITLES = [
    "《资本论》第一卷",
    "《共产党宣言》",
    "《德意志意识形态》",
    "《关于费尔巴哈的提纲》",
    "《反杜林论》",
    "《哥达纲领批判》",
]

FALLBACK_CONCEPTS = [
    "剩余价值",
    "异化劳动",
    "商品拜物教",
    "历史唯物主义",
    "阶级斗争",
    "无产阶级专政",
    "生产关系",
    "资本积累",
    "意识形态",
    "辩证法",
]

FALLBACK_QUOTES = [
    "全世界无产者，联合起来！",
    "哲学家们只是用不同的方式解释世界，问题在于改变世界。",
    "宗教是人民的鸦片。",
    "人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
]

OUT_OF_DOMAIN_TOPICS = [
    "今天上海天气怎么样",
    "帮我写一首现代诗",
    "Python 怎么读取 JSON 文件",
    "推荐几本入门社会学教材",
    "如何安排一周健身计划",
    "解释一下机器学习里的过拟合",
]


TEMPLATES: dict[str, list[str]] = {
    "bibliographic_lookup": [
        "{title}收录在哪一卷？",
        "{title}在《马克思恩格斯全集》第几卷？",
        "请帮我查一下{title}的卷册位置",
        "{alias}出自哪部著作或哪一卷？",
        "马克思关于{concept}的论述主要在哪些文本里？",
        "恩格斯谈{concept}的材料应查哪一卷？",
        "{title}有没有对应的全集卷次？",
        "我想找{title}的原始出处",
        "{concept}相关篇目在马恩全集里怎么定位？",
        "查找{alias}的收录信息",
        "{title}大概在哪个版本的文集中？",
        "哪里可以找到{title}这篇文章？",
    ],
    "quote_lookup": [
        "{quote}出自哪里？",
        "“{quote}”这句话在哪一页？",
        "请核对“{quote}”的原文出处",
        "{quote}这段话是哪部著作里的？",
        "马克思是不是说过“{quote}”？",
        "帮我找一下“{quote}”的准确页码",
        "这句话“{quote}”收录在哪一卷？",
        "{quote}的上下文是什么？",
        "请给出“{quote}”的来源和页码",
        "“{quote}”是否出自{title}？",
        "我看到一句“{quote}”，它的出处可靠吗？",
        "{quote}这句经典表述应怎样引用？",
    ],
    "concept_explain": [
        "什么是{concept}？",
        "{concept}是什么意思？",
        "请解释马克思所说的{concept}",
        "{title}中的{concept}是什么？",
        "如何理解{concept}这个概念？",
        "{concept}的基本内涵是什么？",
        "简述{concept}在马克思主义中的含义",
        "{concept}和日常说法有什么不同？",
        "马克思为什么重视{concept}？",
        "{concept}的理论来源是什么？",
        "用通俗语言解释{concept}",
        "{alias}里提到的{concept}该怎么理解？",
    ],
    "comparison": [
        "比较{concept_a}和{concept_b}的区别",
        "{concept_a}与{concept_b}有什么异同？",
        "{title_a}和{title_b}在{concept}问题上有什么不同？",
        "马克思和恩格斯关于{concept}的论述有何差异？",
        "对比{title_a}与{title_b}的核心观点",
        "{concept_a}是不是等同于{concept_b}？",
        "{title_a}中的{concept_a}和{title_b}中的{concept_b}如何比较？",
        "从理论脉络看{concept_a}和{concept_b}的关系",
        "{concept}在早期马克思和晚期马克思那里是否不同？",
        "{title_a}与{title_b}对历史发展的理解有什么差别？",
        "{concept_a}、{concept_b}和{concept}三者是什么关系？",
        "请列出{concept_a}和{concept_b}的主要区别",
    ],
    "deep_analysis": [
        "运用马克思主义分析当代{concept}问题",
        "从马克思主义视角分析数字时代的{concept}",
        "写一篇关于{concept}现实意义的理论短文",
        "结合{title}论述{concept}的当代价值",
        "如何用马克思主义解释平台经济中的{concept}？",
        "请系统梳理马克思关于{concept}的理论发展",
        "围绕{concept}写一个论文提纲",
        "从历史唯物主义角度分析当代社会的{concept}",
        "{concept}理论对理解人工智能劳动有什么启示？",
        "结合现实案例分析{concept}的理论意义",
        "请做一个关于{concept}的研究综述",
        "从{title}出发分析现代资本主义的新变化",
    ],
    "theory_analysis": [
        "马克思如何论述{concept}？",
        "分析{title}中的主要观点",
        "{title}的核心论证是什么？",
        "为什么说{concept}是马克思主义的重要范畴？",
        "如何理解{title}中的这段理论逻辑？",
        "恩格斯如何评价{concept}？",
        "请说明{concept}在马克思理论体系中的地位",
        "{alias}的理论贡献是什么？",
        "马克思关于{concept}的观点有哪些层次？",
        "从文本角度分析{concept}的含义",
        "{title}主要批判了什么？",
        "为什么{title}对理解马克思思想很重要？",
    ],
    "rag_answer": [
        "介绍一下马克思主义的基本原理",
        "马克思有哪些主要著作？",
        "恩格斯的理论贡献有哪些？",
        "马克思主义的发展经历了哪些阶段？",
        "请给我一个学习马克思主义的阅读顺序",
        "什么是科学社会主义？",
        "马克思和恩格斯是什么关系？",
        "马克思主义为什么影响深远？",
        "{out_of_domain}",
        "能不能简单介绍一下19世纪欧洲工人运动？",
        "给我整理一个马克思主义入门书单",
        "请概括马克思主义哲学、政治经济学和科学社会主义的关系",
        "我应该先读{title}还是先了解{concept}？",
        "学习{concept}需要哪些背景知识？",
        "请推荐理解{title}的入门路径",
        "能不能概括一下{title}的阅读价值？",
        "初学者怎样进入{concept}这个主题？",
        "请用简短方式介绍{concept}相关的马克思主义知识",
        "{title}适合放在马克思主义学习的哪个阶段？",
        "围绕{concept}有哪些基础问题值得先了解？",
        "请给我一个关于{concept}的学习提纲",
        "读{title}之前需要知道什么？",
        "{alias}和马克思主义整体框架有什么关系？",
        "帮我整理一组关于{concept}的入门问题",
        "如果只想初步了解{title}，应该抓住哪些关键词？",
        "请概括{concept}在马克思主义知识体系中的位置",
        "围绕{title}可以设计哪些课堂讨论问题？",
        "我想了解{concept}，可以从哪些文本开始？",
        "请列出阅读{title}时容易困惑的几个点",
        "马克思主义研究中为什么经常讨论{concept}？",
        "请给本科生介绍一下{title}的背景",
        "围绕{concept}做读书报告可以怎么展开？",
    ],
}


def load_work_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def collect_entities(catalog: dict[str, Any]) -> dict[str, list[str]]:
    works = catalog.get("works") or []
    titles: list[str] = []
    aliases: list[str] = []
    concepts: list[str] = []
    quotes: list[str] = []
    work_entities: list[dict[str, list[str]]] = []

    for work in works:
        title = work.get("title")
        work_titles = [str(title)] if title else []
        work_aliases = [str(a) for a in work.get("aliases") or [] if a]
        work_concepts = [str(c) for c in work.get("concepts") or [] if c]
        work_quotes = [str(q) for q in work.get("quotes") or [] if q]
        if title:
            titles.append(str(title))
        aliases.extend(work_aliases)
        concepts.extend(work_concepts)
        quotes.extend(work_quotes)
        if work_titles or work_aliases or work_concepts or work_quotes:
            work_entities.append({
                "titles": unique_nonempty(work_titles),
                "aliases": unique_nonempty(work_aliases + work_titles),
                "concepts": unique_nonempty(work_concepts),
                "quotes": unique_nonempty(work_quotes),
            })

    return {
        "titles": unique_nonempty(titles + FALLBACK_TITLES),
        "aliases": unique_nonempty(aliases + titles + FALLBACK_TITLES),
        "concepts": unique_nonempty(concepts + FALLBACK_CONCEPTS),
        "quotes": unique_nonempty(quotes + FALLBACK_QUOTES),
        "works": work_entities,  # type: ignore[dict-item]
    }


def target_counts(total: int) -> dict[str, int]:
    base = total // len(INTENTS)
    remainder = total % len(INTENTS)
    return {
        intent: base + (1 if index < remainder else 0)
        for index, intent in enumerate(INTENTS)
    }


def split_counts_for_intent(index: int, total_for_intent: int) -> dict[str, int]:
    # 72 * 3 + 71 * 4 = 500 for validation and the same for test when total=10000.
    val_or_test = 72 if index < 3 else 71
    train = total_for_intent - val_or_test * 2
    return {"train": train, "validation": val_or_test, "test": val_or_test}


def choose_entities(rng: random.Random, entities: dict[str, list[str]]) -> dict[str, str]:
    concepts = entities["concepts"]
    titles = entities["titles"]
    aliases = entities["aliases"]
    quotes = entities["quotes"]
    works = entities.get("works") or []  # type: ignore[assignment]
    work = rng.choice(works) if works else {}
    work_concepts = work.get("concepts") or concepts  # type: ignore[union-attr]
    work_titles = work.get("titles") or titles  # type: ignore[union-attr]
    work_aliases = work.get("aliases") or aliases  # type: ignore[union-attr]
    work_quotes = work.get("quotes") or quotes  # type: ignore[union-attr]

    if len(work_concepts) >= 2:
        concept_a, concept_b = rng.sample(work_concepts, 2)
    else:
        concept_a, concept_b = rng.sample(concepts, 2)
    title_a, title_b = rng.sample(titles, 2)
    return {
        "title": rng.choice(work_titles),
        "alias": rng.choice(work_aliases),
        "concept": rng.choice(work_concepts),
        "quote": rng.choice(work_quotes),
        "title_a": title_a,
        "title_b": title_b,
        "concept_a": concept_a,
        "concept_b": concept_b,
        "out_of_domain": rng.choice(OUT_OF_DOMAIN_TOPICS),
    }


def make_query(intent: str, rng: random.Random, entities: dict[str, list[str]]) -> tuple[str, str, dict[str, str]]:
    template = rng.choice(TEMPLATES[intent])
    values = choose_entities(rng, entities)
    return template.format(**values), template, values


def make_record(
    rec_id: int,
    split: str,
    intent: str,
    rng: random.Random,
    entities: dict[str, list[str]],
) -> dict[str, Any]:
    query, template, values = make_query(intent, rng, entities)
    return {
        "id": f"intent_{rec_id:05d}",
        "query": query,
        "intent": intent,
        "question_type": intent,
        "split": split,
        "source": "synthetic_intent_v1",
        "template": template,
        "difficulty": rng.choice(DIFFICULTIES),
        "discipline": rng.choice(DISCIPLINES),
        "entities": {
            key: values[key]
            for key in (
                "title",
                "alias",
                "concept",
                "quote",
                "title_a",
                "title_b",
                "concept_a",
                "concept_b",
            )
        },
    }


def build_dataset(total: int, seed: int, catalog_path: Path) -> list[dict[str, Any]]:
    if total != 10000:
        raise ValueError("This builder currently enforces total=10000 to keep exact 90/5/5 splits.")

    rng = random.Random(seed)
    entities = collect_entities(load_work_catalog(catalog_path))
    totals = target_counts(total)
    records: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    rec_id = 1

    for intent_index, intent in enumerate(INTENTS):
        split_targets = split_counts_for_intent(intent_index, totals[intent])
        for split, count in split_targets.items():
            made = 0
            attempts = 0
            while made < count:
                attempts += 1
                if attempts > count * 100:
                    raise RuntimeError(f"Could not generate enough unique records for {intent}/{split}")
                record = make_record(rec_id, split, intent, rng, entities)
                if record["query"] in seen_queries:
                    continue
                seen_queries.add(record["query"])
                records.append(record)
                rec_id += 1
                made += 1

    rng.shuffle(records)
    return records


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    intent_counts = Counter(record["intent"] for record in records)
    by_split_intent: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        by_split_intent[split] = dict(
            Counter(record["intent"] for record in records if record["split"] == split)
        )
    return {
        "total": len(records),
        "splits": dict(split_counts),
        "intents": dict(intent_counts),
        "by_split_intent": by_split_intent,
        "source": "synthetic_intent_v1",
        "note": "Synthetic labels for intent-router pretraining/distillation; do not treat as a human gold benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 10k synthetic MarxOS intent labels")
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=ROOT / "rag" / "work_catalog.json")
    args = parser.parse_args()

    records = build_dataset(args.total, args.seed, args.catalog)
    splits = {
        "train": [record for record in records if record["split"] == "train"],
        "validation": [record for record in records if record["split"] == "validation"],
        "test": [record for record in records if record["split"] == "test"],
    }

    write_json(args.output_dir / "intent_dataset_10000.json", records)
    write_json(args.output_dir / "intent_train.json", splits["train"])
    write_json(args.output_dir / "intent_validation.json", splits["validation"])
    write_json(args.output_dir / "intent_test.json", splits["test"])
    write_jsonl(args.output_dir / "intent_train.jsonl", splits["train"])
    write_jsonl(args.output_dir / "intent_validation.jsonl", splits["validation"])
    write_jsonl(args.output_dir / "intent_test.jsonl", splits["test"])
    write_json(args.output_dir / "summary.json", summarize(records))

    summary = summarize(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
