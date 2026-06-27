#!/usr/bin/env python3
"""Build a higher-quality AI-assisted intent dataset for MarxOS.

This dataset is intentionally smaller and less template-clean than
intent_dataset_10000.  It focuses on natural Chinese query forms, boundary
cases, explicit label reasons, and review flags.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "intent_dataset_ai_assisted_2000"
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

LABEL_GUIDE = {
    "bibliographic_lookup": {
        "definition": "定位著作、篇目、卷册、版本、页码范围或文本出处位置；核心目标是找到文献位置，不是解释内容。",
        "positive_signals": ["在哪一卷", "收录在哪", "第几卷", "哪部著作", "篇目位置", "查原始出处"],
        "boundary": "若用户粘贴一段具体原文/名句并问出处，优先 quote_lookup；若问某主题在哪里论述，优先 bibliographic_lookup。",
    },
    "quote_lookup": {
        "definition": "核验具体引文、名句、段落、页码、上下文或是否为马克思/恩格斯原话。",
        "positive_signals": ["这句话", "这段话", "引文", "原文", "准确页码", "是否说过", "上下文"],
        "boundary": "具体句子即使没有引号，也优先 quote_lookup；没有具体句子而只是找主题位置，则 bibliographic_lookup。",
    },
    "concept_explain": {
        "definition": "解释概念、定义、基本内涵、通俗说明；核心目标是理解术语本身。",
        "positive_signals": ["什么是", "是什么意思", "概念", "定义", "通俗解释", "基本内涵"],
        "boundary": "若要求分析文本论证、理论地位、历史发展，转 theory_analysis；若要求当代应用/论文综合，转 deep_analysis。",
    },
    "comparison": {
        "definition": "比较两个或多个概念、著作、阶段、作者观点、理论路径之间的异同和关系。",
        "positive_signals": ["比较", "区别", "异同", "差别", "是否等同", "关系", "一致吗"],
        "boundary": "只解释一个概念不是 comparison；隐含早晚期/两文本/两概念对照也归 comparison。",
    },
    "deep_analysis": {
        "definition": "跨文本综合、论文写作、研究综述、当代问题应用、现实案例分析或复杂理论框架搭建。",
        "positive_signals": ["写一篇", "论文", "综述", "当代", "现实意义", "数字时代", "结合现实", "研究框架"],
        "boundary": "只分析一篇文本或一个理论点，通常是 theory_analysis；有当代应用、写作任务或跨文本综合时为 deep_analysis。",
    },
    "theory_analysis": {
        "definition": "分析马克思主义理论命题、文本论证、观点层次、理论地位和内在逻辑。",
        "positive_signals": ["如何论述", "核心观点", "理论逻辑", "为什么说", "分析", "批判了什么"],
        "boundary": "定义式问题是 concept_explain；当代应用/论文式任务是 deep_analysis；两个对象对照是 comparison。",
    },
    "rag_answer": {
        "definition": "普通背景问答、学习建议、泛化介绍、轻度离题或无法归入以上强意图的问题。",
        "positive_signals": ["介绍一下", "学习路径", "入门书单", "主要著作", "发展历程", "怎么学习"],
        "boundary": "只要出现强定位、引文核验、比较、论文综合等明确任务，应归入对应强意图。",
    },
}

DISCIPLINES = ["philosophy", "political_economy", "scientific_socialism", "history", "letters"]
DIFFICULTIES = ["easy", "medium", "hard"]

FALLBACK_TITLES = [
    "《资本论》第一卷",
    "《共产党宣言》",
    "《德意志意识形态》",
    "《关于费尔巴哈的提纲》",
    "《反杜林论》",
    "《哥达纲领批判》",
    "《1844年经济学哲学手稿》",
]

CURATED_CONCEPTS = [
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
    "劳动价值论",
    "国家理论",
    "资本主义生产方式",
    "雇佣劳动",
    "资本积累",
    "原始积累",
    "劳动力商品",
    "价值形式",
    "交换价值",
    "使用价值",
    "抽象劳动",
    "具体劳动",
    "必要劳动",
    "剩余劳动",
    "生产力",
    "经济基础",
    "上层建筑",
    "社会存在",
    "社会意识",
    "实践",
    "人的本质",
    "宗教批判",
    "市民社会",
    "政治解放",
    "人类解放",
    "无产阶级革命",
    "共产主义",
    "科学社会主义",
    "空想社会主义",
    "工人阶级",
    "国际主义",
    "殖民主义批判",
    "土地私有制",
    "地租",
    "利润率",
    "资本循环",
    "社会再生产",
    "经济危机",
    "机器大工业",
    "劳动过程",
    "剥削",
    "革命策略",
    "巴黎公社",
    "民主共和国",
    "农民问题",
    "民族问题",
    "世界市场",
    "自由竞争",
    "信用制度",
    "货币转化为资本",
    "拜物教批判",
    "意识形态批判",
    "自然辩证法",
    "家庭私有制和国家",
    "社会形态",
]

CATALOG_CONCEPT_BLACKLIST = {
    "书评", "波斯", "中国", "印度", "俄国", "美国", "英国", "法国", "德国",
    "西班牙", "土耳其", "波兰", "爱尔兰", "奥地利", "意大利", "瑞士",
}

FALLBACK_QUOTES = [
    "全世界无产者，联合起来！",
    "哲学家们只是用不同的方式解释世界，问题在于改变世界。",
    "宗教是人民的鸦片。",
    "人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
    "资本来到世间，从头到脚，每个毛孔都滴着血和肮脏的东西。",
    "劳动创造了人本身。",
]

OUT_OF_DOMAIN = [
    "今天上海天气怎么样",
    "帮我写一首关于劳动的诗",
    "Python 怎么读取 JSONL 文件",
    "怎么安排一周健身计划",
    "解释一下机器学习中的过拟合",
    "推荐几本社会学入门书",
]


TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "bibliographic_lookup": [
        {"q": "{title}收录在哪一卷？", "reason": "询问著作卷册位置，目标是文献定位。"},
        {"q": "我想查{alias}的全集出处，应该看哪一卷？", "reason": "询问篇目在全集中的位置，归为文献定位。"},
        {"q": "马克思在哪里集中论述{concept}？", "reason": "询问主题论述出现在哪些文本，目标是定位相关文献。", "hard": True},
        {"q": "{concept}相关篇目能帮我定位到具体著作吗？", "reason": "要求把主题映射到具体著作，属于文献定位。"},
        {"q": "{title}有没有对应的页码范围或篇目位置？", "reason": "询问页码范围和篇目位置，属于 bibliographic_lookup。"},
        {"q": "查一下{alias}是不是在马恩全集里", "reason": "确认篇目是否被全集收录，属于文献定位。"},
        {"q": "关于{concept}，先找哪几篇原典比较合适？", "reason": "目标是找原典来源，而非解释概念。", "hard": True},
        {"q": "{title}在文集和全集中分别怎么找？", "reason": "询问不同版本中的查找位置，属于文献定位。"},
        {"q": "哪部著作里有马克思关于{concept}的系统表述？", "reason": "询问理论主题的文本出处，属于 bibliographic_lookup。", "hard": True},
        {"q": "帮我定位{alias}这篇文章的原始出处", "reason": "明确要求定位文章出处。"},
    ],
    "quote_lookup": [
        {"q": "“{quote}”出自哪里？", "reason": "用户给出具体引文并询问出处。"},
        {"q": "{quote}这句话的准确页码是多少？", "reason": "具体句子页码核验，属于 quote_lookup。"},
        {"q": "马克思真的说过“{quote}”吗？", "reason": "核验具体表述是否为原话。"},
        {"q": "请找一下“{quote}”的上下文", "reason": "要求具体引文上下文，属于 quote_lookup。"},
        {"q": "我看到一句{quote}，想知道原文出处", "reason": "具体句子即使未加引号，也按引文核验处理。", "hard": True},
        {"q": "“{quote}”是不是出自{title}？", "reason": "核验引文与著作对应关系。"},
        {"q": "{quote}这段话应该怎么规范引用？", "reason": "具体段落引用格式和来源核验。"},
        {"q": "帮我确认{quote}有没有 OCR 错字", "reason": "围绕具体引文核验原文。"},
        {"q": "这句话在全集哪一页：“{quote}”", "reason": "具体引文页码定位，优先 quote_lookup 而不是 bibliographic_lookup。", "hard": True},
        {"q": "“{quote}”的前后几句是什么？", "reason": "要求引文上下文。"},
    ],
    "concept_explain": [
        {"q": "什么是{concept}？", "reason": "询问概念定义，未要求文本定位或深入分析。"},
        {"q": "{concept}是什么意思？", "reason": "要求解释术语含义。"},
        {"q": "能用通俗语言解释{concept}吗？", "reason": "通俗解释概念，属于 concept_explain。"},
        {"q": "{title}里说的{concept}大概是什么意思？", "reason": "虽然出现著作名，但核心是解释概念含义。", "hard": True},
        {"q": "{concept}的基本内涵包括哪些方面？", "reason": "询问概念基本内涵。"},
        {"q": "初学者怎么理解{concept}这个词？", "reason": "面向入门的概念解释。"},
        {"q": "{concept}和日常说法里的意思一样吗？", "reason": "要求澄清概念含义。"},
        {"q": "请给{concept}下一个简明定义", "reason": "明确要求定义。"},
        {"q": "{alias}中提到的{concept}该怎么理解？", "reason": "核心仍是解释概念，不是分析全文论证。", "hard": True},
        {"q": "{concept}为什么是马克思主义里的基本概念？", "reason": "偏概念地位的基础说明，未要求展开文本分析。", "hard": True},
    ],
    "comparison": [
        {"q": "{concept_a}和{concept_b}有什么区别？", "reason": "明确比较两个概念。"},
        {"q": "比较{title_a}和{title_b}的核心观点", "reason": "明确比较两部文本。"},
        {"q": "{title_a}里的{concept_a}与{title_b}里的{concept_b}有什么异同？", "reason": "比较不同文本中的理论概念。"},
        {"q": "{concept_a}是不是等同于{concept_b}？", "reason": "判断两个概念是否等同，属于比较。"},
        {"q": "早期马克思和后期马克思对{concept}的理解一致吗？", "reason": "隐含阶段比较，属于 comparison。", "hard": True},
        {"q": "马克思和恩格斯在{concept}问题上的侧重点有何不同？", "reason": "比较两位作者观点差异。"},
        {"q": "{concept_a}、{concept_b}和{concept}三者是什么关系？", "reason": "多概念关系比较。"},
        {"q": "{title_a}与{title_b}都谈到{concept}，差别在哪里？", "reason": "两文本围绕同一主题对照。"},
        {"q": "不要分别介绍，直接说{concept_a}和{concept_b}的不同", "reason": "用户明确要求差异而非单独解释。", "hard": True},
        {"q": "{concept}在哲学文本和政治经济学文本中的含义一样吗？", "reason": "跨语境比较同一概念。", "hard": True},
    ],
    "deep_analysis": [
        {"q": "结合当代平台经济分析{concept}", "reason": "要求把理论用于当代问题分析。"},
        {"q": "写一篇关于{concept}现实意义的论文提纲", "reason": "论文写作/现实意义任务，属于 deep_analysis。"},
        {"q": "从马克思主义视角分析人工智能时代的{concept}", "reason": "当代技术议题与理论综合。"},
        {"q": "请系统梳理马克思关于{concept}的理论发展线索", "reason": "跨文本系统梳理，属于深度综合。"},
        {"q": "围绕{title}和现实社会治理写一个研究框架", "reason": "文本与现实议题结合，属于 deep_analysis。"},
        {"q": "用{concept}理论分析当代劳动关系变化", "reason": "理论应用于当代现实。"},
        {"q": "请做一个{concept}研究综述，列出主要问题意识", "reason": "研究综述任务。"},
        {"q": "从{title}出发分析现代资本主义的新变化", "reason": "经典文本与现代资本主义综合分析。"},
        {"q": "如果写毕业论文研究{concept}，可以怎么设计章节？", "reason": "论文设计任务，属于 deep_analysis。"},
        {"q": "结合多个原典分析{concept}的当代价值", "reason": "跨原典且有当代价值分析。", "hard": True},
    ],
    "theory_analysis": [
        {"q": "马克思如何论述{concept}？", "reason": "询问理论论述方式，属于理论分析。"},
        {"q": "{title}的核心论证是什么？", "reason": "要求分析文本论证。"},
        {"q": "{title}主要批判了什么？", "reason": "分析文本批判对象和理论观点。"},
        {"q": "为什么说{concept}是马克思主义的重要范畴？", "reason": "要求理论地位和论证说明。"},
        {"q": "{alias}的理论贡献体现在哪里？", "reason": "分析篇目理论贡献。"},
        {"q": "请分析{concept}在马克思理论体系中的地位", "reason": "理论体系内分析。"},
        {"q": "{title}中的{concept}逻辑是怎样展开的？", "reason": "分析文本中概念论证逻辑。", "hard": True},
        {"q": "马克思为什么批判关于{concept}的错误理解？", "reason": "理论批判分析。"},
        {"q": "如何理解{title}的历史背景和理论意义？", "reason": "围绕具体文本做理论分析。"},
        {"q": "{concept}在{title}中起什么作用？", "reason": "分析概念在文本中的理论功能。", "hard": True},
    ],
    "rag_answer": [
        {"q": "介绍一下马克思主义的基本原理", "reason": "普通背景介绍，没有强定位或分析任务。"},
        {"q": "马克思有哪些主要著作？", "reason": "泛化背景问答，非具体文献定位。"},
        {"q": "给我一个学习马克思主义的入门顺序", "reason": "学习建议型问题。"},
        {"q": "马克思和恩格斯是什么关系？", "reason": "普通背景问答。"},
        {"q": "科学社会主义大概讲什么？", "reason": "宽泛介绍，未进入特定概念定义边界。", "hard": True},
        {"q": "读{title}之前需要了解哪些背景？", "reason": "阅读背景建议，不是分析该文本论证。", "hard": True},
        {"q": "帮我整理一个马克思主义入门书单", "reason": "学习资源整理，属于默认问答。"},
        {"q": "19世纪欧洲工人运动有什么基本背景？", "reason": "普通历史背景问答。"},
        {"q": "{out_of_domain}", "reason": "轻度离题/通用问题，归默认 rag_answer。"},
        {"q": "如果完全零基础，怎么开始读马克思？", "reason": "泛学习路径建议。"},
        {"q": "学习{concept}之前需要补哪些基础知识？", "reason": "学习路径和背景建议，未要求概念定义或理论分析。", "hard": True},
        {"q": "{title}适合初学者直接读吗？", "reason": "阅读建议型问题，归默认问答。"},
        {"q": "能不能给我安排一个围绕{concept}的阅读计划？", "reason": "学习计划请求，不是具体理论分析。"},
        {"q": "读{alias}时应该注意哪些背景材料？", "reason": "阅读背景建议，不是文献定位。", "hard": True},
        {"q": "马克思主义哲学和政治经济学怎么衔接起来学？", "reason": "泛学习方法问题。"},
        {"q": "我只想先了解{concept}相关主题，有哪些入门问题？", "reason": "入门问题整理，未要求定义或分析。", "hard": True},
        {"q": "{title}在马克思主义课程里通常放在哪一部分？", "reason": "课程学习背景问题，而非查卷册位置。", "hard": True},
        {"q": "请给本科生介绍一下{title}的阅读背景", "reason": "教学/阅读背景介绍，归默认问答。"},
        {"q": "围绕{concept}做课堂讨论，可以设置哪些问题？", "reason": "课堂讨论设计，属于泛任务。"},
        {"q": "马克思主义研究为什么经常提到{concept}？", "reason": "宽泛背景说明，未进入严格理论分析。", "hard": True},
        {"q": "请把{title}放进马克思主义发展史里简单介绍", "reason": "普通背景介绍，不要求具体文本论证。", "hard": True},
        {"q": "理解{alias}需要知道哪些历史背景？", "reason": "阅读背景型默认问答。"},
        {"q": "如果准备读书会，{title}可以怎么安排讨论顺序？", "reason": "读书会组织建议。"},
        {"q": "围绕{concept}有哪些常见误解需要先知道？", "reason": "入门背景整理，未要求反驳论文式分析。", "hard": True},
        {"q": "请列一个从{title}延伸阅读的书目清单", "reason": "阅读资源整理。"},
        {"q": "{concept}这个主题适合和哪些经典文本一起读？", "reason": "学习建议，不是文献定位。", "hard": True},
        {"q": "我想系统学习马克思主义，有没有阶段性路线？", "reason": "泛学习路径建议。"},
        {"q": "马克思主义三个组成部分分别解决什么问题？", "reason": "宽泛知识介绍。"},
        {"q": "能不能用几句话介绍马克思主义为什么重要？", "reason": "普通背景介绍。"},
        {"q": "围绕{alias}写读书笔记，可以从哪些角度入手？", "reason": "读书笔记建议，不是深度论文分析。", "hard": True},
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
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def collect_entities(catalog: dict[str, Any]) -> dict[str, Any]:
    works = catalog.get("works") or []
    titles: list[str] = []
    aliases: list[str] = []
    concepts: list[str] = []
    quotes: list[str] = []
    work_entities: list[dict[str, list[str]]] = []
    for work in works:
        work_titles = [str(work.get("title"))] if work.get("title") else []
        work_aliases = [str(a) for a in work.get("aliases") or [] if a]
        work_concepts = [str(c) for c in work.get("concepts") or [] if c]
        work_quotes = [str(q) for q in work.get("quotes") or [] if q]
        titles.extend(work_titles)
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
        "concepts": unique_nonempty([
            c for c in concepts
            if c not in CATALOG_CONCEPT_BLACKLIST and 2 <= len(c) <= 12
        ] + CURATED_CONCEPTS),
        "curated_concepts": unique_nonempty(CURATED_CONCEPTS),
        "quotes": unique_nonempty(quotes + FALLBACK_QUOTES),
        "works": work_entities,
    }


def target_counts(total: int) -> dict[str, int]:
    base = total // len(INTENTS)
    remainder = total % len(INTENTS)
    return {intent: base + (1 if i < remainder else 0) for i, intent in enumerate(INTENTS)}


def split_count(intent_index: int, total_for_intent: int) -> dict[str, int]:
    # For total=2000: validation/test sums become exactly 200 each.
    val_test = 29 if intent_index < 4 else 28
    return {"train": total_for_intent - 2 * val_test, "validation": val_test, "test": val_test}


def choose_values(rng: random.Random, entities: dict[str, Any]) -> dict[str, str]:
    work = rng.choice(entities["works"]) if entities["works"] else {}
    work_titles = work.get("titles") or entities["titles"]
    work_aliases = work.get("aliases") or entities["aliases"]
    work_concepts = [
        c for c in (work.get("concepts") or [])
        if c not in CATALOG_CONCEPT_BLACKLIST and 2 <= len(c) <= 12
    ] or entities["concepts"]
    work_quotes = work.get("quotes") or entities["quotes"]
    concept_pool = entities["curated_concepts"] if rng.random() < 0.78 else work_concepts
    if len(concept_pool) < 2:
        concept_pool = entities["concepts"]
    concept_a, concept_b = rng.sample(concept_pool, 2)
    title_a, title_b = rng.sample(entities["titles"], 2)
    return {
        "title": rng.choice(work_titles),
        "alias": rng.choice(work_aliases),
        "concept": rng.choice(concept_pool),
        "quote": rng.choice(work_quotes),
        "concept_a": concept_a,
        "concept_b": concept_b,
        "title_a": title_a,
        "title_b": title_b,
        "out_of_domain": rng.choice(OUT_OF_DOMAIN),
    }


def select_template(intent: str, split: str, rng: random.Random) -> dict[str, Any]:
    templates = TEMPLATES[intent]
    hard_templates = [t for t in templates if t.get("hard")]
    target_hard_rate = 0.38 if split in {"validation", "test"} else 0.28
    if hard_templates and rng.random() < target_hard_rate:
        return rng.choice(hard_templates)
    return rng.choice(templates)


def make_record(
    rec_id: int,
    split: str,
    intent: str,
    rng: random.Random,
    entities: dict[str, Any],
) -> dict[str, Any]:
    template = select_template(intent, split, rng)
    values = choose_values(rng, entities)
    boundary_case = bool(template.get("hard"))
    confidence = round(rng.uniform(0.74, 0.86) if boundary_case else rng.uniform(0.88, 0.98), 2)
    return {
        "id": f"ai_intent_{rec_id:04d}",
        "query": template["q"].format(**values),
        "intent": intent,
        "question_type": intent,
        "split": split,
        "source": "ai_assisted_label_v1",
        "confidence": confidence,
        "needs_human_review": boundary_case and confidence < 0.82,
        "boundary_case": boundary_case,
        "label_reason": template["reason"],
        "difficulty": "hard" if boundary_case else rng.choice(DIFFICULTIES),
        "discipline": rng.choice(DISCIPLINES),
        "entities": {
            key: values[key]
            for key in ["title", "alias", "concept", "quote", "concept_a", "concept_b", "title_a", "title_b"]
        },
    }


def build_dataset(total: int, seed: int, catalog_path: Path) -> list[dict[str, Any]]:
    if total != 2000:
        raise ValueError("This builder enforces total=2000 to keep exact 1600/200/200 splits.")
    rng = random.Random(seed)
    entities = collect_entities(load_work_catalog(catalog_path))
    totals = target_counts(total)
    records: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    rec_id = 1
    for intent_index, intent in enumerate(INTENTS):
        for split, count in split_count(intent_index, totals[intent]).items():
            made = 0
            attempts = 0
            while made < count:
                attempts += 1
                if attempts > count * 200:
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


def write_label_guide(path: Path) -> None:
    lines = [
        "# MarxOS Intent Labeling Guide",
        "",
        "This guide defines the standard used by `ai_assisted_label_v1`.",
        "These labels are AI-assisted silver labels, not human gold labels.",
        "",
    ]
    for intent in INTENTS:
        item = LABEL_GUIDE[intent]
        lines.extend([
            f"## {intent}",
            "",
            f"Definition: {item['definition']}",
            "",
            "Positive signals: " + "；".join(item["positive_signals"]),
            "",
            f"Boundary rule: {item['boundary']}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "splits": dict(Counter(r["split"] for r in records)),
        "intents": dict(Counter(r["intent"] for r in records)),
        "by_split_intent": {
            split: dict(Counter(r["intent"] for r in records if r["split"] == split))
            for split in ["train", "validation", "test"]
        },
        "boundary_cases": sum(1 for r in records if r["boundary_case"]),
        "needs_human_review": sum(1 for r in records if r["needs_human_review"]),
        "avg_confidence": round(sum(r["confidence"] for r in records) / len(records), 4),
        "source": "ai_assisted_label_v1",
        "note": "AI-assisted silver labels with explicit label reasons; keep separate from human gold labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2000 AI-assisted MarxOS intent labels")
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=ROOT / "rag" / "work_catalog.json")
    args = parser.parse_args()

    records = build_dataset(args.total, args.seed, args.catalog)
    splits = {
        "train": [r for r in records if r["split"] == "train"],
        "validation": [r for r in records if r["split"] == "validation"],
        "test": [r for r in records if r["split"] == "test"],
    }
    write_json(args.output_dir / "intent_dataset_ai_assisted_2000.json", records)
    for split, split_records in splits.items():
        write_json(args.output_dir / f"intent_{split}.json", split_records)
        write_jsonl(args.output_dir / f"intent_{split}.jsonl", split_records)
    write_json(args.output_dir / "labeling_guide.json", LABEL_GUIDE)
    write_label_guide(args.output_dir / "LABELING_GUIDE.md")
    write_json(args.output_dir / "summary.json", summarize(records))
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
