from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


LOG_DIR = Path("logs")
REPORT_JSON = LOG_DIR / "hybrid_retrieval_eval.json"
REPORT_MD = LOG_DIR / "hybrid_retrieval_eval.md"


@dataclass(frozen=True)
class HybridProbe:
    label: str
    query: str
    expected_sources: tuple[str, ...] = ()
    expected_article_terms: tuple[str, ...] = ()
    expected_content_terms: tuple[str, ...] = ()
    note: str = ""


PROBES = [
    HybridProbe(
        label="surplus_appropriation_phrase",
        query="请解释“无偿占有的部分”这个表述在马克思那里指什么。",
        expected_sources=("mea05.pdf", "mes02.pdf"),
        expected_article_terms=("资本论", "工资、价格和利润"),
        expected_content_terms=("剩余价值", "无偿"),
        note="口语化问法，不直接说剩余价值。",
    ),
    HybridProbe(
        label="state_machine_phrase",
        query="为什么说旧国家机器不能直接拿来用？",
        expected_sources=("mea03.pdf", "mes04.pdf"),
        expected_article_terms=("法兰西内战", "哥达纲领批判", "家庭、私有制和国家的起源"),
        expected_content_terms=("国家", "机器"),
        note="固定政治表述，稀疏词锚点强。",
    ),
    HybridProbe(
        label="human_essence_abstract_thing",
        query="人的本质为什么不是单个人所固有的抽象物？",
        expected_sources=("mea01.pdf", "mes01.pdf"),
        expected_article_terms=("关于费尔巴哈的提纲",),
        expected_content_terms=("人的本质", "社会关系"),
        note="直击经典句式，但不完整复述。",
    ),
    HybridProbe(
        label="religion_opium_context",
        query="宗教为什么会被说成人民的鸦片？",
        expected_sources=("mea01.pdf", "mes01.pdf"),
        expected_article_terms=("黑格尔法哲学批判导言",),
        expected_content_terms=("宗教", "鸦片"),
        note="短语命中应该对 sparse 友好。",
    ),
    HybridProbe(
        label="small_peasant_policy",
        query="合作社问题上为什么不能简单剥夺小农？",
        expected_sources=("mea04.pdf", "mea03.pdf"),
        expected_article_terms=("法德农民问题", "论土地国有化"),
        expected_content_terms=("农民", "合作", "土地"),
        note="农业问题的口语化追问。",
    ),
    HybridProbe(
        label="commodity_fetishism_phrase",
        query="商品拜物教到底拜的是什么？",
        expected_sources=("mea05.pdf", "mes02.pdf"),
        expected_article_terms=("资本论",),
        expected_content_terms=("商品", "拜物教"),
        note="概念口语化缩写。",
    ),
    HybridProbe(
        label="labor_duality_phrase",
        query="为什么同一劳动会有两种规定？",
        expected_sources=("mea05.pdf", "mes02.pdf"),
        expected_article_terms=("资本论",),
        expected_content_terms=("劳动二重性", "具体劳动", "抽象劳动"),
        note="不直接说劳动二重性。",
    ),
    HybridProbe(
        label="class_history_formula",
        query="“至今一切社会的历史”后面那句到底在讲什么？",
        expected_sources=("mea02.pdf", "mes01.pdf"),
        expected_article_terms=("共产党宣言",),
        expected_content_terms=("阶级斗争",),
        note="半截名句触发。",
    ),
    HybridProbe(
        label="practice_truth_question",
        query="为什么真理性问题在马克思那里首先是实践问题？",
        expected_sources=("mea01.pdf", "mes01.pdf"),
        expected_article_terms=("关于费尔巴哈的提纲",),
        expected_content_terms=("实践", "真理"),
        note="术语对术语的检索。",
    ),
    HybridProbe(
        label="base_superstructure_linear",
        query="为什么不能把经济基础和上层建筑理解成单向直线决定？",
        expected_sources=("mea01.pdf", "mes01.pdf", "mes04.pdf"),
        expected_article_terms=("德意志意识形态", "家庭、私有制和国家的起源"),
        expected_content_terms=("经济基础", "上层建筑"),
        note="教学口语常见误解问法。",
    ),
    HybridProbe(
        label="working_day_struggle",
        query="工作日长短为什么不是单纯技术问题，而是斗争问题？",
        expected_sources=("mea05.pdf", "mes02.pdf"),
        expected_article_terms=("资本论",),
        expected_content_terms=("工作日", "斗争"),
        note="固定搭配明显。",
    ),
    HybridProbe(
        label="communism_not_equalism",
        query="为什么不能把共产主义理解成平均主义？",
        expected_sources=("mea03.pdf", "mes04.pdf"),
        expected_article_terms=("哥达纲领批判",),
        expected_content_terms=("按劳分配", "按需分配"),
        note="概念辨析型问题。",
    ),
]


def normalize_for_match(text: str) -> str:
    text = str(text or "")
    return re.sub(r"[《》“”\"'（）()、，。；：！？\s\-\.\·]", "", text).lower()


def doc_summary(doc) -> dict:
    metadata = dict(doc.metadata or {})
    return {
        "source": metadata.get("source"),
        "article": metadata.get("article") or metadata.get("section"),
        "match_type": metadata.get("match_type"),
        "printed_page": metadata.get("printed_page"),
        "citation_page": metadata.get("citation_page"),
        "sparse_score": metadata.get("sparse_score"),
        "preview": " ".join(str(doc.page_content or "").split())[:120],
    }


def probe_match_passed(probe: HybridProbe, docs: list, top_n: int = 1) -> tuple[bool, str]:
    if not docs:
        return False, "no docs"

    checked = docs[:top_n]
    article_terms = [normalize_for_match(term) for term in probe.expected_article_terms if term]
    content_terms = [normalize_for_match(term) for term in probe.expected_content_terms if term]

    for doc in checked:
        metadata = dict(doc.metadata or {})
        source = metadata.get("source")
        article = normalize_for_match(metadata.get("article") or metadata.get("section") or "")
        content = normalize_for_match(doc.page_content or "")

        if probe.expected_sources and source not in probe.expected_sources:
            continue
        if article_terms and not any(term and term in article for term in article_terms):
            continue
        if content_terms and not all(term and term in content for term in content_terms):
            continue
        return True, f"matched {source}"

    return False, "top docs did not satisfy expected source/article/content constraints"


def run_mode(mode: str, k: int = 3) -> dict:
    os.environ["MARXOS_HYBRID_RETRIEVAL"] = mode
    app.RUNTIME.vectorstore_instance = None
    app.RUNTIME.paragraph_vectorstore_instance = None
    app.RUNTIME.embeddings_instance = None

    db = app.load_vectorstore()
    results = []

    for probe in PROBES:
        docs = app.retrieve_documents(probe.query, db, k=k)
        top1_ok, top1_reason = probe_match_passed(probe, docs, top_n=1)
        top3_ok, top3_reason = probe_match_passed(probe, docs, top_n=min(3, len(docs)))
        sparse_hits = sum(1 for doc in docs if (doc.metadata or {}).get("match_type") == "sparse_candidate")
        results.append(
            {
                "label": probe.label,
                "query": probe.query,
                "note": probe.note,
                "top1_ok": top1_ok,
                "top1_reason": top1_reason,
                "top3_ok": top3_ok,
                "top3_reason": top3_reason,
                "sparse_hits": sparse_hits,
                "docs": [doc_summary(doc) for doc in docs],
            }
        )

    summary = {
        "mode": mode,
        "probes": len(PROBES),
        "top1_passed": sum(1 for item in results if item["top1_ok"]),
        "top3_passed": sum(1 for item in results if item["top3_ok"]),
        "sparse_hit_probes": sum(1 for item in results if item["sparse_hits"] > 0),
    }
    return {"summary": summary, "results": results}


def compare_results(off: dict, on: dict) -> dict:
    by_label_off = {item["label"]: item for item in off["results"]}
    by_label_on = {item["label"]: item for item in on["results"]}
    changed = []
    improved = []
    regressed = []

    for probe in PROBES:
        left = by_label_off[probe.label]
        right = by_label_on[probe.label]
        delta = {
            "label": probe.label,
            "query": probe.query,
            "top1_off": left["top1_ok"],
            "top1_on": right["top1_ok"],
            "top3_off": left["top3_ok"],
            "top3_on": right["top3_ok"],
            "sparse_hits_off": left["sparse_hits"],
            "sparse_hits_on": right["sparse_hits"],
            "top1_source_off": (left["docs"][0]["source"] if left["docs"] else None),
            "top1_source_on": (right["docs"][0]["source"] if right["docs"] else None),
            "top1_match_type_off": (left["docs"][0]["match_type"] if left["docs"] else None),
            "top1_match_type_on": (right["docs"][0]["match_type"] if right["docs"] else None),
        }
        if delta["top1_off"] != delta["top1_on"] or delta["top3_off"] != delta["top3_on"] or delta["top1_source_off"] != delta["top1_source_on"]:
            changed.append(delta)
        if (not delta["top1_off"] and delta["top1_on"]) or (not delta["top3_off"] and delta["top3_on"]):
            improved.append(delta)
        if (delta["top1_off"] and not delta["top1_on"]) or (delta["top3_off"] and not delta["top3_on"]):
            regressed.append(delta)

    return {
        "off": off["summary"],
        "on": on["summary"],
        "changed": changed,
        "improved": improved,
        "regressed": regressed,
    }


def write_reports(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = payload["comparison"]
    lines = [
        "# Hybrid Retrieval Eval",
        "",
        f"- probes: {comparison['off']['probes']}",
        f"- top1 off/on: {comparison['off']['top1_passed']}/{comparison['on']['top1_passed']}",
        f"- top3 off/on: {comparison['off']['top3_passed']}/{comparison['on']['top3_passed']}",
        f"- sparse-hit probes off/on: {comparison['off']['sparse_hit_probes']}/{comparison['on']['sparse_hit_probes']}",
        f"- changed probes: {len(comparison['changed'])}",
        f"- improved probes: {len(comparison['improved'])}",
        f"- regressed probes: {len(comparison['regressed'])}",
        "",
        "## Changed Probes",
        "",
    ]

    if not comparison["changed"]:
        lines.append("- none")
    else:
        for item in comparison["changed"]:
            lines.append(
                f"- `{item['label']}` | top1 {item['top1_off']} -> {item['top1_on']} | "
                f"top3 {item['top3_off']} -> {item['top3_on']} | "
                f"source {item['top1_source_off']} -> {item['top1_source_on']} | "
                f"match_type {item['top1_match_type_off']} -> {item['top1_match_type_on']}"
            )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    off = run_mode("0")
    on = run_mode("1")
    comparison = compare_results(off, on)
    payload = {
        "off": off,
        "on": on,
        "comparison": comparison,
    }
    write_reports(payload)

    print("===== HYBRID RETRIEVAL EVAL =====")
    print(
        f"top1 off/on: {comparison['off']['top1_passed']}/{comparison['on']['top1_passed']}"
    )
    print(
        f"top3 off/on: {comparison['off']['top3_passed']}/{comparison['on']['top3_passed']}"
    )
    print(
        "sparse-hit probes off/on: "
        f"{comparison['off']['sparse_hit_probes']}/{comparison['on']['sparse_hit_probes']}"
    )
    print(f"changed probes: {len(comparison['changed'])}")
    print(f"improved probes: {len(comparison['improved'])}")
    print(f"regressed probes: {len(comparison['regressed'])}")
    print(f"report json: {REPORT_JSON}")
    print(f"report md: {REPORT_MD}")


if __name__ == "__main__":
    main()
