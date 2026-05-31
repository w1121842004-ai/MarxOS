from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
import web_app


LOG_DIR = Path("logs")
REPORT_JSON = LOG_DIR / "web_expert_eval_50.json"
REPORT_MD = LOG_DIR / "web_expert_eval_50.md"


def citation_lines(answer: str) -> list[str]:
    return app.extract_answer_citation_lines(answer or "")


def citation_work_titles(answer: str) -> list[str]:
    titles: list[str] = []
    for line in citation_lines(answer):
        titles.extend(re.findall(r"《([^》]+)》", line))
    return titles


def distinct_work_count(answer: str) -> int:
    return len({title.strip() for title in citation_work_titles(answer) if title.strip()})


def preview(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class TurnSpec:
    query: str
    label: str
    require_multi_work: bool = False
    require_citations: bool = True
    min_answer_len: int = 140
    min_evidence_count: int = 1


TURNS: list[TurnSpec] = [
    TurnSpec("请从《资本论》《政治经济学批判大纲》《工资、价格和利润》等文本出发，系统说明马克思如何理解商品二因素、劳动二重性与价值形式的关系。要求逻辑展开，不要只给定义。", "commodity_theory_overview", require_multi_work=True, min_answer_len=220),
    TurnSpec("你上面谈到劳动二重性。请进一步区分具体劳动和抽象劳动，并解释它们为什么不是两个互不相关的劳动，而是同一劳动的两种规定。", "labor_duality_followup", min_answer_len=180),
    TurnSpec("请继续追问：这种分析为什么能够成为理解价值形态和货币起源的前提？请结合前文而不是另起炉灶。", "value_form_followup", min_answer_len=180),
    TurnSpec("请专门从拜物教问题出发，说明商品拜物教与资本主义社会关系的颠倒呈现之间是什么关系，并尽量援引不止一部著作。", "fetishism_multiwork", require_multi_work=True, min_answer_len=220),
    TurnSpec("如果把这一组观点用于分析今天平台经济中的劳动关系，你会怎样在坚持马克思原意的前提下做理论转述？不要脱离原著。", "commodity_modern_application", min_answer_len=180),

    TurnSpec("请系统说明剩余价值理论的基本结构，至少涉及劳动力商品、必要劳动、剩余劳动、资本增殖，并从多部著作交叉说明。", "surplus_value_overview", require_multi_work=True, min_answer_len=220),
    TurnSpec("进一步区分绝对剩余价值与相对剩余价值，并解释两者为什么都服从资本增殖逻辑。", "absolute_relative_surplus", min_answer_len=180),
    TurnSpec("你上面的回答里如果涉及工作日，请继续追问：马克思为什么说工作日长度的界限不是单纯技术问题，而是阶级斗争问题？", "working_day_followup", min_answer_len=180),
    TurnSpec("请把剩余价值生产与协作、分工、机器大工业联系起来，说明资本主义如何在提高生产力的同时强化对劳动的支配。", "machinery_and_control", min_answer_len=220),
    TurnSpec("再追问一步：剩余价值理论和利润、平均利润、地租这些范畴之间应如何衔接？请提醒其中的中介环节。", "surplus_to_profit_bridge", min_answer_len=200),

    TurnSpec("请从《1844年经济学哲学手稿》《关于费尔巴哈的提纲》《德意志意识形态》等文本出发，说明异化、实践和人的本质三者之间的关系。", "alienation_practice_essence", require_multi_work=True, min_answer_len=220),
    TurnSpec("你刚才提到人的本质。请严谨解释“人的本质是一切社会关系的总和”这句话，尤其要避免把它理解成简单社会决定论。", "human_essence_precision", min_answer_len=200),
    TurnSpec("继续追问：马克思为什么批评费尔巴哈停留在直观唯物主义，而没有把现实的人理解为实践中的人？", "feuerbach_critique", min_answer_len=180),
    TurnSpec("请进一步说明异化劳动分析与历史唯物主义后来形成之间有何连续性，又有哪些表述方式的变化。", "alienation_to_historical_materialism", min_answer_len=200),
    TurnSpec("如果要用马恩原著回应“马克思主义是否取消人的主体性”这一质疑，你会怎样回答？", "subjectivity_reply", min_answer_len=180),

    TurnSpec("请系统阐明历史唯物主义的基本论证结构：生产力、生产关系、经济基础、上层建筑、社会形态更替之间的关系。要求基于马恩原著而不是教科书口径。", "historical_materialism_system", require_multi_work=True, min_answer_len=240),
    TurnSpec("请追问：经济基础与上层建筑关系中，为什么不能把因果关系理解成线性单向决定？", "base_superstructure_precision", min_answer_len=190),
    TurnSpec("进一步说明阶级斗争在历史发展中的地位。它和生产方式变革之间是什么关系？", "class_struggle_place", min_answer_len=180),
    TurnSpec("请把《共产党宣言》和《路易·波拿巴的雾月十八日》联系起来，说明马克思如何具体分析阶级、政治形式和国家权力。", "class_politics_bridge", require_multi_work=True, min_answer_len=220),
    TurnSpec("如果有人把历史唯物主义理解为“经济自动决定一切”，请你基于前面讨论作出反驳，并明确指出这种误解错在哪里。", "economic_determinism_rebuttal", min_answer_len=180),

    TurnSpec("请从《共产党宣言》《法兰西内战》《哥达纲领批判》等文本出发，系统说明马克思恩格斯如何理解国家的阶级性质以及无产阶级专政。", "state_and_dictatorship", require_multi_work=True, min_answer_len=240),
    TurnSpec("进一步追问：为什么巴黎公社被马克思看作无产阶级国家形式的重要线索？请说明它与旧国家机器的关系。", "commune_state_form", min_answer_len=190),
    TurnSpec("你上面如果谈到“打碎旧国家机器”，请继续澄清：这是否意味着任何组织形式都应被取消？", "smash_state_machine_precision", min_answer_len=180),
    TurnSpec("请把恩格斯关于国家起源的论述也纳入，说明国家并非永恒存在，而是历史地产生和消亡的。", "origin_and_withering", require_multi_work=True, min_answer_len=210),
    TurnSpec("再追问一步：怎样区分马克思主义关于国家消亡的观点与无政府主义立即废除国家的主张？", "state_vs_anarchism", min_answer_len=190),

    TurnSpec("请围绕农民问题和合作化问题，说明马克思恩格斯如何在坚持无产阶级立场的同时处理小农、土地和合作社。尽量援引多部文本。", "peasant_question", require_multi_work=True, min_answer_len=230),
    TurnSpec("请继续追问：为什么恩格斯强调不能对小农采取简单剥夺政策？这种立场背后的阶级分析是什么？", "small_peasant_policy", min_answer_len=180),
    TurnSpec("请把土地国有化、合作化和生产社会化之间的关系讲清楚，不要只罗列结论。", "land_and_cooperation", min_answer_len=190),
    TurnSpec("进一步说明：在农业问题上，马克思主义如何避免在原则上革命、在策略上冒进？", "agrarian_strategy", min_answer_len=180),
    TurnSpec("如果将这组观点用于理解当代乡村合作组织，应当坚持哪些理论边界，哪些地方不能随意套用？", "agrarian_modern_limits", min_answer_len=180),

    TurnSpec("请从《共产党宣言》、第一国际相关文献以及恩格斯晚年通信出发，说明马克思主义怎样理解党的作用、工人阶级组织和国际主义。", "party_and_internationalism", require_multi_work=True, min_answer_len=230),
    TurnSpec("请追问：为什么马克思主义既反对宗派主义，又反对放弃理论原则的机会主义？", "sectarianism_vs_opportunism", min_answer_len=180),
    TurnSpec("进一步说明国际主义与民族问题的关系。马克思恩格斯为什么不是抽象地谈世界主义？", "internationalism_and_nation", min_answer_len=190),
    TurnSpec("请把组织问题与阶级自我解放原则联系起来，说明为什么马克思主义反对把群众仅仅看作被动对象。", "self_emancipation_and_organization", min_answer_len=180),
    TurnSpec("如果有人把党的领导理解为对工人阶级的外在替代，请基于马恩观点反驳。", "party_substitutionism_rebuttal", min_answer_len=180),

    TurnSpec("请系统说明从资本主义到共产主义的过渡时期问题，至少涉及生产资料公有、按劳分配、按需分配、旧社会痕迹等内容。", "transition_period_system", require_multi_work=True, min_answer_len=230),
    TurnSpec("进一步区分社会主义和共产主义高级阶段，不要泛泛而谈。", "socialism_vs_communism", min_answer_len=180),
    TurnSpec("请追问：为什么《哥达纲领批判》反对在分配问题上停留于“公平分配”这一抽象口号？", "fair_distribution_critique", min_answer_len=180),
    TurnSpec("请把生产关系变革、人的全面发展和劳动形态变化联系起来，而不是只讨论分配。", "development_and_labor", min_answer_len=190),
    TurnSpec("如果把共产主义理解成平均主义，马克思主义为什么认为这是误解？", "communism_not_equalism", min_answer_len=170),

    TurnSpec("请从宗教批判、法权批判和意识形态批判的角度，说明马克思主义如何理解观念形态与现实社会关系的联系。", "ideology_religion_law", require_multi_work=True, min_answer_len=230),
    TurnSpec("请专门解释“宗教是人民的鸦片”这句话，要求交代上下文，避免断章取义。", "religion_opium_context", min_answer_len=180),
    TurnSpec("进一步说明法权为什么具有历史性和阶级性，同时又不能被理解为纯粹幻想。", "law_historicity", min_answer_len=180),
    TurnSpec("请把意识形态问题与物质生活过程联系起来，说明为什么错误观念并不是靠说服就能根本消除。", "ideology_material_life", min_answer_len=180),
    TurnSpec("如果要回应“马克思主义只讲物质、不讲价值和理想”这种批评，你会怎样回答？", "values_and_ideals_reply", min_answer_len=180),

    TurnSpec("请从方法论上说明马克思如何处理抽象上升到具体、逻辑与历史、现象与本质的关系。尽量跨多部文本说明。", "methodology_system", require_multi_work=True, min_answer_len=240),
    TurnSpec("请追问：为什么《政治经济学批判大纲》导言中的“具体之所以具体”对理解《资本论》写法特别关键？", "concrete_from_abstract", min_answer_len=190),
    TurnSpec("进一步说明“逻辑与历史统一”不能被理解为简单编年史复写。", "logic_and_history", min_answer_len=180),
    TurnSpec("请把辩证法与资本主义社会的矛盾运动联系起来，说明它不是外加的思维技巧。", "dialectics_and_capital", min_answer_len=190),
    TurnSpec("最后请对我们这 50 轮讨论做一个学术性总结：概括马克思主义理论体系内部最核心的几条方法论线索，并再次注明关键出处。", "final_synthesis", require_multi_work=True, min_answer_len=260),
]


def build_history_payload(history: list[dict]) -> list[dict]:
    return history[-12:]


def ask(port: int, query: str, history: list[dict]) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=180)
    body = json.dumps({"query": query, "history": build_history_payload(history)}, ensure_ascii=False).encode("utf-8")
    conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    payload = json.loads(res.read().decode("utf-8"))
    conn.close()
    return res.status, payload


def start_server() -> tuple[web_app.ThreadingHTTPServer, int, threading.Thread]:
    app.load_vectorstore()
    if app.paragraph_vectorstore_exists():
        app.load_paragraph_vectorstore()
    server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], thread


def analyze_turn(spec: TurnSpec, status: int, payload: dict) -> dict:
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    evidence = payload.get("evidence") or []
    citation_audit = payload.get("citation_audit") or {}
    works = sorted({title.strip() for title in citation_work_titles(answer) if title.strip()})
    flags: list[str] = []

    if status != 200:
        flags.append(f"http_{status}")
    if payload.get("error"):
        flags.append("api_error")
    if not answer.strip():
        flags.append("empty_answer")
    if len(answer.strip()) < spec.min_answer_len:
        flags.append("short_answer")
    if spec.require_citations and not citation_lines(answer):
        flags.append("no_citations")
    if len(evidence) < spec.min_evidence_count:
        flags.append("low_evidence")
    if spec.require_multi_work and len(works) < 2:
        flags.append("multi_work_not_met")
    if citation_audit and not citation_audit.get("ok", True):
        flags.append("citation_audit_failed")

    return {
        "label": spec.label,
        "query": spec.query,
        "status": status,
        "intent": payload.get("intent", ""),
        "elapsed_ms": payload.get("elapsed_ms", 0),
        "answer_len": len(answer),
        "citation_count": len(citation_lines(answer)),
        "distinct_works": works,
        "evidence_count": len(evidence),
        "topic": payload.get("topic") or {},
        "citation_audit": citation_audit,
        "flags": flags,
        "answer_preview": preview(answer, 220),
    }


def write_reports(results: list[dict], started_at: float, ended_at: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "turns": len(results),
        "ok_turns": sum(1 for item in results if not item["flags"]),
        "flagged_turns": sum(1 for item in results if item["flags"]),
        "api_error_turns": sum(1 for item in results if any(flag == "api_error" for flag in item["flags"])),
        "avg_elapsed_ms": int(sum(item.get("elapsed_ms", 0) for item in results) / max(len(results), 1)),
        "avg_citation_count": round(sum(item.get("citation_count", 0) for item in results) / max(len(results), 1), 2),
        "avg_evidence_count": round(sum(item.get("evidence_count", 0) for item in results) / max(len(results), 1), 2),
        "multi_work_turns_met": sum(
            1
            for item, spec in zip(results, TURNS)
            if (not spec.require_multi_work) or len(item.get("distinct_works") or []) >= 2
        ),
        "started_at": int(started_at),
        "ended_at": int(ended_at),
        "duration_sec": int(ended_at - started_at),
    }
    REPORT_JSON.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# MarxOS Web Expert Eval (50 turns)",
        "",
        f"- turns: {summary['turns']}",
        f"- ok_turns: {summary['ok_turns']}",
        f"- flagged_turns: {summary['flagged_turns']}",
        f"- api_error_turns: {summary['api_error_turns']}",
        f"- avg_elapsed_ms: {summary['avg_elapsed_ms']}",
        f"- avg_citation_count: {summary['avg_citation_count']}",
        f"- avg_evidence_count: {summary['avg_evidence_count']}",
        f"- duration_sec: {summary['duration_sec']}",
        "",
        "## Turn Results",
        "",
    ]
    for idx, item in enumerate(results, start=1):
        lines.append(f"### {idx}. {item['label']}")
        lines.append(f"- query: {item['query']}")
        lines.append(f"- intent: {item['intent'] or '-'}")
        lines.append(f"- elapsed_ms: {item['elapsed_ms']}")
        lines.append(f"- citation_count: {item['citation_count']}")
        lines.append(f"- evidence_count: {item['evidence_count']}")
        lines.append(f"- distinct_works: {', '.join(item['distinct_works']) if item['distinct_works'] else '-'}")
        lines.append(f"- flags: {', '.join(item['flags']) if item['flags'] else 'none'}")
        lines.append(f"- answer_preview: {item['answer_preview']}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    history: list[dict] = []
    results: list[dict] = []
    started_at = time.time()
    server, port, thread = start_server()
    try:
        for spec in TURNS:
            status, payload = ask(port, spec.query, history)
            result = analyze_turn(spec, status, payload)
            results.append(result)

            history.append({"role": "user", "text": spec.query})
            history.append(
                {
                    "role": "bot",
                    "text": payload.get("answer", ""),
                    "evidence": payload.get("evidence") or [],
                    "topic": payload.get("topic") or {},
                }
            )
    finally:
        ended_at = time.time()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        write_reports(results, started_at, ended_at)

    flagged = sum(1 for item in results if item["flags"])
    print(json.dumps({"report_json": str(REPORT_JSON), "report_md": str(REPORT_MD), "turns": len(results), "flagged_turns": flagged}, ensure_ascii=False))
    return 0 if flagged == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
