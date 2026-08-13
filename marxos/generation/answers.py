import re


def answer_unsupported_claim(query, rules, normalize_for_match):
    normalized_query = normalize_for_match(query)
    for rule in rules:
        if all(normalize_for_match(token) in normalized_query for token in rule["tokens"]):
            return rule["answer"]
    return ""


def answer_insufficient_material(query, constraints=None):
    """Deterministic refusal when retrieval produced no documents at all.

    The LLM must never see an empty context: without evidence cards there is
    nothing verifiable to answer from, and prose fabrication would slip past
    the citation-line audit.
    """
    title = ((constraints or {}).get("title") or "").strip()
    if title:
        return (
            f"当前语料库未检索到《{title}》的相关正文页段，因此本轮不输出跨篇替代性引文。"
            "请先确认该篇目在本地库中的页段映射或OCR文本，或换一种问法。"
        )
    return (
        "当前语料库未检索到与该问题直接相关的原文材料，无法提供有出处的回答。"
        "建议换一种问法，或指定具体著作、篇目后再提问。"
    )


def is_view_list_query(query, normalize_for_match):
    normalized = normalize_for_match(query)
    if not normalized:
        return False
    list_markers = ["列出", "概括", "归纳", "梳理", "观点", "主张", "看法"]
    return any(normalize_for_match(marker) in normalized for marker in list_markers)


def is_original_excerpt_list_query(query, normalize_for_match):
    normalized = normalize_for_match(query)
    if not normalized:
        return False
    list_markers = ["列出", "摘录", "摘出", "整理", "给出", "找出", "罗列"]
    original_markers = ["原文", "论述", "引文", "段落", "材料", "语录", "文献", "原著"]
    return (
        any(normalize_for_match(marker) in normalized for marker in list_markers)
        and any(normalize_for_match(marker) in normalized for marker in original_markers)
    )


def requested_list_limit(query, default=8, max_limit=12):
    match = re.search(r"([0-9０-９]{1,2})\s*[条段则个]", str(query or ""))
    if not match:
        return default
    raw = match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, min(value, max_limit))


def is_topic_view_list_query(query, constraints, normalize_for_match):
    return bool(constraints.get("topic_id")) and is_view_list_query(query, normalize_for_match)


def clean_excerpt_for_display(text, clean_text, article=""):
    text = clean_text(text, "")
    article = clean_text(article, "")
    if not text:
        return ""

    text = text.replace("...", "").replace("……", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[。；，、：:]+", "", text)
    if article:
        text = re.sub(rf"^{re.escape(article)}[呢，。；：:\s]*", "", text)
    text = re.sub(r"[A-Za-z]+", "", text)
    text = text.replace("f田农", "农田").replace("z把", "把").replace("z ", "")
    text = re.split(r"编者注|——+编者注|注：|\(\d+\)|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩", text, maxsplit=1)[0]
    text = re.split(r"共产党宣言这段话在|恩格斯在[０-９0-9１８９]{4}年", text, maxsplit=1)[0]
    text = re.sub(r"([。！？；]){2,}", r"\1", text)
    text = re.sub(r"([，、；：]){2,}", r"\1", text)
    return text.strip("。；，、：: ")


def best_excerpt_span(text, markers, clean_text, normalize_for_match, max_len=88):
    cleaned = clean_excerpt_for_display(text, clean_text)
    if not cleaned:
        return ""

    clauses = [
        chunk.strip("。；，、：: ")
        for chunk in re.split(r"[。！？；]", cleaned)
        if chunk.strip("。；，、：: ")
    ]
    if not clauses:
        return cleaned[:max_len]

    normalized_markers = [normalize_for_match(marker) for marker in markers if normalize_for_match(marker)]
    scored = []
    for index, clause in enumerate(clauses):
        norm = normalize_for_match(clause)
        score = sum(1 for marker in normalized_markers if marker in norm)
        if len(clause) < 12:
            score -= 1
        scored.append((score, -index, clause))
    scored.sort(reverse=True)
    chosen = scored[0][2] if scored else clauses[0]

    if len(chosen) < 28:
        clause_index = clauses.index(chosen)
        if clause_index + 1 < len(clauses):
            chosen = f"{chosen}，{clauses[clause_index + 1]}"

    return chosen[:max_len].rstrip("，、；： ")


def summarize_peasant_cooperative_viewpoint(text, normalize_for_match):
    normalized = normalize_for_match(text)
    if "合作社" in normalized and "共同耕种" in normalized:
        return "农业工人只有把土地从大地产和封建占有中解放出来，转为社会财产并实行合作社共同耕种，才能真正摆脱贫困。"
    if "土地结合起来" in normalized and "大规模经营" in normalized:
        return "分散的小块土地应结合起来实行较大规模经营，并通过合作社方式重新组织生产。"
    if "农民合作社" in normalized and ("资金" in normalized or "机会" in normalized):
        return "国家或社会应为农民合作社提供土地、资金和转向副业的机会，帮助其完成过渡。"
    if "社会帮助" in normalized or "示范" in normalized:
        return "对小农不能采取暴力剥夺，而应通过示范和社会帮助引导其逐步走向合作化。"
    if "暴力去剥夺小农" in normalized or ("暴力" in normalized and "小农" in normalized):
        return "无产阶级政党不应以暴力剥夺小农，而应争取他们自愿走向新的合作经营形式。"
    if "小农" in normalized and "土地结合起来" in normalized:
        return "小农经济的出路不在维持分散经营，而在把分散土地结合起来走合作经营道路。"
    if "公有土地" in normalized or ("小农" in normalized and "饲料" in normalized):
        return "必须重组支撑小农生产的土地条件，否则小农经济会持续陷入贫困和衰败。"
    if "重担下解放出来" in normalized or "债务" in normalized:
        return "单靠减轻债务和保全小块土地并不能真正解放农民，关键是为其转向新的合作经营形式创造条件。"
    if "农村无产者" in normalized or "最低工资" in normalized:
        return "在解决小农问题的同时，还应把农村无产者纳入最低工资和合作经营的政策安排。"
    return ""


def format_topic_viewpoint(item, constraints, clean_text, normalize_for_match):
    article = clean_text(item.get("article") or item.get("section"), "")
    topic_id = constraints.get("topic_id") or ""
    topic_markers = list(constraints.get("topic_markers") or [])
    excerpt = clean_excerpt_for_display(item.get("excerpt"), clean_text, article=article)
    if not excerpt:
        return ""

    if topic_id == "peasant_cooperative":
        summary = summarize_peasant_cooperative_viewpoint(excerpt, normalize_for_match)
        if summary:
            return summary

    best = best_excerpt_span(excerpt, topic_markers, clean_text, normalize_for_match, max_len=88)
    if not best:
        return ""
    if best.endswith(("。", "！", "？")):
        return best
    return f"{best}。"


def strict_title_answer_evidence(query, constraints, evidence, active_concept_terms, clean_text, normalize_for_match, limit=8):
    title_norm = normalize_for_match(constraints.get("title") or "")
    focus_terms = [normalize_for_match(term) for term in active_concept_terms(query) if normalize_for_match(term)]
    preferred_sources = {
        item.get("source")
        for item in constraints.get("entries") or []
        if item.get("priority") == 1 and item.get("source")
    }
    ranked = []
    for item in evidence or []:
        article = clean_text(item.get("article") or item.get("section"), "")
        article_norm = normalize_for_match(article)
        excerpt = clean_excerpt_for_display(item.get("excerpt"), clean_text, article=article)
        excerpt_norm = normalize_for_match(excerpt)
        if not excerpt_norm:
            continue
        score = 0
        if title_norm and title_norm in article_norm:
            score += 25
        if item.get("source") in preferred_sources:
            score += 18
        if item.get("printed_page") is not None:
            score += 8
        if focus_terms:
            score += sum(24 for term in focus_terms if term in excerpt_norm)
            if not any(term in excerpt_norm for term in focus_terms):
                score -= 12
        if any(marker in excerpt_norm for marker in [normalize_for_match("序言"), normalize_for_match("导言"), normalize_for_match("注释")]):
            score -= 16
        if normalize_for_match("编者注") in excerpt_norm:
            score -= 24
        if item.get("printed_page") is not None and item.get("printed_page") <= 30 and focus_terms:
            score -= 10
        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for score, item in ranked if score > 0]
    return selected[:limit]


def build_strict_title_view_list_answer(query, constraints, evidence, active_concept_terms, clean_text, normalize_for_match, limit=8):
    direct = strict_title_answer_evidence(
        query,
        constraints,
        evidence,
        active_concept_terms,
        clean_text,
        normalize_for_match,
        limit=limit,
    )
    if len(direct) < 2:
        return ""

    title = constraints.get("title") or "该文"
    focus_terms = active_concept_terms(query)
    lines = [f"根据当前检索到的《{title}》原著材料，可以先归纳出以下要点：", ""]

    for index, item in enumerate(direct[:limit], start=1):
        viewpoint = format_topic_viewpoint(item, {"topic_markers": focus_terms}, clean_text, normalize_for_match)
        if not viewpoint:
            continue
        citation = item.get("citation") or item.get("sentence_citation") or ""
        lines.append(f"{index}. 观点：{viewpoint}")
        if citation:
            lines.append(f"   {citation}")

    return "\n".join(lines).rstrip()


def topic_direct_evidence(evidence, constraints, clean_text, normalize_for_match):
    markers = [normalize_for_match(marker) for marker in (constraints.get("topic_markers") or []) if normalize_for_match(marker)]
    topic_id = constraints.get("topic_id") or ""
    preferred_title_weights = {}
    direct_excerpt_markers = []
    if topic_id == "peasant_cooperative":
        preferred_title_weights = {
            normalize_for_match("法德农民问题"): 20,
            normalize_for_match("德国农民战争"): 8,
            normalize_for_match("对农村居民土地的剥夺"): 6,
        }
        direct_excerpt_markers = [
            normalize_for_match(marker)
            for marker in [
                "合作社",
                "共同耕种",
                "小农",
                "大土地",
                "农村无产者",
                "土地纲领",
                "土地所有制",
                "社会帮助",
                "示范",
                "暴力去剥夺小农",
            ]
        ]
    direct = []
    for item in evidence or []:
        article = normalize_for_match(clean_text(item.get("article") or item.get("section"), ""))
        excerpt = normalize_for_match(clean_text(item.get("excerpt"), ""))
        score = 0
        direct_hits = 0
        for marker in markers:
            if marker in article:
                score += 3
            if marker in excerpt:
                score += 2
        for marker, bonus in preferred_title_weights.items():
            if marker and marker in article:
                score += bonus
        if direct_excerpt_markers:
            direct_hits = sum(1 for marker in direct_excerpt_markers if marker and marker in excerpt)
            score += direct_hits * 6
            if direct_hits == 0:
                score -= 18
        if topic_id == "peasant_cooperative":
            is_preferred_title = any(marker and marker in article for marker in preferred_title_weights)
            if direct_hits == 0:
                continue
            if not is_preferred_title and direct_hits < 2:
                continue
            if normalize_for_match("法德农民问题") in article and direct_hits >= 1:
                score += 10
            if normalize_for_match("同样") in excerpt:
                score -= 6
        if excerpt.startswith(normalize_for_match("法德农民问题呢")):
            score -= 15
        if normalize_for_match("现在我们来谈一谈较大的农民") in excerpt:
            score -= 8
        if score > 0:
            direct.append((score, item))
    direct.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in direct]


def topic_answer_evidence(evidence, constraints, clean_text, normalize_for_match, limit=10):
    direct = topic_direct_evidence(evidence, constraints, clean_text, normalize_for_match)
    topic_id = constraints.get("topic_id") or ""
    if topic_id == "peasant_cooperative":
        positive_markers = [
            normalize_for_match(marker)
            for marker in [
                "合作社",
                "共同耕种",
                "小农",
                "最低工资",
                "土地纲领",
                "农村无产者",
                "大土地",
                "社会帮助",
                "示范",
                "房产和田产",
                "土地结合起来",
                "暴力去剥夺小农",
            ]
        ]
        negative_markers = [
            normalize_for_match(marker)
            for marker in [
                "只是大概地研究这一问题",
                "现在我们来谈一谈较大的农民",
                "历史上德国人民",
                "论述打算通过对这场斗争的历史进程",
                "由于存在着地方分权",
            ]
        ]
        strong = []
        fallback = []
        for item in direct:
            excerpt_norm = normalize_for_match(clean_text(item.get("excerpt"), ""))
            positive_hits = sum(1 for marker in positive_markers if marker and marker in excerpt_norm)
            negative_hit = any(marker and marker in excerpt_norm for marker in negative_markers)
            if negative_hit:
                continue
            if positive_hits >= 1:
                strong.append(item)
            else:
                fallback.append(item)
        selected = strong + fallback
        return selected[:limit]
    return direct[:limit]


def topic_original_excerpt_evidence(evidence, constraints, clean_text, normalize_for_match, limit=10):
    markers = [
        normalize_for_match(marker)
        for marker in (constraints.get("topic_markers") or [])
        if normalize_for_match(marker)
    ]
    allowed_titles = constraints.get("allowed_titles") or set()
    selected = []
    seen = set()
    for item in evidence or []:
        article = normalize_for_match(clean_text(item.get("article") or item.get("section"), ""))
        excerpt = normalize_for_match(clean_text(item.get("excerpt"), ""))
        title_hit = any(title and title in article for title in allowed_titles)
        marker_hits = sum(1 for marker in markers if marker and (marker in article or marker in excerpt))
        if not title_hit and marker_hits <= 0:
            continue
        key = (
            item.get("source"),
            item.get("printed_page") or item.get("citation_page") or item.get("pdf_page"),
            excerpt[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append((title_hit, marker_hits, item))
    selected.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in selected[:limit]]


def build_topic_view_list_answer(query, constraints, evidence, clean_text, normalize_for_match, limit=8):
    if is_original_excerpt_list_query(query, normalize_for_match):
        limit = min(limit, requested_list_limit(query, default=limit, max_limit=12))
        direct = topic_original_excerpt_evidence(evidence, constraints, clean_text, normalize_for_match, limit=limit)
        if not direct:
            return ""

        topic_label = constraints.get("topic_title") or "该专题"
        lines = [f"根据当前检索到的原著材料，先列出{topic_label}相关原文摘录：", ""]
        for index, item in enumerate(direct[:limit], start=1):
            excerpt = clean_excerpt_for_display(item.get("excerpt"), clean_text, article=item.get("article") or "")
            if not excerpt:
                continue
            citation = item.get("detailed_citation") or item.get("citation") or item.get("sentence_citation") or ""
            lines.append(f"{index}. 原文：{excerpt}")
            if citation:
                lines.append(f"   出处：{citation}")
        if len(direct) < limit:
            lines.append("")
            lines.append(f"说明：当前证据只支持列出 {len(direct)} 条较直接相关的原文摘录；要凑满 {limit} 条需要继续扩大专题语料或提高召回数量。")
        return "\n".join(lines).rstrip()

    direct = topic_answer_evidence(evidence, constraints, clean_text, normalize_for_match, limit=limit)
    if len(direct) < 2:
        return ""

    lines = []
    topic_id = constraints.get("topic_id") or ""
    topic_label = constraints.get("topic_title") or "该专题"
    if topic_id == "peasant_cooperative":
        lines.append(f"根据当前检索到的原著材料，可以先列出{topic_label}中更直接相关的要点：")
    else:
        lines.append(f"根据当前检索到的原著材料，可以先归纳出{topic_label}中的以下要点：")
    lines.append("")

    for index, item in enumerate(direct[:limit], start=1):
        viewpoint = format_topic_viewpoint(item, constraints, clean_text, normalize_for_match)
        if not viewpoint:
            continue
        citation = item.get("citation") or item.get("sentence_citation") or ""
        lines.append(f"{index}. 观点：{viewpoint}")
        if citation:
            lines.append(f"   {citation}")

    lines.append("")
    lines.append("说明：以上为基于当前专题证据的归纳整理，若要继续扩充到更系统的十段专题摘录，还需要继续补入俄国农村公社、土地制度和农民问题相关文本。")
    return "\n".join(lines)
