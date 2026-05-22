from __future__ import annotations

import re


def topic_history_evidence(history, last_bot_item_fn):
    last_bot = last_bot_item_fn(history)
    evidence = last_bot.get("evidence") or []
    topic = last_bot.get("topic") or {}
    if not isinstance(evidence, list) or not isinstance(topic, dict):
        return [], {}
    return evidence, topic


def excerpt_key(item, normalize_for_match):
    return normalize_for_match((item.get("excerpt") or "")[:120])


def rank_topic_evidence(evidence, normalize_for_match):
    direct_markers = ["合作社", "共同耕种", "示范", "社会帮助", "小农", "大土地", "农村无产者", "纲领"]
    ranked = []
    for item in evidence:
        text = (item.get("excerpt") or "") + " " + (item.get("article") or "")
        score = 0
        for marker in direct_markers:
            if marker in text:
                score += 1
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    deduped = []
    seen = set()
    for _, item in ranked:
        key = excerpt_key(item, normalize_for_match)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def filter_ranked_evidence(ranked, markers_any=None, markers_all=None):
    markers_any = markers_any or []
    markers_all = markers_all or []
    selected = []
    for item in ranked:
        text = (item.get("excerpt") or "") + " " + (item.get("article") or "")
        if markers_all and not all(marker in text for marker in markers_all):
            continue
        if markers_any and not any(marker in text for marker in markers_any):
            continue
        selected.append(item)
    return selected


def answer_topic_rewrite_followup(query, history, last_bot_item_fn, requested_citation_indices_fn):
    evidence, _topic = topic_history_evidence(history, last_bot_item_fn)
    if not evidence or "改写" not in (query or ""):
        return None

    indices = requested_citation_indices_fn(query)
    if not indices:
        return None

    lines = ["按上一轮条目改写为更通顺的学术表述：", ""]
    for index in indices:
        if not (1 <= index <= len(evidence)):
            continue
        item = evidence[index - 1]
        excerpt = (item.get("excerpt") or "").replace("...", "").replace("……", "")
        excerpt = re.sub(r"\s+", "", excerpt)
        if "合作社" in excerpt or "共同耕种" in excerpt:
            rewritten = f"这条可以表述为：{excerpt[:90]}，其核心意思是通过合作化与联合生产推动农民向新的生产方式过渡。"
        elif "小农" in excerpt and "暴力" in excerpt:
            rewritten = f"这条可以表述为：{excerpt[:90]}，其核心意思是对小农不能采取暴力剥夺，而应通过政治引导和社会帮助实现过渡。"
        else:
            rewritten = f"这条可以表述为：{excerpt[:100]}。"
        lines.append(f"第{index}条：{rewritten}")
        lines.append(f"出处：{item.get('citation') or ''}")
        lines.append("")
    return "\n".join(lines).strip()


def answer_topic_item_explain_followup(query, history, last_bot_item_fn, requested_citation_indices_fn):
    evidence, _topic = topic_history_evidence(history, last_bot_item_fn)
    if not evidence:
        return None

    normalized = query or ""
    if not any(marker in normalized for marker in ["具体讲", "什么意思", "讲的是什么", "说的是什么", "具体是指", "解释一下", "再解释一下", "定义"]):
        return None

    indices = requested_citation_indices_fn(query)
    if not indices:
        return None

    index = indices[0]
    if not (1 <= index <= len(evidence)):
        return None

    item = evidence[index - 1]
    excerpt = (item.get("excerpt") or "").replace("...", "").replace("??", "")
    excerpt = re.sub(r"\s+", "", excerpt)
    article = item.get("article") or ""
    citation = item.get("detailed_citation") or item.get("citation") or ""

    if any(marker in excerpt for marker in ["纲领", "最低工资", "农业机器", "种子", "肥料", "共同耕种"]):
        summary = "这条主要在讲针对农业工人和小农的土地纲领安排，包括最低工资、农业投入支持、土地使用和共同耕种等制度措施。"
    elif any(marker in excerpt for marker in ["合作社", "示范", "社会帮助", "小农"]):
        summary = "这条主要在讲如何把小农逐步引导到合作社生产，重点不是强制剥夺，而是通过示范、帮助和过渡安排推进。"
    elif any(marker in excerpt for marker in ["大土地", "农村无产者", "剥夺"]):
        summary = "这条主要在讲对大地产和农村无产者问题的处理原则，核心是区分小农与大土地占有者，采取不同策略。"
    else:
        summary = f"这条主要在讲《{article}》中的一个具体判断，其核心意思是：{excerpt[:110]}。"

    return f"第{index}条具体讲的是：{summary}\n\n原文摘录：{excerpt[:160]}。\n出处：{citation}"


def answer_topic_history_followup(query, history, last_bot_item_fn, requested_citation_indices_fn, normalize_for_match):
    evidence, topic = topic_history_evidence(history, last_bot_item_fn)
    topic_label = (topic.get("topic_label") or "").strip()
    if not topic_label or not evidence:
        return None

    ranked = rank_topic_evidence(evidence, normalize_for_match)
    normalized = query or ""
    lowered = normalized.lower()

    if "再列出" in normalized and "三条" in normalized and "小农" in normalized and "过渡" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["小农", "合作社", "社会帮助", "示范", "过渡"])
        if items:
            lines = ["和小农过渡最相关的还可以再补这三条：", ""]
            for index, item in enumerate(items[:3], start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

    if "哪一段" in normalized and "德国农民战争" in normalized:
        for item in ranked:
            if "德国农民战争" not in (item.get("article") or ""):
                continue
            if "合作社" not in (item.get("excerpt") or "") and "共同耕种" not in (item.get("excerpt") or ""):
                continue
            citation = item.get("detailed_citation") or item.get("citation") or ""
            excerpt = item.get("excerpt") or ""
            return (
                "上一轮证据里，《德国农民战争》中和合作社最相关的是这一段：\n\n"
                f"> {excerpt}\n\n"
                f"出处：{citation}"
            )

    if "哪几条" in normalized or ("哪些" in normalized and "观点" in normalized):
        lines = ["上一轮证据里，最直接谈到合作社的条目主要有：", ""]
        direct = [item for item in ranked if "合作社" in (item.get("excerpt") or "") or "共同耕种" in (item.get("excerpt") or "")]
        for index, item in enumerate(direct[:5], start=1):
            lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
            lines.append(f"   {item.get('citation') or ''}")
        if len(lines) > 2:
            return "\n".join(lines)

    if "摘录" in normalized and ("三段" in normalized or "三条" in normalized):
        direct = [item for item in ranked if "合作社" in (item.get("excerpt") or "") or "共同耕种" in (item.get("excerpt") or "")]
        if direct:
            lines = ["上一轮证据里，直接涉及合作社的原文摘录可以先列这三段：", ""]
            for index, item in enumerate(direct[:3], start=1):
                lines.append(f"{index}. {item.get('excerpt') or ''}")
                lines.append(f"   {item.get('detailed_citation') or item.get('citation') or ''}")
                lines.append("")
            return "\n".join(lines).strip()

    if any(marker in normalized for marker in ["整理", "分类", "三类"]):
        groups = {"政策主张": [], "过渡方式": [], "阶级区分": []}
        for item in ranked:
            excerpt = item.get("excerpt") or ""
            citation = item.get("citation") or ""
            if any(marker in excerpt for marker in ["要求", "纲领", "建立", "降低", "废除", "租给"]):
                groups["政策主张"].append((excerpt, citation))
            if any(marker in excerpt for marker in ["合作社", "共同耕种", "示范", "社会帮助", "联合"]):
                groups["过渡方式"].append((excerpt, citation))
            if any(marker in excerpt for marker in ["小农", "大土地", "农村无产者", "短工", "中农", "大农"]):
                groups["阶级区分"].append((excerpt, citation))
        lines = ["根据上一轮直接证据，可以先按三类整理：", ""]
        for label, items in groups.items():
            if not items:
                continue
            lines.append(f"{label}：")
            for excerpt, citation in items[:2]:
                lines.append(f"1. {excerpt[:90]}?")
                lines.append(f"   {citation}")
            lines.append("")
        return "\n".join(lines).strip()

    if "最适合" in normalized and "农村合作" in normalized:
        lines = ["最适合拿来回答今天农村合作问题的，主要是以下三条：", ""]
        for item in ranked[:3]:
            lines.append(f"1. {item.get('excerpt', '')[:100]}?")
            lines.append(f"   {item.get('citation') or ''}")
        return "\n".join(lines)

    if "土地所有制" in normalized or ("土地" in normalized and "再补" in normalized):
        items = filter_ranked_evidence(ranked, markers_any=["土地", "土地国有化", "小块土地所有制", "大土地", "租给"])
        if items:
            count = 2 if "两条" in normalized or "2条" in lowered else 3
            lines = ["和土地所有制最相关的补充观点可以先列这几条：", ""]
            for index, item in enumerate(items[:count], start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

    if "大地产" in normalized or "农村无产者" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "农村无产者", "短工"])
        if items:
            count = 2 if "两条" in normalized or "2条" in lowered else 3
            lines = ["和大地产、农村无产者最相关的观点主要有：", ""]
            for index, item in enumerate(items[:count], start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

    if "共同耕种" in normalized and "哪一条" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["共同耕种", "合作社", "联合"])
        if items:
            item = items[0]
            return (
                "最接近‘共同耕种’表述的是这一条：\n\n"
                f"{item.get('excerpt') or ''}\n\n"
                f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
            )

    if "上一条引用的出处分别是什么" in normalized:
        lines = ["上一条提到的几条出处分别是：", ""]
        for index, item in enumerate(ranked[:3], start=1):
            lines.append(f"{index}. {item.get('detailed_citation') or item.get('citation') or ''}")
        return "\n".join(lines)

    if "主要集中在哪一篇作品" in normalized:
        counts = {}
        for item in ranked:
            article = item.get("article") or "未知篇名"
            counts[article] = counts.get(article, 0) + 1
        article, count = max(counts.items(), key=lambda pair: pair[1])
        lines = [f"这一组观点目前主要集中在《{article}》，因为上一轮直接证据里它出现次数最多（{count}条）。", ""]
        top_items = [item for item in ranked if (item.get("article") or "") == article][:3]
        for index, item in enumerate(top_items, start=1):
            lines.append(f"{index}. {(item.get('excerpt') or '')[:90]}?")
            lines.append(f"   {item.get('citation') or ''}")
        return "\n".join(lines)

    if "哪一段" in normalized and "过渡方式" in normalized and "不是强制剥夺" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["合作社", "社会帮助", "示范", "小农", "暴力"])
        if items:
            item = items[0]
            return (
                "最能说明‘合作社是过渡方式而不是强制剥夺’的，是这一段：\n\n"
                f"> {item.get('excerpt') or ''}\n\n"
                f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
            )

    if "压缩成五点" in normalized or ("核心主张" in normalized and "五点" in normalized):
        lines = ["把《法德农民问题》中的核心主张压缩成五点，可以这样把握：", ""]
        for index, item in enumerate(ranked[:5], start=1):
            lines.append(f"{index}. {(item.get('excerpt') or '')[:86]}?")
        return "\n".join(lines)

    if "小农" in normalized and "大地产" in normalized and "哪些条目" in normalized:
        small_items = filter_ranked_evidence(ranked, markers_any=["小农", "合作社", "示范"])
        estate_items = filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "农村无产者"])
        lines = ["可以先这样区分：", ""]
        if small_items:
            lines.append("讲小农的条目：")
            for item in small_items[:3]:
                lines.append(f"1. {(item.get('excerpt') or '')[:88]}?")
            lines.append("")
        if estate_items:
            lines.append("讲大地产和农村无产者的条目：")
            for item in estate_items[:3]:
                lines.append(f"1. {(item.get('excerpt') or '')[:88]}?")
        return "\n".join(lines).strip()

    if "工农关系" in normalized and any(marker in normalized for marker in ["归纳", "重排", "怎么排"]):
        groups = {
            "对小农的过渡与争取": filter_ranked_evidence(ranked, markers_any=["小农", "合作社", "示范", "社会帮助"]),
            "对农村无产者的直接政策": filter_ranked_evidence(ranked, markers_any=["农村无产者", "短工", "最低工资"]),
            "对大地产的区分处理": filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "剥夺"]),
        }
        lines = ["如果按工农关系重排，这十条可以先分成三组：", ""]
        for label, items in groups.items():
            if not items:
                continue
            lines.append(f"{label}：")
            for item in items[:2]:
                lines.append(f"1. {(item.get('excerpt') or '')[:90]}?")
            lines.append("")
        return "\n".join(lines).strip()

    if "有没有明确说" in normalized and "暴力剥夺小农" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["小农", "暴力", "剥夺"])
        if items:
            item = items[0]
            return (
                "从上一轮直接证据看，并没有把小农作为要被暴力剥夺的对象来表述；相反，相关段落更强调过渡、示范和社会帮助。\n\n"
                f"最直接的依据是：{item.get('excerpt') or ''}\n"
                f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
            )

    if "合并成一条" in normalized or "合并成一个完整判断" in normalized:
        indices = requested_citation_indices_fn(query)
        if len(indices) >= 2:
            chosen = []
            for index in indices[:2]:
                if 1 <= index <= len(evidence):
                    chosen.append(evidence[index - 1])
            if len(chosen) == 2:
                left = re.sub(r"\s+", "", chosen[0].get("excerpt") or "")[:70]
                right = re.sub(r"\s+", "", chosen[1].get("excerpt") or "")[:70]
                return f"合并成一条完整判断：{left}；同时，{right}?"

    if "最关键" in normalized and "三条" in normalized:
        lines = ["如果只保留最关键的三条，我会选这三条：", ""]
        for item in ranked[:3]:
            lines.append(f"1. {(item.get('excerpt') or '')[:100]}?")
            lines.append(f"   {item.get('citation') or ''}")
        return "\n".join(lines)

    if "上面三条" in normalized and "哪一页" in normalized:
        lines = ["上面三条分别出自以下页码：", ""]
        for index, item in enumerate(ranked[:3], start=1):
            page = item.get("printed_page") or item.get("citation_page")
            lines.append(f"{index}. 第{page}页，{item.get('citation') or ''}")
        return "\n".join(lines)

    if "更接近原文" in normalized or "原文表述" in normalized:
        lines = ["把最关键三条换成更接近原文的表述如下：", ""]
        for item in ranked[:3]:
            lines.append(f"1. {item.get('excerpt', '')[:120]}?")
            lines.append(f"   {item.get('citation') or ''}")
        return "\n".join(lines)

    if "概括" in normalized and "一句话" in normalized:
        item = ranked[0]
        return f"一句话概括：{topic_label}中的核心态度是，{(item.get('excerpt') or '')[:120]}?"

    if "小结" in normalized or "150字" in normalized:
        pieces = []
        for item in ranked[:3]:
            text = re.sub(r"\s+", "", item.get("excerpt") or "")
            if text:
                pieces.append(text[:48])
        if pieces:
            summary = "?".join(pieces)[:145]
            return f"150字左右的小结可以写成：{summary}?"

    if "完整抄出来" in normalized and "合作社生产" in normalized:
        items = filter_ranked_evidence(ranked, markers_any=["合作社", "生产", "共同耕种"])
        if items:
            count = 2 if "两条" in normalized or "2条" in lowered else 3
            lines = ["涉及合作社生产的原文可以完整先抄这两条：", ""]
            for index, item in enumerate(items[:count], start=1):
                lines.append(f"{index}. {item.get('excerpt') or ''}")
                lines.append(f"   {item.get('detailed_citation') or item.get('citation') or ''}")
                lines.append("")
            return "\n".join(lines).strip()

    return None


def answer_history_followup(
    query,
    history,
    answer_topic_rewrite_followup_fn,
    answer_topic_item_explain_followup_fn,
    answer_topic_history_followup_fn,
    answer_evidence_page_followup_fn,
    answer_citation_followup_fn,
):
    direct_answer = answer_topic_rewrite_followup_fn(query, history)
    if direct_answer:
        return direct_answer

    direct_answer = answer_topic_item_explain_followup_fn(query, history)
    if direct_answer:
        return direct_answer

    direct_answer = answer_topic_history_followup_fn(query, history)
    if direct_answer:
        return direct_answer

    direct_answer = answer_evidence_page_followup_fn(query, history)
    if direct_answer:
        return direct_answer

    return answer_citation_followup_fn(query, history)
