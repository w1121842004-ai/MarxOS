"""
Book Locator Agent — LLM-driven work/chapter locator for complex queries.

When rule-based WorkCatalog.match_query() fails or returns low confidence,
this agent uses DeepSeek to map the user's question to specific works and
chapters. It returns structured constraints for the retrieval pipeline.

Usage:
    from marxos_book_locator import BookLocator
    locator = BookLocator(client, catalog)
    result = locator.locate("《资本论》中关于商品拜物教的论述在第几章？")
    # → {work_ids: ["capital-vol1"], constraints: {...}, confidence: 0.9}
"""
import json
import os

# ── Catalog summary builder ──────────────────────────────────────

def build_catalog_summary(catalog):
    """Build a compact text summary of the work catalog for the LLM prompt.

    Groups works by discipline, lists work_id + title + key concepts.
    Fits in ~3K tokens for 89 works.
    """
    lines = []
    by_discipline = {}

    for w in catalog.works:
        for d in w.get("discipline", []):
            by_discipline.setdefault(d, []).append(w)

    disc_labels = {
        "philosophy": "马克思主义哲学",
        "political_economy": "政治经济学",
        "scientific_socialism": "科学社会主义",
    }

    for disc in ["philosophy", "political_economy", "scientific_socialism"]:
        works = by_discipline.get(disc, [])
        lines.append(f"## {disc_labels.get(disc, disc)}")
        for w in works:
            wid = w["work_id"]
            title = w["title"]
            author = w.get("author", "")
            co = w.get("co_author", "")
            author_str = f"{author}、{co}" if co else author
            year = w.get("writing_year", "")
            primary = w.get("primary_concepts", [])[:5]
            aliases = w.get("aliases", [])[:3]
            concept_str = "、".join(primary) if primary else ""
            alias_str = "、".join(aliases) if aliases else ""

            line = f"- [{wid}] {title} ({author_str}, {year})"
            if concept_str:
                line += f" 核心主题: {concept_str}"
            if alias_str:
                line += f" 别名: {alias_str}"
            lines.append(line)

    return "\n".join(lines)


# ── Locator prompt ───────────────────────────────────────────────

LOCATOR_SYSTEM_PROMPT = """你是一个马克思主义著作定位助手。用户会提出一个关于马克思/恩格斯著作的问题，
你需要判断这个问题应该查阅哪些著作来回答。

你的任务是输出一个JSON对象，包含：
- work_ids: 应该查阅的著作work_id列表（从下方目录中选择，最多3个）
- confidence: 你对这个判断的确信度（0.0-1.0）
- reasoning: 一句话解释你的判断依据

规则：
1. 如果用户提到了具体的篇名（如"关于费尔巴哈的提纲"），优先匹配那个work_id
2. 如果用户问的是某个概念（如"剩余价值""异化劳动"），匹配核心主题中包含该概念的著作
3. 如果问题涉及多个著作，列出最相关的2-3个
4. 只从下方目录中选择work_id，不要编造
5. 只输出JSON，不要任何其他文字"""


def build_locate_prompt(query, catalog_summary):
    """Build the user prompt for the book locator."""
    return f"""## 著作目录

{catalog_summary}

## 用户问题

{query}

## 请输出JSON"""


# ── Book Locator ─────────────────────────────────────────────────

class BookLocator:
    """LLM-driven work locator. Uses DeepSeek to map queries → work_ids."""

    def __init__(self, client, catalog, model="deepseek-chat"):
        self.client = client
        self.catalog = catalog
        self.model = model
        self._summary = None

    @property
    def catalog_summary(self):
        if self._summary is None:
            self._summary = build_catalog_summary(self.catalog)
        return self._summary

    def locate(self, query):
        """Map a user query to work_ids with confidence.

        Returns dict: {work_ids: [...], confidence: float, reasoning: str}
        Returns None on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LOCATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_locate_prompt(query, self.catalog_summary)},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON — handle markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)

            # Validate
            if "work_ids" not in result:
                return None

            # Filter to known work_ids
            known_ids = {w["work_id"] for w in self.catalog.works}
            valid_ids = [wid for wid in result.get("work_ids", []) if wid in known_ids]

            if not valid_ids:
                return None

            return {
                "work_ids": valid_ids[:3],
                "confidence": min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
                "reasoning": result.get("reasoning", ""),
            }

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Log but don't crash — locator is best-effort
            return None

    def get_constraints(self, query):
        """Full pipeline: locate works → build retrieval constraints.

        Returns constraints dict compatible with retrieval.constraints format,
        or empty dict if location fails.
        """
        result = self.locate(query)
        if not result or not result.get("work_ids"):
            return {}

        all_entries = []
        all_sources = set()
        all_page_ranges = {}

        for wid in result["work_ids"]:
            work = self.catalog.lookup_by_id(wid)
            if not work:
                continue
            entries = self.catalog.get_entries(work)
            all_entries.extend(entries)
            for e in entries:
                all_sources.add(e["source"])
                all_page_ranges.setdefault(e["source"], []).append(
                    (e["start_page"], e["end_page"])
                )

        if not all_entries:
            return {}

        title = self.catalog.lookup_by_id(result["work_ids"][0])["title"]

        return {
            "title": title,
            "strict_title": True,
            "entries": all_entries,
            "sources": all_sources,
            "page_ranges": all_page_ranges,
            "_locator_confidence": result.get("confidence", 0.5),
            "_locator_reasoning": result.get("reasoning", ""),
            "_locator_work_ids": result["work_ids"],
        }
