"""
MarxOS Work Catalog — 著作元数据查询模块。

从 work_catalog.json 加载 94 篇著作的结构化元数据，提供：
- 标题匹配（精确/模糊）
- 概念词 → 著作反向检索
- 查询意图 → 检索约束（sources + page_ranges）

用法:
    from marxos_work_catalog import WorkCatalog
    catalog = WorkCatalog()
    work = catalog.match_query("费尔巴哈的实践观")  # → work dict or None
    entries = catalog.get_entries(work)               # → [{source, start_page, end_page, ...}]
"""
import json
import os
import re
from pathlib import Path

_WORK_CATALOG_PATH = os.getenv(
    "WORK_CATALOG_PATH",
    str(Path(__file__).resolve().parent / "rag" / "work_catalog.json"),
)


def _normalize(text):
    """Strip punctuation, whitespace, lower-case for matching."""
    if not text:
        return ""
    text = re.sub(r"[《》（）()〈〉\[\]【】\s\.·•\-—_\"\'﹐，。；：！？、]", "", text)
    text = re.sub(r"[*]", "", text)
    return text.strip()


class WorkCatalog:
    """Load and query the MarxOS work metadata catalog."""

    def __init__(self, path=None):
        path = path or _WORK_CATALOG_PATH
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.works = data["works"]
        self.editions_meta = data.get("editions", {})
        self.taxonomy = data.get("discipline_taxonomy", {})

        # Build lookup indices
        self._by_id = {w["work_id"]: w for w in self.works}
        self._title_index = {}     # normalized title → work
        self._alias_index = {}     # normalized alias → work
        self._concept_index = {}   # concept → [work_ids]
        self._quote_index = {}     # normalized quote (first 40 chars) → work

        for w in self.works:
            # Title index
            norm_title = _normalize(w["title"])
            self._title_index[norm_title] = w

            # Alias index
            for alias in w.get("aliases", []):
                norm_alias = _normalize(alias)
                if norm_alias and norm_alias not in self._alias_index:
                    self._alias_index[norm_alias] = w

            # Quote index — index both full quote and key phrase (first 30 chars)
            for quote in w.get("quotes", []):
                norm_quote = _normalize(quote)
                if norm_quote and len(norm_quote) >= 8:
                    self._quote_index[norm_quote[:60]] = w
                    # Also index the first key phrase (first 20+ chars)
                    key_phrase = norm_quote[:30]
                    if key_phrase not in self._quote_index:
                        self._quote_index[key_phrase] = w

            # Concept index — index both regular concepts AND primary_concepts
            for concept in w.get("concepts", []):
                norm_concept = _normalize(concept)
                if norm_concept not in self._concept_index:
                    self._concept_index[norm_concept] = []
                self._concept_index[norm_concept].append(w["work_id"])
            # Primary concepts also indexed (they're the most distinctive)
            for concept in w.get("primary_concepts", []):
                norm_concept = _normalize(concept)
                if norm_concept not in self._concept_index:
                    self._concept_index[norm_concept] = []
                if w["work_id"] not in self._concept_index[norm_concept]:
                    self._concept_index[norm_concept].append(w["work_id"])

    # ── Query Interface ──────────────────────────────────────────

    def match_query(self, query, normalize_fn=None):
        """Match a user query to the most relevant work.

        Strategy (in order):
          1. Full normalized title appears in query → direct hit
          2. Alias appears in query → direct hit
          3. Significant title segment (≥4 chars) appears in query → best match wins
          4. Concept-based fallback (delegates to match_by_concepts)

        Returns the work dict if matched, None otherwise.
        """
        norm_fn = normalize_fn or _normalize
        qn = norm_fn(query)
        if not qn:
            return None

        # 1. Full title match (longest match wins for disambiguation)
        best_title_match = None
        best_title_len = 0
        for norm_title, work in self._title_index.items():
            if norm_title and len(norm_title) >= 4 and norm_title in qn:
                if len(norm_title) > best_title_len:
                    best_title_match = work
                    best_title_len = len(norm_title)
        if best_title_match:
            return best_title_match

        # 1.5. Quote match — check if query contains a known quote
        # (works even when title/alias matching fails, e.g. "宗教是人民的鸦片")
        best_quote_match = None
        best_quote_len = 0
        for norm_quote, work in self._quote_index.items():
            # Check if the quote or its key phrase appears in the query
            if len(norm_quote) >= 8 and norm_quote[:30] in qn:
                if len(norm_quote) > best_quote_len:
                    best_quote_match = work
                    best_quote_len = len(norm_quote)
            # Also check: shorter key phrases
            elif len(norm_quote) >= 15 and norm_quote[:20] in qn:
                if len(norm_quote) > best_quote_len:
                    best_quote_match = work
                    best_quote_len = len(norm_quote)
        if best_quote_match:
            return best_quote_match

        # 2. Alias match — longest matching alias wins (handles disambiguation)
        # Min 3 chars for distinctive short titles (资本论, 宣言, etc.)
        best_alias_match = None
        best_alias_len = 0
        for norm_alias, work in self._alias_index.items():
            min_len = 3
            if norm_alias and len(norm_alias) >= min_len and norm_alias in qn:
                if len(norm_alias) > best_alias_len:
                    best_alias_match = work
                    best_alias_len = len(norm_alias)
        if best_alias_match:
            return best_alias_match

        # 3. Partial segment match — split title into meaningful segments
        best_match = None
        best_len = 0
        for norm_title, work in self._title_index.items():
            # Split title on common delimiters to get meaningful segments
            segments = re.split(r"[—\-－与和及、，《》（）()\s]", norm_title)
            # Also try the full title (for short single-segment titles)
            if len(norm_title) <= 12 and len(norm_title) >= 4:
                segments.append(norm_title)
            for segment in segments:
                segment = segment.strip()
                # Require ≥4 chars AND segment must be in query
                if len(segment) >= 4 and segment in qn:
                    if len(segment) > best_len:
                        best_match = work
                        best_len = len(segment)

        if best_match:
            return best_match

        # 4. Concept-based fallback with primary_concept scoring
        concept_hits = self.match_by_concepts(query, norm_fn)
        if concept_hits:
            # Score each candidate: primary concepts get massive priority
            scored = []
            for work, matched_concepts in concept_hits:
                primary = set(work.get("primary_concepts", []))
                regular = set(work.get("concepts", []))
                score = 0
                for c in matched_concepts:
                    if c in primary:
                        score += 100  # Primary concept — this work is THE source
                    elif c in regular:
                        score += 10   # Regular concept — this work discusses it
                score += len(matched_concepts) * 2  # Bonus for multiple matches
                scored.append((score, len(matched_concepts), work))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_score = scored[0][0]
            # Require at least score >= 100 (primary hit) or >= 22 (2+ regular concepts)
            if best_score >= 100 or best_score >= 22:
                return scored[0][2]

        return None

    def match_by_concepts(self, query, normalize_fn=None):
        """Find works whose concepts appear in the query.

        Returns list of (work, matched_concepts) tuples, sorted by relevance.
        """
        norm_fn = normalize_fn or _normalize
        qn = norm_fn(query)
        if not qn:
            return []

        hits = {}  # work_id → [matched_concepts]
        for concept, work_ids in self._concept_index.items():
            if concept and len(concept) >= 2 and concept in qn:
                for wid in work_ids:
                    if wid not in hits:
                        hits[wid] = []
                    hits[wid].append(concept)

        results = []
        for wid, matched_concepts in hits.items():
            work = self._by_id.get(wid)
            if work:
                results.append((work, matched_concepts))

        # Sort by number of matched concepts (desc)
        results.sort(key=lambda x: len(x[1]), reverse=True)
        return results

    def get_entries(self, work, preferred_edition="xuanji"):
        """Convert a work dict to constraint entries in the format expected by retrieval.

        Returns list of entry dicts: [{source, article, start_page, end_page, ...}]
        """
        entries = []
        editions = work.get("editions", {})

        # Sort: preferred edition first, then others by priority
        def sort_key(item):
            ek, ev = item
            is_preferred = ek.startswith(preferred_edition)
            is_full = ev.get("is_full_text", True)
            return (not is_preferred, not is_full)

        for ek, ev in sorted(editions.items(), key=sort_key):
            entry = {
                "source": ev["source"],
                "article": ev.get("article_title", work["title"]),
                "start_page": ev["start_page"],
                "end_page": ev["end_page"],
                "classic_title": work["title"],
                "classic_author": work.get("author"),
                "classic_work_type": work.get("work_type"),
                "entry_type": ev.get("entry_type", "primary"),
                "is_full_text": ev.get("is_full_text", True),
            }
            entries.append(entry)

        return entries

    def get_constraints(self, work, preferred_edition="xuanji"):
        """Build retrieval constraints dict for a matched work.

        Returns dict with title, sources, page_ranges, entries, strict_title.
        """
        entries = self.get_entries(work, preferred_edition)
        if not entries:
            return {}

        page_ranges = {}
        for entry in entries:
            page_ranges.setdefault(entry["source"], []).append(
                (entry["start_page"], entry["end_page"])
            )

        return {
            "title": work["title"],
            "strict_title": True,
            "entries": entries,
            "sources": {e["source"] for e in entries},
            "page_ranges": page_ranges,
        }

    def lookup_by_id(self, work_id):
        """Get a work by its work_id."""
        return self._by_id.get(work_id)

    def get_works_by_discipline(self, discipline):
        """List all works in a given discipline."""
        return [w for w in self.works if discipline in w.get("discipline", [])]

    def __len__(self):
        return len(self.works)


# ── Module-level convenience ─────────────────────────────────────

_catalog = None


def get_catalog():
    """Get or lazily create the global WorkCatalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = WorkCatalog()
    return _catalog
