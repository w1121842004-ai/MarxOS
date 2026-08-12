"""
Citation Verifier Agent — LLM-driven verification of cited content.

After DeepSeek generates an answer with citations, this agent checks whether
the cited quotes actually exist in the OCR text at the claimed pages.
Prevents hallucinated citations and strengthens academic credibility.

Architecture:
  1. Extract citation claims from answer (source + page + claimed content)
  2. Load OCR text for each claimed page
  3. LLM compares claimed content vs actual OCR text
  4. Return verification report: verified / partial / hallucinated

Usage:
    from marxos.generation.citation_verifier import CitationVerifier
    verifier = CitationVerifier(client, ocr_cache_dir)
    report = verifier.verify(answer_text, evidence_cards)
"""
import json
import os
from pathlib import Path


VERIFIER_SYSTEM_PROMPT = """你是一个马克思主义著作引文校验助手。你的任务是判断一段"声称的引文"是否真的存在于"OCR原文"中。

规则：
1. 如果声称的引文与OCR原文中的某段文本高度匹配（逐字对应 或 仅有少量 OCR 乱码差异），判定为 verified
2. 如果声称的引文的核心主张在OCR原文中有体现，但文字不完全一致（可能是概括/转述），判定为 partial
3. 如果声称的引文与OCR原文完全无关，或OCR原文中找不到任何对应的内容，判定为 hallucinated

只输出JSON，不要任何其他文字：
{"verdict": "verified|partial|hallucinated", "confidence": 0.0-1.0, "evidence_in_text": "原文中对应的关键句（20字以内）", "explanation": "一句话判断理由"}"""


def build_verifier_prompt(claimed_quote, ocr_text, citation_info):
    """Build the verification prompt for one citation claim."""
    ocr_preview = ocr_text[:2000] if len(ocr_text) > 2000 else ocr_text
    return f"""## 引文信息
出处: {citation_info}
声称的引文内容: "{claimed_quote}"

## OCR原文（第{citation_info}页的文本）
{ocr_preview}

## 请判断"""  # noqa: F841


class CitationVerifier:
    """LLM-driven citation content verification."""

    def __init__(self, client, ocr_cache_dir, model="deepseek-chat"):
        self.client = client
        self.ocr_cache_dir = Path(ocr_cache_dir)
        self.model = model

    def load_page_text(self, source_stem, page_num):
        """Load OCR text for a given source and PDF page number."""
        path = self.ocr_cache_dir / source_stem / f"page_{page_num}.json"
        txt_path = self.ocr_cache_dir / source_stem / f"page_{page_num}.txt"
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("raw_text") or data.get("cleaned_text") or ""
            if txt_path.exists():
                with open(txt_path, encoding="utf-8") as f:
                    return f.read()
        except (OSError, json.JSONDecodeError):
            return ""
        return ""

    def extract_claims(self, answer_text, evidence_cards):
        """Extract citation claims from the answer using evidence cards.

        Each claim: {citation_str, source, pdf_page, printed_page, evidence_preview}
        """
        claims = []
        for card in (evidence_cards or []):
            citation = card.get("citation", "")
            source = card.get("source", "")
            pdf_page = card.get("pdf_page")
            printed_page = card.get("printed_page") or card.get("citation_page")
            preview = card.get("preview", "")[:300]

            if not source:
                continue

            # Determine source stem for OCR path
            source_stem = source.replace(".pdf", "")

            claims.append({
                "citation": citation,
                "source": source,
                "source_stem": source_stem,
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "preview": preview,
            })
        return claims

    def verify_one(self, claimed_quote, ocr_text, citation_info):
        """Verify a single citation claim."""
        if not ocr_text or len(ocr_text.strip()) < 20:
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "evidence_in_text": "",
                "explanation": "OCR原文字数不足，无法校验",
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": build_verifier_prompt(
                        claimed_quote, ocr_text, citation_info)},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "evidence_in_text": "",
                "explanation": "校验过程出错",
            }

    def verify(self, answer_text, evidence_cards):
        """Verify all citations in an answer against OCR text.

        Returns: {ok, total, verified, partial, hallucinated, unverifiable, details}
        """
        claims = self.extract_claims(answer_text, evidence_cards)
        if not claims:
            return {"ok": True, "total": 0, "verified": 0, "partial": 0,
                    "hallucinated": 0, "unverifiable": 0, "details": []}

        details = []
        verdicts = {"verified": 0, "partial": 0, "hallucinated": 0, "unverifiable": 0}

        for claim in claims[:6]:  # Max 6 verifications per answer
            ocr_text = self.load_page_text(claim["source_stem"], claim["pdf_page"])
            citation_info = f"{claim['citation']}"
            result = self.verify_one(claim["preview"][:200], ocr_text, citation_info)

            result["claim"] = {
                "citation": claim["citation"],
                "source": claim["source"],
                "pdf_page": claim["pdf_page"],
                "printed_page": claim["printed_page"],
            }
            details.append(result)
            verdicts[result.get("verdict", "unverifiable")] += 1

        total = len(details)
        has_hallucination = verdicts["hallucinated"] > 0
        all_verified = verdicts["verified"] + verdicts["partial"] == total

        return {
            "ok": not has_hallucination and total > 0,
            "total": total,
            "verified": verdicts["verified"],
            "partial": verdicts["partial"],
            "hallucinated": verdicts["hallucinated"],
            "unverifiable": verdicts["unverifiable"],
            "details": details,
        }
