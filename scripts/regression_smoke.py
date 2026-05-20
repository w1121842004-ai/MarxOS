import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


CASES = [
    {
        "name": "concept_ai_alchemy",
        "query": "当前，AI时代的“炼丹”其背后的本质是什么",
        "expected_intent": "concept_explain",
    },
    {
        "name": "concept_human_essence",
        "query": "人的本质是什么？",
        "expected_intent": "concept_explain",
    },
    {
        "name": "concept_surplus_value",
        "query": "如何理解剩余价值这个概念？",
        "expected_intent": "concept_explain",
    },
    {
        "name": "analysis_capital_logic",
        "query": "结合现实怎么看待资本逻辑？",
        "expected_intent": "theory_analysis",
    },
    {
        "name": "quote_manifesto_ghost",
        "query": "“一个幽灵，共产主义的幽灵，在欧洲游荡。”出自哪里？",
        "expected_intent": "quote_lookup",
        "must_contain": ["共产党宣言", "2012年", "第"],
    },
    {
        "name": "quote_workers_unite",
        "query": "“全世界无产者，联合起来！”出自哪一页？",
        "expected_intent": "quote_lookup",
        "must_contain": ["共产党宣言", "2012年", "第435页"],
    },
    {
        "name": "biblio_manifesto",
        "query": "《共产党宣言》收录在哪里？",
        "expected_intent": "bibliographic_lookup",
        "must_contain": ["共产党宣言", "第376-435页"],
    },
    {
        "name": "quote_fake_should_fail",
        "query": "请给出“这是一句随便编造的引文”的准确页码",
        "expected_intent": "quote_lookup",
        "must_equal": "未能在当前 OCR 缓存中确认该引文的精确出处。",
    },
]


def check_case(case):
    name = case["name"]
    query = case["query"]
    expected_intent = case["expected_intent"]
    actual_intent = app.classify_query(query)

    errors = []
    if actual_intent != expected_intent:
        errors.append(f"intent mismatch: expected={expected_intent}, actual={actual_intent}")

    answer = None
    if "must_contain" in case or "must_equal" in case:
        answer = app.run_query(query)
        if "must_contain" in case:
            for token in case["must_contain"]:
                if token not in answer:
                    errors.append(f"missing token in answer: {token}")
        if "must_equal" in case and answer != case["must_equal"]:
            errors.append(f"answer mismatch: expected={case['must_equal']}, actual={answer}")

    return name, actual_intent, answer, errors


def main():
    total = len(CASES)
    failed = 0

    for case in CASES:
        name, actual_intent, answer, errors = check_case(case)
        if errors:
            failed += 1
            print(f"[FAIL] {name} | intent={actual_intent}")
            for err in errors:
                print(f"  - {err}")
            if answer is not None:
                first_line = answer.splitlines()[0] if answer else ""
                print(f"  - answer_head: {first_line}")
        else:
            print(f"[PASS] {name} | intent={actual_intent}")

    print(f"\nSummary: {total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
