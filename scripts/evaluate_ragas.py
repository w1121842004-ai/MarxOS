from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


DEFAULT_DATASET = ROOT_DIR / "eval_dataset.json"
DEFAULT_SAMPLES = ROOT_DIR / "logs" / "ragas_samples.jsonl"
DEFAULT_REPORT = ROOT_DIR / "logs" / "ragas_report.json"

_APP = None


def load_env_file(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_app():
    global _APP
    if _APP is None:
        import app as marxos_app

        _APP = marxos_app
    return _APP


def compact_text(text: Any, limit: int = 4000) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:limit]


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    return data[:limit] if limit else data


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def reference_from_case(case: dict[str, Any]) -> str:
    reference = case.get("reference") or case.get("ground_truth") or case.get("answer")
    if reference:
        return str(reference)

    expected_work = case.get("expected_work")
    expected_author = case.get("expected_author")
    parts = []
    if expected_work:
        parts.append(f"预期出处：{expected_work}")
    if expected_author:
        parts.append(f"预期作者：{expected_author}")
    notes = case.get("notes")
    if notes:
        parts.append(f"备注：{notes}")
    return "；".join(parts) or "无参考答案。"


def doc_to_context(doc) -> str:
    app = get_app()
    metadata = app.normalize_metadata(doc.metadata)
    title = metadata.get("article") or metadata.get("section") or metadata.get("book") or metadata.get("source")
    page = metadata.get("citation_page") or metadata.get("printed_page") or metadata.get("page")
    source = metadata.get("source")
    prefix = f"来源：{source or ''}；篇目：{title or ''}；页码：{page or ''}"
    return compact_text(prefix + "\n" + str(doc.page_content or ""))


def build_sample(case: dict[str, Any], db, top_k: int, generate_answers: bool, answer_mode: str) -> dict[str, Any]:
    app = get_app()
    question = str(case.get("question") or "").strip()
    docs = app.retrieve_documents(question, db, k=top_k, allow_exact_quote=False)
    contexts = [doc_to_context(doc) for doc in docs]
    if generate_answers:
        answer = app.run_query(
            question,
            force_intent=case.get("intent"),
            performance=answer_mode,
        )
    else:
        answer = str(case.get("response") or case.get("generated_answer") or "")

    return {
        "id": case.get("id"),
        "user_input": question,
        "question": question,
        "retrieved_contexts": contexts,
        "contexts": contexts,
        "response": answer,
        "answer": answer,
        "reference": reference_from_case(case),
        "ground_truth": reference_from_case(case),
        "expected_work": case.get("expected_work"),
        "expected_author": case.get("expected_author"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_samples(args) -> list[dict[str, Any]]:
    if args.input_samples:
        return load_jsonl(Path(args.input_samples), limit=args.limit)

    app = get_app()
    dataset = load_dataset(Path(args.dataset), limit=args.limit)
    db = app.load_vectorstore()
    samples = []
    for index, case in enumerate(dataset, start=1):
        sample = build_sample(case, db, args.top_k, args.generate_answers, args.answer_mode)
        samples.append(sample)
        print(
            json.dumps(
                {
                    "event": "ragas_sample_built",
                    "index": index,
                    "id": sample.get("id"),
                    "contexts": len(sample.get("retrieved_contexts") or []),
                    "has_response": bool(sample.get("response")),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return samples


def make_judge_llm(args):
    if not args.judge_model:
        return None

    load_env_file()

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS judge LLM requires optional dependencies. "
            "Install with: .venv/bin/pip install -r requirements-ragas.txt"
        ) from exc

    api_key = args.judge_api_key or os.getenv("RAGAS_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = args.judge_base_url or os.getenv("RAGAS_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    llm = ChatOpenAI(
        model=args.judge_model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,
        timeout=args.judge_timeout,
    )
    return LangchainLLMWrapper(llm)


def select_metrics(metric_names: list[str]):
    try:
        import ragas.metrics as metrics_module
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Install with: .venv/bin/pip install -r requirements-ragas.txt"
        ) from exc

    registry = {
        # RAGAS >= 0.2 names.
        "faithfulness": "Faithfulness",
        "answer_relevancy": "ResponseRelevancy",
        "response_relevancy": "ResponseRelevancy",
        "context_precision": "LLMContextPrecisionWithoutReference",
        "context_recall": "LLMContextRecall",
        "factual_correctness": "FactualCorrectness",
    }
    selected = []
    for name in metric_names:
        normalized = name.strip().lower()
        class_name = registry.get(normalized)
        metric_obj = getattr(metrics_module, normalized, None)
        if metric_obj is not None and not isinstance(metric_obj, type):
            selected.append(metric_obj)
            continue
        metric_cls = getattr(metrics_module, class_name or name, None)
        if metric_cls is None:
            raise ValueError(f"Unsupported or unavailable RAGAS metric: {name}")
        selected.append(metric_cls())
    return selected


def run_ragas(samples: list[dict[str, Any]], args) -> dict[str, Any]:
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Install with: .venv/bin/pip install -r requirements-ragas.txt"
        ) from exc

    rows = [
        {
            "user_input": sample["user_input"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "response": sample.get("response") or "",
            "reference": sample.get("reference") or "",
        }
        for sample in samples
    ]
    if any(not row["response"] for row in rows):
        raise ValueError(
            "RAGAS metrics need response text. Use --generate-answers, or pass --input-samples "
            "containing response/generated_answer values."
        )

    dataset = EvaluationDataset.from_list(rows)
    metrics = select_metrics(args.metrics)
    judge_llm = make_judge_llm(args)
    kwargs = {"dataset": dataset, "metrics": metrics}
    if judge_llm is not None:
        kwargs["llm"] = judge_llm
    result = evaluate(**kwargs)

    if hasattr(result, "to_pandas"):
        rows_report = json.loads(result.to_pandas().to_json(orient="records", force_ascii=False))
    else:
        rows_report = []

    if hasattr(result, "to_dict"):
        summary = result.to_dict()
    else:
        summary = dict(result) if isinstance(result, dict) else {"result": str(result)}

    return {
        "summary": summary,
        "rows": rows_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for MarxOS RAG outputs.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--input-samples", help="Existing JSONL samples with user_input/retrieved_contexts/response/reference.")
    parser.add_argument("--samples-out", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--answer-mode", choices=["fast", "standard", "deep"], default="fast")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy", "context_recall", "factual_correctness"],
    )
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", ""))
    parser.add_argument("--judge-base-url", default=os.getenv("RAGAS_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("RAGAS_API_KEY", ""))
    parser.add_argument("--judge-timeout", type=float, default=float(os.getenv("RAGAS_JUDGE_TIMEOUT", "120")))
    args = parser.parse_args()

    samples = build_samples(args)
    samples_path = Path(args.samples_out)
    write_jsonl(samples_path, samples)
    print(f"Samples: {samples_path}")

    if args.prepare_only:
        return 0

    report = run_ragas(samples, args)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": args.dataset,
                    "input_samples": args.input_samples,
                    "top_k": args.top_k,
                    "limit": args.limit,
                    "metrics": args.metrics,
                    "judge_model": args.judge_model,
                },
                **report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
