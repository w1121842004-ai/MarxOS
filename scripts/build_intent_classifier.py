#!/usr/bin/env python3
"""
Build a lightweight intent classifier from project data sources.

**Data sources (priority order):**

1. ``eval_dataset_v2.json`` (200 questions) — manually labeled
2. ``eval_dataset_me_200.json`` (200 questions) — manually labeled
3. ``rag/work_catalog.json`` (89 works) — synthetic queries from titles/aliases/concepts
4. Rule-based system auto-labels — used as weak supervision

**Output:** ``data/intent_classifier.pkl`` (~10 KB)

**Architecture:** LogisticRegression on 384-dim sentence-transformer embeddings.
Training completes in <30 seconds on CPU.  Inference adds <1 ms after embedding.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_PATH = Path(os.getenv("INTENT_CLASSIFIER_PATH", str(ROOT / "data/intent_classifier.pkl")))
RANDOM_SEED = 42

QUERY_TEMPLATES: dict[str, list[str]] = {
    "bibliographic_lookup": [
        "《{title}》收录在哪一卷",
        "《{title}》在第几卷",
        "{title}在哪一卷",
        "《{title}》出自马恩全集哪一卷",
        "{title}的收录位置",
        "查找《{title}》的卷册信息",
    ],
    "quote_lookup": [
        "{quote}出自哪里",
        "{quote}这句话在哪一卷",
        "查找“{quote}”的出处",
        "请问“{quote}”这句话的原文出处",
        "{quote}这句话在第几页",
    ],
    "concept_explain": [
        "什么是{concept}",
        "{concept}的概念是什么",
        "如何理解{concept}",
        "请解释{concept}的定义",
        "{concept}是什么意思",
        "简述{concept}的内涵",
        "《{title}》中的{concept}是什么",
    ],
    "comparison": [
        "比较{title_a}和{title_b}的{concept}理论",
        "{concept_a}和{concept_b}有什么区别",
        "马克思和恩格斯在{concept}上的观点有什么异同",
        "对比{concept_a}与{concept_b}",
    ],
    "deep_analysis": [
        "运用马克思主义分析当代{concept}问题",
        "从马克思主义视角看数字时代的{concept}",
        "论述{concept}的当代意义",
        "写一篇关于{concept}的理论分析",
        "结合现实分析{concept}的当代价值",
        "总结《{title}》中关于{concept}的论述",
        "梳理马克思关于{concept}的理论发展线索",
    ],
    "theory_analysis": [
        "如何理解{concept}",
        "分析{concept}的理论内涵",
        "马克思如何看待{concept}",
        "为什么{concept}是马克思主义的重要概念",
        "如何评价{concept}理论",
        "{concept}在马克思主义中的地位",
    ],
    "rag_answer": [
        "介绍一下马克思主义的基本原理",
        "谈谈你对共产主义的理解",
        "马克思的主要贡献有哪些",
        "恩格斯的代表作有哪些",
        "马克思主义的发展历程",
        "今天天气怎么样",
        "帮我写一首诗",
    ],
}


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


def load_work_catalog(path: Path | None = None) -> dict:
    path = path or ROOT / "rag" / "work_catalog.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_eval_type_to_intent(qtype: str) -> str:
    """Map eval dataset question_type → canonical intent."""
    mapping = {
        "quote_lookup": "quote_lookup",
        "concept_explain": "concept_explain",
        "analysis": "theory_analysis",
        "bibliographic": "bibliographic_lookup",
        "source_context": "quote_lookup",
        "synthesis": "deep_analysis",
        "paper_writing": "deep_analysis",
        "misconception_rebuttal": "deep_analysis",
        "research_review": "deep_analysis",
    }
    return mapping.get(qtype, "rag_answer")


def generate_synthetic_queries(catalog: dict, n_per_intent: int = 80) -> list[dict]:
    """Generate synthetic labeled queries from work catalog entities."""
    random.seed(RANDOM_SEED)
    works = catalog.get("works") or []
    synthetic: list[dict] = []

    # Collect entities
    titles = [w.get("title", "") for w in works if w.get("title")]
    aliases = []
    for w in works:
        for a in w.get("aliases") or []:
            if a and len(a) >= 3:
                aliases.append(a)
    concepts = []
    for w in works:
        for c in (w.get("concepts") or []):
            if c and len(c) >= 2:
                concepts.append(c)
    quotes = []
    for w in works:
        for q in (w.get("quotes") or []):
            if q and len(q) >= 6:
                quotes.append(q)

    # Ensure uniqueness
    concepts = list(set(concepts))
    quotes = list(set(quotes))
    all_titles = titles + aliases

    for intent, templates in QUERY_TEMPLATES.items():
        for _ in range(n_per_intent):
            template = random.choice(templates)
            title = random.choice(all_titles) if all_titles else "资本论"
            concept = random.choice(concepts) if concepts else "剩余价值"
            quote = random.choice(quotes) if quotes else "全世界无产者联合起来"

            # Random pair for comparison templates
            title_a = random.choice(all_titles) if all_titles else "德意志意识形态"
            title_b = random.choice(all_titles) if all_titles else "共产党宣言"
            concept_a = random.choice(concepts) if concepts else "异化劳动"
            concept_b = random.choice(concepts) if concepts else "商品拜物教"

            try:
                query = template.format(
                    title=title,
                    concept=concept,
                    quote=quote,
                    title_a=title_a,
                    title_b=title_b,
                    concept_a=concept_a,
                    concept_b=concept_b,
                )
            except (KeyError, ValueError):
                continue

            if 5 <= len(query) <= 120:
                synthetic.append({"query": query, "intent": intent, "source": "synthetic"})

    return synthetic


def load_labeled_queries() -> list[dict]:
    """Load all manually labeled queries from eval datasets."""
    labeled: list[dict] = []

    # eval_dataset_v2.json
    v2_path = ROOT / "eval_dataset_v2.json"
    if v2_path.exists():
        for item in load_eval_dataset(v2_path):
            labeled.append({
                "query": item.get("question", ""),
                "intent": map_eval_type_to_intent(item.get("question_type", "")),
                "source": "eval_dataset_v2",
            })

    # eval_dataset_me_200.json
    me_path = ROOT / "eval_dataset_me_200.json"
    if me_path.exists():
        for item in load_eval_dataset(me_path):
            labeled.append({
                "query": item.get("query", ""),
                "intent": map_eval_type_to_intent(item.get("question_type", "")),
                "source": "eval_dataset_me_200",
            })

    return [item for item in labeled if item["query"] and item["intent"]]


def load_intent_split(path: Path) -> list[dict]:
    """Load a fixed intent dataset split produced by build_intent_dataset.py."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "query": item.get("query", ""),
            "intent": item.get("intent") or item.get("question_type", ""),
            "source": item.get("source", "intent_dataset"),
        }
        for item in data
        if item.get("query") and (item.get("intent") or item.get("question_type"))
    ]


def auto_label_with_rules(queries: list[str]) -> list[dict]:
    """Use the current rule-based system to auto-label queries (weak supervision)."""
    from marxos_query_intent import classify_query

    def _clean(text):
        import re
        return re.sub(r"[^0-9A-Za-z一-鿿]", "", str(text or "")).lower()

    labeled = []
    for q in queries:
        intent = classify_query(q, _clean)
        labeled.append({"query": q, "intent": intent, "source": "auto_labeled"})
    return labeled


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_embedding_model():
    """Lazy-load the embedding model (heavy, ~470 MB)."""
    from marxos_embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def embed_queries(queries: list[str], embedding_model) -> np.ndarray:
    """Batch-embed a list of query strings."""
    return np.array(embedding_model.embed_documents(queries), dtype=np.float32)


def train_classifier(
    embeddings: np.ndarray,
    labels: list[str],
    test_size: float = 0.15,
) -> tuple[LogisticRegression, LabelEncoder, dict]:
    """Train a logistic regression classifier and return metrics."""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y,
    )

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    label_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test, y_pred, target_names=[str(n) for n in label_names], output_dict=True,
    )

    config = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": len(label_encoder.classes_),
        "classes": [str(c) for c in label_encoder.classes_],
        "accuracy": report.get("accuracy", 0.0),
        "macro_avg_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_avg_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "embedding_dim": embeddings.shape[1],
    }

    return model, label_encoder, config


def train_classifier_fixed_splits(
    X_train: np.ndarray,
    y_train_labels: list[str],
    X_validation: np.ndarray,
    y_validation_labels: list[str],
    X_test: np.ndarray,
    y_test_labels: list[str],
) -> tuple[LogisticRegression, LabelEncoder, dict]:
    """Train on a fixed split and report validation/test metrics separately."""
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_labels + y_validation_labels + y_test_labels)

    y_train = label_encoder.transform(y_train_labels)
    y_validation = label_encoder.transform(y_validation_labels)
    y_test = label_encoder.transform(y_test_labels)

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    label_names = [str(c) for c in label_encoder.classes_]

    def _report(X: np.ndarray, y: np.ndarray) -> dict:
        pred = model.predict(X)
        return classification_report(
            y,
            pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

    validation_report = _report(X_validation, y_validation)
    test_report = _report(X_test, y_test)
    config = {
        "training_samples": len(X_train),
        "validation_samples": len(X_validation),
        "test_samples": len(X_test),
        "num_classes": len(label_encoder.classes_),
        "classes": label_names,
        "accuracy": test_report.get("accuracy", 0.0),
        "macro_avg_f1": test_report.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_avg_f1": test_report.get("weighted avg", {}).get("f1-score", 0.0),
        "validation_accuracy": validation_report.get("accuracy", 0.0),
        "validation_macro_avg_f1": validation_report.get("macro avg", {}).get("f1-score", 0.0),
        "validation_weighted_avg_f1": validation_report.get("weighted avg", {}).get("f1-score", 0.0),
        "test_accuracy": test_report.get("accuracy", 0.0),
        "test_macro_avg_f1": test_report.get("macro avg", {}).get("f1-score", 0.0),
        "test_weighted_avg_f1": test_report.get("weighted avg", {}).get("f1-score", 0.0),
        "validation_report": validation_report,
        "test_report": test_report,
        "embedding_dim": X_train.shape[1],
        "dataset": "intent_dataset_fixed_splits",
    }

    return model, label_encoder, config


def print_report_summary(name: str, report: dict) -> None:
    print(f"   {name} accuracy: {report.get('accuracy', 0.0):.1%}")
    print(f"   {name} macro F1:  {report.get('macro avg', {}).get('f1-score', 0.0):.1%}")
    print(f"   {name} weighted F1: {report.get('weighted avg', {}).get('f1-score', 0.0):.1%}")
    print(f"   {name} per-class F1:")
    for label, metrics in report.items():
        if not isinstance(metrics, dict) or label in {"macro avg", "weighted avg"}:
            continue
        print(
            f"     {label}: "
            f"P={metrics.get('precision', 0.0):.3f} "
            f"R={metrics.get('recall', 0.0):.3f} "
            f"F1={metrics.get('f1-score', 0.0):.3f} "
            f"N={int(metrics.get('support', 0))}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intent classifier")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--synthetic-per-intent", type=int, default=80,
                        help="Synthetic queries per intent (default: 80)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    parser.add_argument("--intent-dataset-dir", type=Path,
                        help="Use fixed train/validation/test splits from build_intent_dataset.py")
    parser.add_argument("--include-manual-labels", action="store_true",
                        help="Add existing manual labels to the fixed-split training set")
    parser.add_argument("--extra-train-json", action="append", type=Path, default=[],
                        help="Additional JSON list files to append to the training split")
    args = parser.parse_args()

    print("=== MarxOS Intent Classifier Builder ===\n")

    if args.intent_dataset_dir:
        dataset_dir = args.intent_dataset_dir
        print("1. Loading fixed intent dataset splits...")
        train_items = load_intent_split(dataset_dir / "intent_train.json")
        validation_items = load_intent_split(dataset_dir / "intent_validation.json")
        test_items = load_intent_split(dataset_dir / "intent_test.json")
        manual_items = load_labeled_queries() if args.include_manual_labels else []
        if manual_items:
            train_items = train_items + manual_items
        extra_items: list[dict] = []
        for extra_path in args.extra_train_json:
            extra_items.extend(load_intent_split(extra_path))
        if extra_items:
            train_items = train_items + extra_items
        print(f"   Train: {len(train_items)}")
        print(f"   Validation: {len(validation_items)}")
        print(f"   Test: {len(test_items)}")
        if manual_items:
            print(f"   Included manual calibration labels: {len(manual_items)}")
        if extra_items:
            print(f"   Included extra train labels: {len(extra_items)}")

        from collections import Counter
        for name, items in (
            ("train", train_items),
            ("validation", validation_items),
            ("test", test_items),
        ):
            dist = Counter(item["intent"] for item in items)
            print(f"\n   {name} intent distribution:")
            for intent in sorted(dist):
                print(f"     {intent}: {dist[intent]}")

        print("\n2. Loading embedding model...")
        emb = load_embedding_model()
        print("   Embedding train/validation/test queries...")
        X_train = embed_queries([item["query"] for item in train_items], emb)
        X_validation = embed_queries([item["query"] for item in validation_items], emb)
        X_test = embed_queries([item["query"] for item in test_items], emb)
        print(f"   Train embeddings shape: {X_train.shape}")
        print(f"   Validation embeddings shape: {X_validation.shape}")
        print(f"   Test embeddings shape: {X_test.shape}")

        print("\n3. Training classifier...")
        model, label_encoder, config = train_classifier_fixed_splits(
            X_train,
            [item["intent"] for item in train_items],
            X_validation,
            [item["intent"] for item in validation_items],
            X_test,
            [item["intent"] for item in test_items],
        )
        print_report_summary("Validation", config["validation_report"])
        print_report_summary("Test", config["test_report"])
        print(f"   Classes: {config['classes']}")

        print(f"\n4. Saving to {args.output}...")
        from marxos_intent_classifier import IntentClassifier
        classifier = IntentClassifier(model, label_encoder, config)
        classifier.save(args.output)
        file_size = args.output.stat().st_size
        print(f"   Saved: {args.output} ({file_size:,} bytes)")

        print("\n=== Done ===")
        print(f"Training data: {len(train_items)} train + "
              f"{len(validation_items)} validation + {len(test_items)} test")
        print(f"Model: LogisticRegression, {config['num_classes']} classes")
        print(f"Size: {file_size:,} bytes (~{file_size/1024:.0f} KB)")
        print(f"Final test accuracy: {config['test_accuracy']:.1%}")
        print(f"Final test macro F1: {config['test_macro_avg_f1']:.1%}")
        return

    # ── Step 1: Collect labeled data ──
    print("1. Collecting labeled queries...")
    labeled = load_labeled_queries()
    print(f"   Manual labels: {len(labeled)} queries")

    synthetic = []
    if not args.no_synthetic:
        print("2. Generating synthetic queries from work catalog...")
        catalog = load_work_catalog()
        synthetic = generate_synthetic_queries(catalog, n_per_intent=args.synthetic_per_intent)
        print(f"   Synthetic queries: {len(synthetic)}")

    all_queries = labeled + synthetic
    random.seed(RANDOM_SEED)
    random.shuffle(all_queries)

    # Show distribution
    from collections import Counter
    dist = Counter(q["intent"] for q in all_queries)
    print("\n   Intent distribution:")
    for intent in sorted(dist):
        print(f"     {intent}: {dist[intent]}")

    # ── Step 2: Embed ──
    print("\n3. Loading embedding model...")
    emb = load_embedding_model()
    texts = [q["query"] for q in all_queries]
    print(f"   Embedding {len(texts)} queries...")
    X = embed_queries(texts, emb)
    y = [q["intent"] for q in all_queries]
    print(f"   Embeddings shape: {X.shape}")

    # ── Step 3: Train ──
    print("\n4. Training classifier...")
    model, label_encoder, config = train_classifier(X, y)
    print(f"   Accuracy: {config['accuracy']:.1%}")
    print(f"   Macro F1:  {config['macro_avg_f1']:.1%}")
    print(f"   Weighted F1: {config['weighted_avg_f1']:.1%}")
    print(f"   Classes: {config['classes']}")

    # ── Step 4: Save ──
    print(f"\n5. Saving to {args.output}...")
    from marxos_intent_classifier import IntentClassifier
    classifier = IntentClassifier(model, label_encoder, config)
    classifier.save(args.output)
    file_size = args.output.stat().st_size
    print(f"   Saved: {args.output} ({file_size:,} bytes)")

    # ── Summary ──
    print(f"\n=== Done ===")
    print(f"Training data: {len(all_queries)} queries "
          f"({len(labeled)} manual + {len(synthetic)} synthetic)")
    print(f"Model: LogisticRegression, {config['num_classes']} classes")
    print(f"Size: {file_size:,} bytes (~{file_size/1024:.0f} KB)")
    print(f"Accuracy: {config['accuracy']:.1%}")
    print(f"\nTo use: set INTENT_CLASSIFIER_PATH={args.output}")


if __name__ == "__main__":
    main()
