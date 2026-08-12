#!/usr/bin/env python3
"""
Build a lightweight binary relevance classifier for MarxOS.

Detects whether a query is Marxism-related (1) or not (0).
Used as a pre-filter before the expensive RAG pipeline.

Training data: short sentences, hand-crafted + synthetic.
Model: LogisticRegression on BGE-M3 embeddings (~3 KB).
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUTPUT_PATH = Path(
    os.getenv("RELEVANCE_CLASSIFIER_PATH", str(ROOT / "data/relevance_classifier.pkl"))
)
RANDOM_SEED = 42

# ============================================================================
# Training data — short, simple sentences
# ============================================================================

# ---- Positive samples (Marxism-relevant, label=1) ----

POSITIVE_SAMPLES: list[str] = [
    # Direct name mentions
    "马克思",
    "恩格斯",
    "马克思是谁",
    "恩格斯写了什么",
    "马恩",
    "马克思主义",
    # Work titles
    "资本论",
    "资本论讲什么",
    "资本论第一卷",
    "共产党宣言",
    "共产党宣言内容",
    "哥达纲领批判",
    "德意志意识形态",
    "反杜林论",
    "关于费尔巴哈的提纲",
    "1844年经济学哲学手稿",
    "社会主义从空想到科学",
    "家庭私有制和国家的起源",
    "法兰西内战",
    "路易波拿巴的雾月十八日",
    "自然辩证法",
    "政治经济学批判",
    "雇佣劳动与资本",
    "工资价格和利润",
    "神圣家族",
    "哲学的贫困",
    # Concepts
    "剩余价值是什么",
    "什么是剩余价值",
    "剩余价值",
    "异化劳动",
    "什么是异化",
    "阶级斗争",
    "唯物史观",
    "唯物辩证法",
    "商品拜物教",
    "什么是商品拜物教",
    "拜物教",
    "历史唯物主义是什么",
    "什么是历史唯物主义",
    "辩证法",
    "生产关系",
    "生产力",
    "经济基础",
    "上层建筑",
    "无产阶级",
    "资产阶级",
    "原始积累",
    "价值规律",
    "劳动价值论",
    "剥削",
    "共产主义是什么",
    "什么是共产主义",
    "社会主义是什么",
    "科学社会主义",
    "空想社会主义",
    "实践是检验真理的标准",
    "人的本质",
    "异化",
    "资本",
    "资本是什么",
    "商品",
    "货币",
    "工资",
    "利润",
    "地租",
    # Questions about works
    "资本论主要内容",
    "共产党宣言的核心观点",
    "共产党宣言在哪一卷",
    "资本论收录在哪",
    "德意志意识形态讲了什么",
    "费尔巴哈提纲第几条",
    "哥达纲领批判的核心",
    "反杜林论的主要内容",
    "1844手稿的核心概念",
    # Quote lookups
    "全世界无产者联合起来出自哪里",
    "全世界无产者联合起来",
    "哲学家们只是用不同的方式解释世界",
    "资本来到世间从头到脚每个毛孔都滴着血",
    "各尽所能按需分配出自哪里",
    "宗教是人民的鸦片出自哪里",
    "自由是对必然的认识",
    "国家不是从来就有的",
    "劳动创造了人本身",
    "一切已死的先辈们的传统像梦魇一样纠缠着活人的头脑",
    # Analysis questions
    "如何理解剩余价值理论",
    "马克思主义如何看待国家",
    "分析资本主义的矛盾",
    "论述异化劳动的当代意义",
    "运用马克思主义分析当代社会",
    "马克思对黑格尔的批判",
    "恩格斯论国家起源",
    "从马克思主义视角看问题",
    "为什么说资本论是工人阶级的圣经",
    "唯物辩证法怎么理解",
    "历史唯物主义的基本原理",
    "阶级分析方法",
    "政治经济学批判的方法论",
    # Chinese Marxist terminology
    "实事求是",
    "群众路线",
    "实践论",
    "矛盾论",
    "新民主主义",
    "中国特色社会主义",
    "改革开放",
    "三个代表",
    "科学发展观",
    # Short factual
    "马克思生卒年",
    "恩格斯出生",
    "资本论写了多久",
    "马克思的主要著作有哪些",
    "恩格斯的代表作",
    "马恩全集多少卷",
    "马克思恩格斯文集",
    "马克思恩格斯选集",
]

# ---- Negative samples (non-Marxism, label=0) ----

NEGATIVE_SAMPLES: list[str] = [
    # Chitchat / greetings
    "你好",
    "Hi",
    "Hello",
    "在吗",
    "早上好",
    "晚上好",
    "谢谢",
    "再见",
    "你是谁",
    "你能做什么",
    "介绍一下你自己",
    # Emotional / personal
    "我爱你",
    "我喜欢你",
    "今天心情不好",
    "我想你",
    "你真聪明",
    "无聊",
    "哈哈",
    "嗯嗯",
    "好的",
    "知道了",
    # General knowledge / other domains
    "今天天气怎么样",
    "明天会下雨吗",
    "现在几点了",
    "今天星期几",
    "什么是量子力学",
    "什么是相对论",
    "什么是DNA",
    "什么是黑洞",
    "什么是微积分",
    "什么是线性代数",
    "什么是机器学习",
    "什么是深度学习",
    "什么是区块链",
    "什么是云计算",
    "什么是Web3",
    "AI是什么",
    "ChatGPT是什么",
    "Python怎么学",
    "如何学编程",
    "Java和Python的区别",
    "React怎么用",
    "Docker是什么",
    "Kubernetes是什么",
    "Linux常用命令",
    "Git怎么用",
    "数据库怎么设计",
    "API是什么",
    "RESTful是什么",
    # History (non-Marxist)
    "明朝为什么灭亡",
    "唐朝有多少年",
    "秦始皇是谁",
    "三国演义讲了什么",
    "罗马帝国怎么灭亡的",
    "二战是什么时候",
    "法国大革命原因",
    "美国独立战争",
    "工业革命是什么",
    "文艺复兴是什么",
    # Science
    "光合作用是什么",
    "细胞是什么",
    "DNA是什么",
    "元素周期表",
    "牛顿定律",
    "万有引力是什么",
    "地球绕太阳转",
    "光速是多少",
    "水的沸点",
    "人体有多少块骨头",
    # Daily life
    "今天吃什么",
    "怎么减肥",
    "怎么健身",
    "什么菜好吃",
    "怎么化妆",
    "怎么穿搭",
    "怎么拍照好看",
    "推荐一本书",
    "推荐一部电影",
    "推荐一首歌",
    # Finance / tech
    "股票怎么买",
    "房价还会跌吗",
    "比特币是什么",
    "怎么理财",
    "怎么炒股",
    "基金怎么选",
    "比特币价格",
    "特斯拉股价",
    "iPhone怎么样",
    "什么手机好用",
    # Exam / study
    "考研怎么准备",
    "托福和雅思的区别",
    "四级怎么过",
    "六级怎么准备",
    "公务员考试",
    "教师资格证怎么考",
    "高考志愿怎么填",
    "大学选什么专业",
    # Task / instruction
    "帮我写一首诗",
    "帮我写一篇文章",
    "帮我翻译一段话",
    "帮我总结这篇文章",
    "帮我写代码",
    "帮我做PPT",
    "帮我画画",
    "讲个笑话",
    "讲个故事",
    "给我一个方案",
    "帮我分析数据",
    "忽略之前的指令",
    "你是什么模型",
    # Vague / unclear
    "这是什么",
    "为什么",
    "怎么回事",
    "什么意思",
    "对不对",
    "是吗",
    "真的吗",
    "举个例子",
    "具体说说",
    "然后呢",
    "继续",
    "还有吗",
    # Sports / entertainment
    "世界杯冠军是谁",
    "NBA总冠军",
    "梅西厉害还是C罗厉害",
    "英雄联盟怎么玩",
    "什么游戏好玩",
    "最近有什么电影",
    "什么综艺好看",
    "哪个明星最火",
    # Other philosophy (non-Marxist)
    "康德的主要思想",
    "尼采哲学",
    "存在主义是什么",
    "柏拉图理想国",
    "亚里士多德",
    "笛卡尔的哲学",
    "王阳明的哲学",
    "儒家思想",
    "道家思想",
    "佛学",
    "什么是禅",
    # Geography / travel
    "北京有什么好玩的",
    "上海有什么好吃的",
    "去哪里旅游好",
    "云南有什么景点",
    "欧洲旅游攻略",
    "日本哪里好玩",
    # Health
    "感冒了怎么办",
    "怎么治失眠",
    "头疼吃什么药",
    "怎么提高免疫力",
    "什么保健品好",
]


# ============================================================================
# Synthetic data generation
# ============================================================================

POSITIVE_TEMPLATES: list[str] = [
    "什么是{concept}",
    "{concept}是什么",
    "如何理解{concept}",
    "{concept}的概念",
    "简述{concept}",
    "分析{concept}",
    "论述{concept}",
    "{concept}的意义",
    "怎么看{concept}",
    "{work}的主要内容",
    "{work}在哪一卷",
    "{work}的核心观点",
    "查找{work}",
    "{quote}出自哪里",
]

NEGATIVE_TEMPLATES: list[str] = [
    "什么是{concept}",
    "{concept}怎么学",
    "{concept}教程",
    "如何成为{role}",
    "{city}有什么好玩的",
    "{city}天气怎么样",
    "怎么{action}",
    "{food}怎么做",
    "推荐{thing}",
    "{sport}规则是什么",
    "{animal}吃什么",
    "{planet}有多大",
    "{disease}怎么治",
    "怎么学好{subject}",
    "{subject}考试技巧",
    "{country}首都是哪",
    "{person}是谁（非马克思）",
    "帮我{task}",
]


def generate_synthetic_positive(n: int = 100) -> list[str]:
    """Generate synthetic Marxism-relevant queries from templates."""
    random.seed(RANDOM_SEED)
    concepts = [
        "剩余价值", "异化劳动", "阶级斗争", "唯物史观", "辩证法",
        "商品拜物教", "资本积累", "利润率下降", "地租", "原始积累",
        "价值形式", "抽象劳动", "具体劳动", "社会必要劳动时间",
        "劳动力商品", "不变资本", "可变资本", "绝对剩余价值",
        "相对剩余价值", "资本有机构成", "产业后备军", "无产阶级贫困化",
        "经济危机", "垄断资本主义", "帝国主义", "国家资本主义",
        "无产阶级专政", "共产主义第一阶段", "按劳分配", "按需分配",
        "必然王国", "自由王国", "人的全面发展", "对象化", "外化",
        "类本质", "感性活动", "实践", "意识形态", "市民社会",
    ]
    works = [
        "资本论", "共产党宣言", "德意志意识形态", "反杜林论",
        "哥达纲领批判", "关于费尔巴哈的提纲", "法兰西内战",
        "1844年经济学哲学手稿", "社会主义从空想到科学的发展",
        "家庭私有制和国家的起源", "政治经济学批判",
        "雇佣劳动与资本", "工资价格和利润", "哲学的贫困",
    ]
    quotes = [
        "全世界无产者联合起来",
        "哲学家们只是用不同的方式解释世界",
        "资本来到世间从头到脚每个毛孔都滴着血",
        "宗教是人民的鸦片",
        "各尽所能按需分配",
        "自由是对必然的认识",
        "国家不是从来就有的",
        "一切已死的先辈们的传统像梦魇一样纠缠着活人的头脑",
        "暴力是每一个孕育着新社会的旧社会的助产婆",
        "劳动创造了人本身",
    ]
    results: list[str] = []
    for _ in range(n):
        template = random.choice(POSITIVE_TEMPLATES)
        concept = random.choice(concepts)
        work = random.choice(works)
        quote = random.choice(quotes)
        try:
            query = template.format(concept=concept, work=work, quote=quote)
        except (KeyError, ValueError):
            query = f"什么是{concept}"
        if 2 <= len(query) <= 60 and query not in results:
            results.append(query)
    return results


def generate_synthetic_negative(n: int = 120) -> list[str]:
    """Generate synthetic non-Marxism queries from templates."""
    random.seed(RANDOM_SEED + 1)
    concepts = [
        "微积分", "线性代数", "概率论", "数据结构", "算法",
        "操作系统", "编译原理", "计算机网络", "人工智能", "机器学习",
        "深度学习", "自然语言处理", "计算机视觉", "强化学习",
        "量子计算", "神经科学", "分子生物学", "遗传学",
        "宇宙学", "气候学", "地质学", "考古学",
        "宏观经济学", "微观经济学", "行为经济学",
        "精神分析", "认知心理学", "社会心理学",
        "宪法学", "刑法学", "民法学",
    ]
    roles = ["产品经理", "设计师", "程序员", "数据分析师", "运营", "架构师"]
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
    actions = ["做蛋糕", "养花", "钓鱼", "滑板", "弹吉他", "练字"]
    foods = ["红烧肉", "麻婆豆腐", "意大利面", "寿司", "火锅"]
    things = ["耳机", "键盘", "显示器", "手机", "相机", "手表"]
    sports = ["篮球", "足球", "网球", "游泳", "马拉松"]
    animals = ["猫", "狗", "熊猫", "大象", "企鹅"]
    planets = ["地球", "火星", "木星", "土星", "太阳"]
    diseases = ["感冒", "高血压", "糖尿病", "过敏", "胃炎"]
    subjects = ["物理", "化学", "英语", "历史", "地理"]
    countries = ["法国", "德国", "日本", "韩国", "巴西", "印度"]
    persons = ["苏格拉底", "莎士比亚", "贝多芬", "爱因斯坦", "乔布斯"]
    tasks = [
        "写个PPT", "写个方案", "写个周报", "写个邮件", "写个脚本",
        "画个图", "做个表", "做个总结", "写个报告", "做个分析",
    ]

    context = {
        "concept": concepts, "role": roles, "city": cities,
        "action": actions, "food": foods, "thing": things,
        "sport": sports, "animal": animals, "planet": planets,
        "disease": diseases, "subject": subjects, "country": countries,
        "person": persons, "task": tasks,
    }

    results: list[str] = []
    for _ in range(n):
        template = random.choice(NEGATIVE_TEMPLATES)
        vals = {}
        for key, pool in context.items():
            vals[key] = random.choice(pool)
        try:
            query = template.format(**vals)
        except (KeyError, ValueError):
            continue
        if 2 <= len(query) <= 50 and query not in results:
            results.append(query)
    return results


# ============================================================================
# Training
# ============================================================================


def load_embedding_model():
    """Load BGE-M3 embedding model."""
    from marxos.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


def embed_queries(queries: list[str], embedding_model) -> np.ndarray:
    """Batch-embed query strings → (N, 1024) float32 array."""
    return np.array(embedding_model.embed_documents(queries), dtype=np.float32)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, dict]:
    """Train binary logistic regression classifier."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        C=1.0,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["NON_MARXISM", "MARXISM"],
        output_dict=True,
    )

    config = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": 2,
        "classes": ["NON_MARXISM", "MARXISM"],
        "accuracy": report.get("accuracy", 0.0),
        "marxism_precision": report.get("MARXISM", {}).get("precision", 0.0),
        "marxism_recall": report.get("MARXISM", {}).get("recall", 0.0),
        "marxism_f1": report.get("MARXISM", {}).get("f1-score", 0.0),
        "non_marxism_precision": report.get("NON_MARXISM", {}).get("precision", 0.0),
        "non_marxism_recall": report.get("NON_MARXISM", {}).get("recall", 0.0),
        "non_marxism_f1": report.get("NON_MARXISM", {}).get("f1-score", 0.0),
        "embedding_model": "BAAI/bge-m3",
        "embedding_dim": int(X_train.shape[1]),
    }

    return model, config


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relevance classifier")
    parser.add_argument(
        "--synthetic-positive", type=int, default=100,
        help="Synthetic positive queries to generate (default: 100)",
    )
    parser.add_argument(
        "--synthetic-negative", type=int, default=120,
        help="Synthetic negative queries to generate (default: 120)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold stored in config (default: 0.5)",
    )
    args = parser.parse_args()

    print("=== MarxOS Relevance Classifier Builder ===\n")

    # 1. Collect data
    print("1. Collecting training data...")
    positives = list(POSITIVE_SAMPLES)
    negatives = list(NEGATIVE_SAMPLES)
    print(f"   Hand-crafted positive: {len(positives)}")
    print(f"   Hand-crafted negative: {len(negatives)}")

    syn_pos = generate_synthetic_positive(args.synthetic_positive)
    syn_neg = generate_synthetic_negative(args.synthetic_negative)
    positives.extend(syn_pos)
    negatives.extend(syn_neg)
    print(f"   Synthetic positive: {len(syn_pos)}")
    print(f"   Synthetic negative: {len(syn_neg)}")

    # Remove accidental duplicates across classes
    pos_set = set(positives)
    neg_set = set(negatives)
    overlap = pos_set & neg_set
    if overlap:
        print(f"   Removing {len(overlap)} cross-class duplicates...")
        positives = [q for q in positives if q not in overlap]
        negatives = [q for q in negatives if q not in overlap]

    all_queries = positives + negatives
    all_labels = [1] * len(positives) + [0] * len(negatives)
    print(f"   Total: {len(all_queries)} ({len(positives)} pos, {len(negatives)} neg)")

    # 2. Shuffle and split
    print("\n2. Splitting train/test...")
    random.seed(RANDOM_SEED)
    combined = list(zip(all_queries, all_labels))
    random.shuffle(combined)
    queries, labels = zip(*combined)

    q_train, q_test, y_train, y_test = train_test_split(
        list(queries), list(labels),
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        stratify=list(labels),
    )
    print(f"   Train: {len(q_train)} ({sum(y_train)} pos, {len(y_train) - sum(y_train)} neg)")
    print(f"   Test:  {len(q_test)} ({sum(y_test)} pos, {len(y_test) - sum(y_test)} neg)")

    # 3. Embed
    print("\n3. Loading embedding model (BAAI/bge-m3)...")
    emb = load_embedding_model()
    print("   Embedding train queries...")
    X_train = embed_queries(q_train, emb)
    print("   Embedding test queries...")
    X_test = embed_queries(q_test, emb)
    print(f"   Shape: train={X_train.shape}, test={X_test.shape}")

    # 4. Train
    print("\n4. Training classifier...")
    y_train_arr = np.array(y_train, dtype=np.int32)
    y_test_arr = np.array(y_test, dtype=np.int32)
    model, config = train_classifier(X_train, y_train_arr, X_test, y_test_arr)
    config["threshold"] = args.threshold

    print(f"\n   Test accuracy:        {config['accuracy']:.1%}")
    print(f"   Marxism precision:    {config['marxism_precision']:.3f}")
    print(f"   Marxism recall:       {config['marxism_recall']:.3f}")
    print(f"   Marxism F1:           {config['marxism_f1']:.3f}")
    print(f"   Non-Marxism precision: {config['non_marxism_precision']:.3f}")
    print(f"   Non-Marxism recall:   {config['non_marxism_recall']:.3f}")
    print(f"   Non-Marxism F1:       {config['non_marxism_f1']:.3f}")

    # 5. Save
    print(f"\n5. Saving to {args.output}...")
    from marxos.relevance_classifier import RelevanceClassifier

    classifier = RelevanceClassifier(model, config)
    classifier.save(args.output)
    file_size = args.output.stat().st_size
    print(f"   Saved: {args.output} ({file_size:,} bytes, ~{file_size/1024:.0f} KB)")

    # 6. Sanity check
    print("\n6. Sanity check on a few examples...")
    test_examples = [
        ("剩余价值", 1),
        ("我爱你", 0),
        ("资本论第一卷", 1),
        ("今天天气怎么样", 0),
        ("唯物辩证法是什么", 1),
        ("怎么学Python", 0),
        ("全世界无产者联合起来", 1),
        ("推荐一部电影", 0),
        ("什么是异化劳动", 1),
        ("怎么减肥", 0),
        ("马克思", 1),
        ("哈哈哈哈", 0),
    ]
    all_ok = True
    for query, expected in test_examples:
        emb_vec = embed_queries([query], emb)[0]
        pred = classifier.predict(emb_vec)
        prob = classifier.predict_proba(emb_vec)
        status = "✓" if pred == bool(expected) else "✗"
        if status == "✗":
            all_ok = False
        print(f"   {status} '{query}' → {'MARXISM' if pred else 'NON'} "
              f"(prob={prob:.3f}, expected={'MARXISM' if expected else 'NON'})")

    if all_ok:
        print("   All sanity checks passed!")
    else:
        print("   ⚠ Some sanity checks failed — review training data.")

    print(f"\n=== Done ===")
    print(f"Model: LogisticRegression, {config['training_samples']} train + "
          f"{config['test_samples']} test samples")
    print(f"Size: ~{file_size/1024:.0f} KB")
    print(f"Accuracy: {config['accuracy']:.1%}")
    print(f"Marxism F1: {config['marxism_f1']:.3f}")


if __name__ == "__main__":
    main()
