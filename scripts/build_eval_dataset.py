"""
Build expanded evaluation dataset with work-level annotations.

Outputs eval_dataset_v2.json with 120 questions covering:
  - 哲学 (philosophy): ~40 questions
  - 政治经济学 (political_economy): ~40 questions
  - 科学社会主义 (scientific_socialism): ~40 questions

Each entry:
  - id, question, question_type
  - expected_work_id (from work_catalog.json)
  - expected_source (primary PDF)
  - discipline
  - difficulty: easy/medium/hard
  - hard_negative: [wrong works that plausibly match]
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load work_catalog to validate work_ids
with open(ROOT / "rag/work_catalog.json", encoding="utf-8") as f:
    wc = json.load(f)
work_ids = {w["work_id"] for w in wc["works"]}
work_titles = {w["title"]: w["work_id"] for w in wc["works"]}

# ── Question definitions ────────────────────────────────────────

def Q(qid, question, qtype, expected_work_id, discipline, difficulty="medium",
      hard_negative=None, notes=None):
    """Define a test question with work-level annotations."""
    assert expected_work_id in work_ids, f"Unknown work_id: {expected_work_id}"
    entry = {
        "id": qid,
        "question": question,
        "question_type": qtype,
        "expected_work_id": expected_work_id,
        "discipline": discipline,
        "difficulty": difficulty,
    }
    if hard_negative:
        for hn in hard_negative:
            assert hn in work_ids, f"Unknown hard_negative: {hn}"
        entry["hard_negative"] = hard_negative
    if notes:
        entry["notes"] = notes
    return entry


questions = []

# ═══════════════════════════════════════════════════════════════════
# 马克思主义哲学 (40 questions)
# ═══════════════════════════════════════════════════════════════════

pid = 1

# ── 关于费尔巴哈的提纲 ──
questions.append(Q(pid, "哲学家们只是用不同的方式解释世界，问题在于改变世界。",
    "quote_lookup", "theses-feuerbach", "philosophy", "easy",
    ["german-ideology", "communist-manifesto"], "高频误检句")); pid += 1

questions.append(Q(pid, "人的本质不是单个人所固有的抽象物，在其现实性上，它是一切社会关系的总和。",
    "quote_lookup", "theses-feuerbach", "philosophy", "easy",
    ["german-ideology"])); pid += 1

questions.append(Q(pid, "费尔巴哈的实践观是怎么样的？",
    "concept_explain", "theses-feuerbach", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "马克思如何理解实践这个概念？",
    "concept_explain", "theses-feuerbach", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "《关于费尔巴哈的提纲》中马克思批判了费尔巴哈唯物主义的什么缺陷？",
    "analysis", "theses-feuerbach", "philosophy", "easy")); pid += 1

# ── 德意志意识形态 ──
questions.append(Q(pid, "意识在任何时候都只能是被意识到了的存在。",
    "quote_lookup", "german-ideology", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "不是意识决定生活，而是生活决定意识。",
    "quote_lookup", "german-ideology", "philosophy", "easy",
    ["preface-critique-political-economy"], "易与《政治经济学批判序言》混淆")); pid += 1

questions.append(Q(pid, "什么是历史唯物主义的基本原理？",
    "concept_explain", "german-ideology", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "马克思和恩格斯在《德意志意识形态》中如何论述分工与所有制的关系？",
    "analysis", "german-ideology", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "《德意志意识形态》中'消灭分工'的含义是什么？",
    "concept_explain", "german-ideology", "philosophy", "hard")); pid += 1

# ── 1844年经济学哲学手稿 ──
questions.append(Q(pid, "马克思在《1844年经济学哲学手稿》中如何论述异化劳动的四个规定？",
    "analysis", "economic-philosophic-manuscripts-1844", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "异化劳动这个概念在马克思著作中是什么意思？",
    "concept_explain", "economic-philosophic-manuscripts-1844", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "马克思在手稿中如何批判黑格尔的辩证法？",
    "analysis", "economic-philosophic-manuscripts-1844", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "《1844年手稿》中'劳动为富人生产了奇迹般的东西，但是为工人生产了赤贫'的含义",
    "quote_lookup", "economic-philosophic-manuscripts-1844", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "马克思的'人的本质'思想在《1844年手稿》和《关于费尔巴哈的提纲》中有何不同？",
    "analysis", "economic-philosophic-manuscripts-1844", "philosophy", "hard")); pid += 1

# ── 《黑格尔法哲学批判》导言 ──
questions.append(Q(pid, "宗教是人民的鸦片。",
    "quote_lookup", "critique-hegel-law-intro", "philosophy", "easy",
    ["german-ideology", "on-the-jewish-question"], "极易误检")); pid += 1

questions.append(Q(pid, "批判的武器当然不能代替武器的批判，物质力量只能用物质力量来摧毁。",
    "quote_lookup", "critique-hegel-law-intro", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "马克思在《〈黑格尔法哲学批判〉导言》中如何论述无产阶级的历史使命？",
    "analysis", "critique-hegel-law-intro", "philosophy", "medium")); pid += 1

# ── 路德维希·费尔巴哈论 ──
questions.append(Q(pid, "全部哲学，特别是近代哲学的重大的基本问题，是思维和存在的关系问题。",
    "quote_lookup", "ludwig-feuerbach", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "恩格斯如何论述哲学基本问题？",
    "concept_explain", "ludwig-feuerbach", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "恩格斯在《路德维希·费尔巴哈和德国古典哲学的终结》中如何评价黑格尔哲学？",
    "analysis", "ludwig-feuerbach", "philosophy", "medium")); pid += 1

# ── 反杜林论 + 自然辩证法 ──
questions.append(Q(pid, "恩格斯在《反杜林论》中如何阐述唯物辩证法？",
    "analysis", "anti-duhring", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "《反杜林论》哲学编的主要内容是什么？",
    "analysis", "anti-duhring", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "劳动创造了人本身。",
    "quote_lookup", "dialectics-nature", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "恩格斯在《自然辩证法》中如何论述劳动在从猿到人转变过程中的作用？",
    "analysis", "dialectics-nature", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "《自然辩证法》中关于物质运动形式的论述",
    "analysis", "dialectics-nature", "philosophy", "hard")); pid += 1

# ── 神圣家族 ──
questions.append(Q(pid, "历史活动是群众的活动，随着历史活动的深入，必将是群众队伍的扩大。",
    "quote_lookup", "holy-family", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "《神圣家族》中马克思和恩格斯如何批判青年黑格尔派？",
    "analysis", "holy-family", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "《神圣家族》对法国唯物主义的论述",
    "analysis", "holy-family", "philosophy", "hard")); pid += 1

# ── 哲学的贫困 ──
questions.append(Q(pid, "《哲学的贫困》中马克思如何批判蒲鲁东的经济学方法？",
    "analysis", "poverty-philosophy", "philosophy", "hard")); pid += 1

# ── 《政治经济学批判》序言 ──
questions.append(Q(pid, "不是人们的意识决定人们的存在，而是人们的社会存在决定人们的意识。",
    "quote_lookup", "preface-critique-political-economy", "philosophy", "easy",
    ["german-ideology"], "易与《德意志意识形态》混淆")); pid += 1

questions.append(Q(pid, "物质生活的生产方式制约着整个社会生活、政治生活和精神生活的过程。",
    "quote_lookup", "preface-critique-political-economy", "philosophy", "easy")); pid += 1

questions.append(Q(pid, "马克思在《〈政治经济学批判〉序言》中对历史唯物主义的经典表述",
    "analysis", "preface-critique-political-economy", "philosophy", "easy")); pid += 1

# ── 概念类（跨著作） ──
questions.append(Q(pid, "马克思的意识形态概念是如何形成的？",
    "concept_explain", "german-ideology", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "什么是唯物辩证法？",
    "concept_explain", "anti-duhring", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "马克思如何理解社会存在和社会意识的关系？",
    "concept_explain", "preface-critique-political-economy", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "马克思的共产主义思想与空想社会主义有何区别？",
    "analysis", "communist-manifesto", "philosophy", "medium")); pid += 1

questions.append(Q(pid, "如何理解'人也按照美的规律来构造'？",
    "quote_lookup", "economic-philosophic-manuscripts-1844", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "恩格斯在《国民经济学批判大纲》中如何批判私有制？",
    "analysis", "outlines-critique-political-economy", "philosophy", "hard")); pid += 1

questions.append(Q(pid, "马克思对宗教批判的核心观点是什么？",
    "concept_explain", "critique-hegel-law-intro", "philosophy", "easy")); pid += 1


# ═══════════════════════════════════════════════════════════════════
# 政治经济学 (40 questions)
# ═══════════════════════════════════════════════════════════════════

# ── 资本论 第一卷 ──
questions.append(Q(pid, "资本来到世间，从头到脚，每个毛孔都滴着血和肮脏的东西。",
    "quote_lookup", "capital-vol1", "political_economy", "easy",
    ["wage-labour-capital"])); pid += 1

questions.append(Q(pid, "商品是天生的平等派。",
    "quote_lookup", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思在哪里论述了商品拜物教？",
    "bibliographic", "capital-vol1", "political_economy", "easy",
    ["economic-philosophic-manuscripts-1844"])); pid += 1

questions.append(Q(pid, "什么是商品拜物教？",
    "concept_explain", "capital-vol1", "political_economy", "easy")); pid += 1

questions.append(Q(pid, "马克思的劳动二重性理论",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "剩余价值是怎么产生的？",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "绝对剩余价值和相对剩余价值的区别是什么？",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思如何论述资本原始积累？",
    "analysis", "capital-vol1", "political_economy", "easy")); pid += 1

questions.append(Q(pid, "什么是资本的有机构成？",
    "concept_explain", "capital-vol1", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "马克思在《资本论》中如何论述价值形式的发展？",
    "analysis", "capital-vol1", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "资本主义积累的一般规律是什么？",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思如何理解'货币转化为资本'？",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "《资本论》第一卷中工作日一章的主要内容",
    "analysis", "capital-vol1", "political_economy", "hard")); pid += 1

# ── 资本论 第二卷 ──
questions.append(Q(pid, "马克思如何分析资本的循环和周转？",
    "analysis", "capital-vol2", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "社会总资本的再生产理论",
    "concept_explain", "capital-vol2", "political_economy", "hard")); pid += 1

# ── 资本论 第三卷 ──
questions.append(Q(pid, "马克思的平均利润率理论",
    "concept_explain", "capital-vol3", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "马克思的地租理论",
    "analysis", "capital-vol3", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "利润率趋向下降的规律",
    "concept_explain", "capital-vol3", "political_economy", "hard")); pid += 1

# ── 雇佣劳动与资本 ──
questions.append(Q(pid, "什么是雇佣劳动？",
    "concept_explain", "wage-labour-capital", "political_economy", "easy")); pid += 1

questions.append(Q(pid, "《雇佣劳动与资本》中工资的本质是什么？",
    "analysis", "wage-labour-capital", "political_economy", "medium")); pid += 1

# ── 工资、价格和利润 ──
questions.append(Q(pid, "工资和利润的关系是怎样的？",
    "concept_explain", "value-price-profit", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思在《工资、价格和利润》中如何论述工人阶级争取提高工资的斗争？",
    "analysis", "value-price-profit", "political_economy", "medium")); pid += 1

# ── 经济学手稿 ──
questions.append(Q(pid, "马克思在《1857-1858年经济学手稿》中如何论述前资本主义所有制形式？",
    "analysis", "grundrisse-selections", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "马克思如何区分生产劳动和非生产劳动？",
    "concept_explain", "manuscripts-1861-1863-selections", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "马克思在《政治经济学批判》导言中如何论述生产与消费的关系？",
    "analysis", "grundrisse-introduction", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "《1861-1863年手稿》中关于机器的论述",
    "analysis", "manuscripts-1861-1863-selections", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "'资本主义生产以前的各种形式'出自哪部著作？",
    "bibliographic", "grundrisse-selections", "political_economy", "medium")); pid += 1

# ── 其他政治经济学著作 ──
questions.append(Q(pid, "关于自由贸易问题的演说中马克思的核心观点",
    "analysis", "speech-free-trade", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "恩格斯如何论述保护关税和自由贸易？",
    "analysis", "protection-and-free-trade", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "国民经济学批判大纲中恩格斯的政治经济学批判",
    "analysis", "outlines-critique-political-economy", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思对法国动产信用公司的分析",
    "analysis", "credit-mobilier", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "什么是价值规律？",
    "concept_explain", "capital-vol1", "political_economy", "easy")); pid += 1

questions.append(Q(pid, "马克思如何分析经济危机？",
    "analysis", "manuscripts-1861-1863-selections", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "《资本论》中'原始积累的秘密'指的是什么？",
    "concept_explain", "capital-vol1", "political_economy", "easy")); pid += 1

questions.append(Q(pid, "马克思的剩余价值理论的主要内容",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "恩格斯在《国民经济学批判大纲》中对马尔萨斯人口论的批判",
    "analysis", "outlines-critique-political-economy", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "《哲学的贫困》中马克思的政治经济学方法",
    "analysis", "poverty-philosophy", "political_economy", "hard")); pid += 1

questions.append(Q(pid, "'资本是死劳动，它像吸血鬼一样，只有吮吸活劳动才有生命'出自哪里？",
    "bibliographic", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "马克思的劳动价值论的主要内容",
    "concept_explain", "capital-vol1", "political_economy", "medium")); pid += 1

questions.append(Q(pid, "资本主义生产的总过程",
    "analysis", "capital-vol3", "political_economy", "hard")); pid += 1


# ═══════════════════════════════════════════════════════════════════
# 科学社会主义 (40 questions)
# ═══════════════════════════════════════════════════════════════════

# ── 共产党宣言 ──
questions.append(Q(pid, "一个幽灵，共产主义的幽灵，在欧洲游荡。",
    "quote_lookup", "communist-manifesto", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "全世界无产者，联合起来！",
    "quote_lookup", "communist-manifesto", "scientific_socialism", "easy",
    ["principles-communism", "capital-vol1"])); pid += 1

questions.append(Q(pid, "每个人的自由发展是一切人的自由发展的条件。",
    "quote_lookup", "communist-manifesto", "scientific_socialism", "easy",
    ["critique-gotha-programme"])); pid += 1

questions.append(Q(pid, "共产党宣言的主要内容是什么？",
    "analysis", "communist-manifesto", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "《共产党宣言》中关于阶级斗争的论述",
    "analysis", "communist-manifesto", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "《共产党宣言》中'工人没有祖国'的含义是什么？",
    "concept_explain", "communist-manifesto", "scientific_socialism", "medium")); pid += 1

# ── 法兰西阶级斗争 + 雾月十八日 ──
questions.append(Q(pid, "马克思关于无产阶级专政的论述出自哪部著作？",
    "bibliographic", "class-struggles-france", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "人们自己创造自己的历史，但并不是随心所欲地创造。",
    "quote_lookup", "eighteenth-brumaire", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "马克思在《路易·波拿巴的雾月十八日》中如何分析国家机器？",
    "analysis", "eighteenth-brumaire", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "马克思如何分析法国农民阶级？",
    "analysis", "eighteenth-brumaire", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "'革命是历史的火车头'出自哪部著作？",
    "bibliographic", "class-struggles-france", "scientific_socialism", "medium")); pid += 1

# ── 法兰西内战 + 巴黎公社 ──
questions.append(Q(pid, "工人阶级不能简单地掌握现成的国家机器，并运用它来达到自己的目的。",
    "quote_lookup", "civil-war-france", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "马克思如何总结巴黎公社的经验教训？",
    "analysis", "civil-war-france", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "《法兰西内战》中关于打碎旧国家机器的论述",
    "analysis", "civil-war-france", "scientific_socialism", "medium")); pid += 1

# ── 哥达纲领批判 ──
questions.append(Q(pid, "各尽所能，按需分配。",
    "quote_lookup", "critique-gotha-programme", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "马克思在《哥达纲领批判》中如何论述共产主义社会的两个阶段？",
    "analysis", "critique-gotha-programme", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "过渡时期的国家只能是无产阶级的革命专政",
    "quote_lookup", "critique-gotha-programme", "scientific_socialism", "medium")); pid += 1

# ── 社会主义从空想到科学的发展 ──
questions.append(Q(pid, "恩格斯如何论述社会主义从空想到科学的发展？",
    "analysis", "socialism-utopian-scientific", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "科学社会主义与空想社会主义的根本区别是什么？",
    "concept_explain", "socialism-utopian-scientific", "scientific_socialism", "medium")); pid += 1

# ── 家庭、私有制和国家的起源 ──
questions.append(Q(pid, "恩格斯如何论述国家的起源？",
    "concept_explain", "origin-family-private-property-state", "scientific_socialism", "easy")); pid += 1

questions.append(Q(pid, "《家庭、私有制和国家的起源》中关于妇女解放的论述",
    "analysis", "origin-family-private-property-state", "scientific_socialism", "medium")); pid += 1

# ── 德国农民战争 / 德国革命 ──
questions.append(Q(pid, "恩格斯在《德国农民战争》中如何分析宗教改革时期的阶级关系？",
    "analysis", "peasant-war-germany", "scientific_socialism", "hard")); pid += 1

questions.append(Q(pid, "《德国的革命和反革命》的主要内容",
    "analysis", "revolution-counterrevolution-germany", "scientific_socialism", "hard")); pid += 1

# ── 法德农民问题 ──
questions.append(Q(pid, "恩格斯在《法德农民问题》中如何论述农民合作社？",
    "analysis", "peasant-question-france-germany", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "恩格斯关于工农联盟的思想",
    "concept_explain", "peasant-question-france-germany", "scientific_socialism", "medium")); pid += 1

# ── 论权威 / 论住宅问题 ──
questions.append(Q(pid, "恩格斯为什么说权威在任何社会都是必要的？",
    "concept_explain", "on-authority", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "恩格斯的《论住宅问题》主要批判了哪种观点？",
    "analysis", "housing-question", "scientific_socialism", "hard")); pid += 1

# ── 国际工人协会 ──
questions.append(Q(pid, "《国际工人协会成立宣言》的主要内容",
    "analysis", "inaugural-address-iwa", "scientific_socialism", "hard")); pid += 1

questions.append(Q(pid, "《国际工人协会共同章程》的核心原则",
    "analysis", "inaugural-address-iwa", "scientific_socialism", "hard")); pid += 1

# ── 论殖民主义/东方问题 ──
questions.append(Q(pid, "马克思如何论述英国对印度的殖民统治？",
    "analysis", "british-rule-india", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "马克思论鸦片贸易",
    "analysis", "opium-trade", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "马克思如何分析中国革命和欧洲革命的关系？",
    "analysis", "revolution-china-europe", "scientific_socialism", "medium")); pid += 1

# ── 恩格斯晚年著作 ──
questions.append(Q(pid, "恩格斯在1895年《〈法兰西阶级斗争〉导言》中如何论述革命策略？",
    "analysis", "introduction-class-struggles-france-1895", "scientific_socialism", "hard")); pid += 1

questions.append(Q(pid, "恩格斯如何论述俄国沙皇政府的对外政策？",
    "analysis", "foreign-policy-russian-tsarism", "scientific_socialism", "hard")); pid += 1

questions.append(Q(pid, "《1891年社会民主党纲领草案批判》的主要内容",
    "analysis", "critique-social-democratic-program-1891", "scientific_socialism", "hard")); pid += 1

# ── 共产主义原理 ──
questions.append(Q(pid, "恩格斯在《共产主义原理》中如何回答'什么是共产主义'？",
    "concept_explain", "principles-communism", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "废除私有制的主张在恩格斯著作中是如何阐述的？",
    "analysis", "principles-communism", "scientific_socialism", "medium")); pid += 1

# ── 跨著作 / 综合 ──
questions.append(Q(pid, "马克思主义的国家学说",
    "concept_explain", "origin-family-private-property-state", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "无产阶级政党的性质和作用",
    "analysis", "communist-manifesto", "scientific_socialism", "medium")); pid += 1

questions.append(Q(pid, "马克思和恩格斯如何看待民族问题？",
    "analysis", "speech-on-poland", "scientific_socialism", "hard")); pid += 1


# ═══════════════════════════════════════════════════════════════════
# Validate & Write
# ═══════════════════════════════════════════════════════════════════

# Validate all work_ids
for q in questions:
    wid = q["expected_work_id"]
    assert wid in work_ids, f"Question {q['id']}: unknown work_id '{wid}'"
    for hn in q.get("hard_negative", []):
        assert hn in work_ids, f"Question {q['id']}: unknown hard_negative '{hn}'"

# Check distribution
from collections import Counter
disc_counts = Counter(q["discipline"] for q in questions)
type_counts = Counter(q["question_type"] for q in questions)
diff_counts = Counter(q["difficulty"] for q in questions)

print(f"Total questions: {len(questions)}")
print(f"By discipline: {dict(disc_counts)}")
print(f"By type: {dict(type_counts)}")
print(f"By difficulty: {dict(diff_counts)}")

# Verify unique IDs
ids = [q["id"] for q in questions]
assert len(ids) == len(set(ids)), "Duplicate IDs!"

# Write
out_path = ROOT / "eval_dataset_v2.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"\nWritten {len(questions)} questions to {out_path}")
