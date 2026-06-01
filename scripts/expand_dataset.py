"""Add 55 questions to eval_dataset_v2.json, targeting 200 total."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "eval_dataset_v2.json", encoding="utf-8") as f:
    ds = json.load(f)

nid = max(q["id"] for q in ds) + 1

new = [
    # === party_labor +7 (8->15) ===
    (nid+0, '马克思在《国际工人协会成立宣言》中如何论述工人阶级的解放？', 'analysis', 'inaugural-address-iwa', 'party_labor', 'medium'),
    (nid+1, '《共产主义者同盟中央委员会告同盟书》的核心内容', 'analysis', 'address-central-authority', 'party_labor', 'hard'),
    (nid+2, '恩格斯论述美国工人运动的特点', 'analysis', 'american-labor-movement', 'party_labor', 'hard'),
    (nid+3, '《给倍倍尔、李卜克内西、白拉克等人的通告信》的背景', 'analysis', 'circular-letter-bebel-liebknecht', 'party_labor', 'hard'),
    (nid+4, '恩格斯在《德国的社会主义》中如何评价德国社会民主党？', 'analysis', 'socialism-germany', 'party_labor', 'hard'),
    (nid+5, '马克思和恩格斯关于党的纪律的论述', 'concept_explain', 'circular-letter-bebel-liebknecht', 'party_labor', 'hard'),
    (nid+6, '第一国际时期马克思和恩格斯的主要活动', 'analysis', 'inaugural-address-iwa', 'party_labor', 'medium'),

    # === peasant_land +7 (8->15) ===
    (nid+7, '马克思在《资本论》中如何论述土地私有制？', 'analysis', 'capital-vol1', 'peasant_land', 'hard'),
    (nid+8, '恩格斯对法国和德国农民问题的比较分析', 'analysis', 'peasant-question-france-germany', 'peasant_land', 'hard'),
    (nid+9, '《德国农民战争》中恩格斯对闵采尔的评价', 'analysis', 'peasant-war-germany', 'peasant_land', 'hard'),
    (nid+10, '马克思论小块土地所有制', 'concept_explain', 'eighteenth-brumaire', 'peasant_land', 'hard'),
    (nid+11, '恩格斯论农民同盟军的作用', 'analysis', 'peasant-question-france-germany', 'peasant_land', 'medium'),
    (nid+12, '《法德农民问题》中合作社的具体组织形式', 'analysis', 'peasant-question-france-germany', 'peasant_land', 'hard'),
    (nid+13, '马克思和恩格斯对农民阶级的革命性分析', 'concept_explain', 'class-struggles-france', 'peasant_land', 'medium'),

    # === national_colonial +5 (10->15) ===
    (nid+14, '马克思论英国在印度的双重使命', 'analysis', 'future-results-british-rule-india', 'national_colonial', 'medium'),
    (nid+15, '《英人在华的残暴行动》中马克思对鸦片战争的谴责', 'analysis', 'english-atrocities-china', 'national_colonial', 'medium'),
    (nid+16, '马克思论俄国在远东的扩张', 'analysis', 'russian-success-far-east', 'national_colonial', 'hard'),
    (nid+17, '恩格斯论波兰民族解放运动的意义', 'analysis', 'speech-on-poland', 'national_colonial', 'hard'),
    (nid+18, '马克思对自由贸易的批判与殖民地问题', 'analysis', 'speech-free-trade', 'national_colonial', 'hard'),

    # === scientific_socialism +5 (15->20) ===
    (nid+19, '恩格斯在《共产主义原理》中如何描述未来共产主义社会？', 'analysis', 'principles-communism', 'scientific_socialism', 'medium'),
    (nid+20, '《社会主义从空想到科学的发展》中唯物史观与剩余价值的关系', 'analysis', 'socialism-utopian-scientific', 'scientific_socialism', 'medium'),
    (nid+21, '马克思在《哥达纲领批判》中批判了拉萨尔主义的哪些观点？', 'analysis', 'critique-gotha-programme', 'scientific_socialism', 'medium'),
    (nid+22, '《共产党宣言》1872年德文版序言的主要内容', 'analysis', 'communist-manifesto', 'scientific_socialism', 'hard'),
    (nid+23, '共产主义社会两个阶段的理论', 'concept_explain', 'critique-gotha-programme', 'scientific_socialism', 'medium'),

    # === state_revolution_military +5 (15->20) ===
    (nid+24, '恩格斯在《德国的革命和反革命》中如何分析1848年革命的失败？', 'analysis', 'revolution-counterrevolution-germany', 'state_revolution_military', 'hard'),
    (nid+25, '马克思论波拿巴主义的历史特征', 'concept_explain', 'eighteenth-brumaire', 'state_revolution_military', 'hard'),
    (nid+26, '恩格斯军事理论的核心观点', 'concept_explain', 'conditions-war-1852', 'state_revolution_military', 'hard'),
    (nid+27, '马克思和恩格斯对巴枯宁无政府主义的批判', 'analysis', 'bakunin-state-anarchism-summary', 'state_revolution_military', 'hard'),
    (nid+28, '恩格斯晚年对革命策略的重新思考', 'analysis', 'introduction-class-struggles-france-1895', 'state_revolution_military', 'medium'),

    # === history_religion_culture +5 (10->15) ===
    (nid+29, '恩格斯论原始基督教与现代社会主义的相似性', 'analysis', 'primitive-christianity', 'history_religion_culture', 'hard'),
    (nid+30, '马克思和恩格斯如何评价费尔巴哈的历史地位？', 'analysis', 'ludwig-feuerbach', 'history_religion_culture', 'medium'),
    (nid+31, '《马克思和新莱茵报》中恩格斯对1848年革命的回顾', 'analysis', 'marx-and-neue-rheinische-zeitung', 'history_religion_culture', 'hard'),
    (nid+32, '论封建制度的瓦解和民族国家的产生——恩格斯的历史分析', 'analysis', 'decline-feudalism-rise-nation-states', 'history_religion_culture', 'hard'),
    (nid+33, '资本主义为什么首先在西欧产生', 'analysis', 'capital-vol1', 'history_religion_culture', 'hard'),

    # === philosophy +5 (37->42) ===
    (nid+34, '《关于费尔巴哈的提纲》中革命的实践的含义', 'concept_explain', 'theses-feuerbach', 'philosophy', 'medium'),
    (nid+35, '恩格斯在《自然辩证法》中如何论述辩证法的三大规律？', 'analysis', 'dialectics-nature', 'philosophy', 'medium'),
    (nid+36, '《神圣家族》中历史活动是群众的活动这一命题', 'concept_explain', 'holy-family', 'philosophy', 'hard'),
    (nid+37, '马克思对黑格尔唯心主义辩证法的批判与继承', 'analysis', 'economic-philosophic-manuscripts-1844', 'philosophy', 'hard'),
    (nid+38, '《德意志意识形态》中市民社会概念的含义', 'concept_explain', 'german-ideology', 'philosophy', 'hard'),

    # === political_economy +6 (42->48) ===
    (nid+39, '马克思对重农学派的评价', 'analysis', 'manuscripts-1861-1863-selections', 'political_economy', 'hard'),
    (nid+40, '马克思论资本主义信用制度', 'analysis', 'capital-vol3', 'political_economy', 'hard'),
    (nid+41, '《资本论》中自由的工人的含义', 'concept_explain', 'capital-vol1', 'political_economy', 'medium'),
    (nid+42, '马克思如何分析简单商品流通与资本主义流通的区别？', 'analysis', 'capital-vol1', 'political_economy', 'hard'),
    (nid+43, '马克思论机器大工业对工人阶级的影响', 'analysis', 'capital-vol1', 'political_economy', 'medium'),
    (nid+44, '恩格斯对洛贝尔图斯地租理论的批判', 'analysis', 'marx-and-rodbertus', 'political_economy', 'hard'),

    # === 书信卷 +5 ===
    (nid+45, '马克思致安年科夫的信（1846年12月28日）的主要内容', 'analysis', 'letters-1842-1848', 'history_religion_culture', 'hard'),
    (nid+46, '恩格斯致康拉德施米特信中关于历史唯物主义的论述', 'analysis', 'letters-1884-1895', 'history_religion_culture', 'hard'),
    (nid+47, '马克思致魏德迈信中关于阶级专政的论述', 'analysis', 'letters-1849-1859', 'state_revolution_military', 'hard'),
    (nid+48, '恩格斯致博尔吉乌斯信中关于历史唯物主义的阐述', 'analysis', 'letters-1884-1895', 'philosophy', 'hard'),
    (nid+49, '马克思致查苏利奇信中关于俄国农村公社的论述', 'analysis', 'letter-to-zasulich', 'history_religion_culture', 'hard'),

    # === 手稿 + 边缘卷 +6 ===
    (nid+50, '《政治经济学批判（1857-1858年手稿）》中关于自动化与共产主义', 'analysis', 'grundrisse-selections', 'political_economy', 'hard'),
    (nid+51, '马克思论资本主义条件下机器的应用', 'analysis', 'manuscripts-1861-1863-selections', 'political_economy', 'hard'),
    (nid+52, '恩格斯在《论住宅问题》中如何批判蒲鲁东主义的解决方案？', 'analysis', 'housing-question', 'state_revolution_military', 'hard'),
    (nid+53, '法国工人党纲领导言的主要内容', 'analysis', 'french-worker-party-program', 'party_labor', 'hard'),
    (nid+54, '马克思对劳动所得这一口号的批判', 'quote_lookup', 'critique-gotha-programme', 'scientific_socialism', 'medium'),
]

for q in new:
    ds.append({
        "id": q[0], "question": q[1], "question_type": q[2],
        "expected_work_id": q[3], "discipline": q[4], "difficulty": q[5],
    })

with open(ROOT / "eval_dataset_v2.json", "w", encoding="utf-8") as f:
    json.dump(ds, f, ensure_ascii=False, indent=2)

dc = Counter(q["discipline"] for q in ds)
tc = Counter(q["question_type"] for q in ds)
df = Counter(q["difficulty"] for q in ds)
print(f"Total: {len(ds)} questions")
print(f"By discipline:")
for d, c in dc.most_common():
    print(f"  {d}: {c}")
print(f"By type: {dict(tc)}")
print(f"By difficulty: {dict(df)}")
