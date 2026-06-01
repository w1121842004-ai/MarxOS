"""Remap all 89 works from 3-discipline to 8-discipline taxonomy."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "rag/work_catalog.json", encoding="utf-8") as f:
    wc = json.load(f)

discipline_map = {
    # Vol.1: 1843-1848
    "critique-hegel-law-intro": ["philosophy", "history_religion_culture"],
    "on-the-jewish-question": ["philosophy", "history_religion_culture"],
    "outlines-critique-political-economy": ["political_economy"],
    "condition-england-18th-century": ["history_religion_culture"],
    "economic-philosophic-manuscripts-1844": ["philosophy", "political_economy"],
    "holy-family": ["philosophy"],
    "condition-working-class-england": ["party_labor", "history_religion_culture"],
    "theses-feuerbach": ["philosophy"],
    "german-ideology": ["philosophy"],
    "poverty-philosophy": ["political_economy", "philosophy"],
    "communists-and-karl-heinzen": ["scientific_socialism"],
    "principles-communism": ["scientific_socialism"],
    "speech-on-poland": ["national_colonial"],
    "wage-labour-capital": ["political_economy"],
    "speech-free-trade": ["political_economy"],

    # Vol.2: 1848-1859
    "communist-manifesto": ["scientific_socialism", "party_labor"],
    "crisis-and-counterrevolution": ["state_revolution_military"],
    "bourgeoisie-and-counterrevolution": ["state_revolution_military"],
    "class-struggles-france": ["state_revolution_military", "scientific_socialism"],
    "address-central-authority": ["party_labor"],
    "peasant-war-germany": ["peasant_land", "history_religion_culture"],
    "conditions-war-1852": ["state_revolution_military"],
    "revolution-counterrevolution-germany": ["state_revolution_military"],
    "eighteenth-brumaire": ["state_revolution_military", "history_religion_culture"],
    "speech-peoples-paper": ["scientific_socialism"],
    "credit-mobilier": ["political_economy"],
    "preface-critique-political-economy": ["philosophy", "political_economy"],
    "engels-review-critique-political-economy": ["political_economy", "history_religion_culture"],
    "revolution-china-europe": ["national_colonial"],
    "russian-trade-china": ["national_colonial"],
    "english-atrocities-china": ["national_colonial"],
    "persia-and-china": ["national_colonial"],
    "opium-trade": ["national_colonial"],
    "anglo-chinese-treaty": ["national_colonial"],
    "china-britain-treaty": ["national_colonial"],
    "russian-success-far-east": ["national_colonial"],
    "new-war-china": ["national_colonial", "state_revolution_military"],
    "trade-with-china": ["national_colonial", "political_economy"],
    "british-rule-india": ["national_colonial"],
    "future-results-british-rule-india": ["national_colonial"],

    # Vol.3: 1864-1883
    "inaugural-address-iwa": ["party_labor"],
    "on-proudhon": ["history_religion_culture"],
    "value-price-profit": ["political_economy"],
    "civil-war-france": ["state_revolution_military", "party_labor"],
    "political-indifferentism": ["state_revolution_military"],
    "refugee-literature": ["state_revolution_military"],
    "russian-social-conditions": ["national_colonial"],
    "critique-gotha-programme": ["scientific_socialism", "state_revolution_military"],
    "socialism-utopian-scientific": ["scientific_socialism"],
    "letter-to-otechestvennye-zapiski": ["history_religion_culture", "philosophy"],
    "circular-letter-bebel-liebknecht": ["party_labor"],

    # Vol.4: 1884-1895
    "marx-and-neue-rheinische-zeitung": ["history_religion_culture"],
    "origin-family-private-property-state": ["state_revolution_military", "history_religion_culture"],
    "marx-and-rodbertus": ["political_economy", "history_religion_culture"],
    "decline-feudalism-rise-nation-states": ["history_religion_culture"],
    "history-communist-league": ["party_labor", "history_religion_culture"],
    "prussian-peasant-history": ["peasant_land", "history_religion_culture"],
    "ludwig-feuerbach": ["philosophy"],
    "american-labor-movement": ["party_labor"],
    "protection-and-free-trade": ["political_economy"],
    "foreign-policy-russian-tsarism": ["national_colonial", "state_revolution_military"],
    "critique-social-democratic-program-1891": ["party_labor", "state_revolution_military"],
    "socialism-germany": ["party_labor", "scientific_socialism"],
    "primitive-christianity": ["history_religion_culture"],
    "peasant-question-france-germany": ["peasant_land", "scientific_socialism"],
    "introduction-class-struggles-france-1895": ["state_revolution_military", "party_labor"],

    # Vol.5-7: Capital
    "capital-vol1": ["political_economy"],
    "capital-vol2": ["political_economy"],
    "capital-vol3": ["political_economy"],

    # Vol.8: Capital Manuscripts
    "grundrisse-introduction": ["political_economy", "philosophy"],
    "grundrisse-selections": ["political_economy"],
    "manuscripts-1861-1863-selections": ["political_economy"],
    "capital-manuscript-1863-1865": ["political_economy"],

    # Vol.9: Anti-Duhring + Dialectics
    "anti-duhring": ["philosophy", "political_economy"],
    "dialectics-nature": ["philosophy"],

    # Vol.10: Letters
    "letters-1842-1848": ["history_religion_culture", "party_labor"],
    "letters-1849-1859": ["political_economy", "history_religion_culture"],
    "letters-1860-1870": ["political_economy", "party_labor"],
    "letters-1871-1883": ["party_labor", "state_revolution_military"],
    "letters-1884-1895": ["philosophy", "scientific_socialism"],

    # Xuanji only
    "on-authority": ["state_revolution_military", "party_labor"],
    "housing-question": ["state_revolution_military"],
    "letter-to-bebel-1875": ["party_labor"],
    "bakunin-state-anarchism-summary": ["state_revolution_military"],
    "french-worker-party-program": ["party_labor"],
    "letter-to-zasulich": ["history_religion_culture", "national_colonial"],
    "marx-graveside-speech": ["history_religion_culture"],
    "engels-review-capital-vol1": ["political_economy"],
}

# Apply
unmapped = []
for w in wc["works"]:
    wid = w["work_id"]
    if wid in discipline_map:
        w["discipline"] = discipline_map[wid]
    else:
        unmapped.append(wid)
        w["discipline"] = ["scientific_socialism"]

if unmapped:
    print(f"UNMAPPED: {unmapped}")

# Update taxonomy definition
wc["discipline_taxonomy"] = {
    "philosophy": {
        "label": "马克思主义哲学",
        "sub_categories": ["历史唯物主义", "辩证唯物主义", "实践哲学", "异化理论", "意识形态批判"],
    },
    "political_economy": {
        "label": "政治经济学",
        "sub_categories": ["剩余价值理论", "资本积累", "地租理论", "劳动价值论", "经济危机理论"],
    },
    "scientific_socialism": {
        "label": "科学社会主义",
        "sub_categories": ["阶级斗争", "无产阶级革命", "共产主义理论"],
    },
    "party_labor": {
        "label": "党的建设与工人运动",
        "sub_categories": ["无产阶级政党", "第一国际", "工会运动", "党纲党章"],
    },
    "peasant_land": {
        "label": "农民问题与土地问题",
        "sub_categories": ["农民合作社", "土地所有制", "工农联盟", "农业问题"],
    },
    "national_colonial": {
        "label": "民族问题与殖民主义",
        "sub_categories": ["殖民主义批判", "民族解放", "东方问题", "中国与印度"],
    },
    "state_revolution_military": {
        "label": "国家、革命与军事",
        "sub_categories": ["国家学说", "无产阶级专政", "革命策略", "军事理论"],
    },
    "history_religion_culture": {
        "label": "历史、宗教与文化",
        "sub_categories": ["原始社会", "宗教批判", "德国历史", "人物传记"],
    },
}

with open(ROOT / "rag/work_catalog.json", "w", encoding="utf-8") as f:
    json.dump(wc, f, ensure_ascii=False, indent=2)

# Stats
disc_counts = Counter()
for w in wc["works"]:
    for d in w["discipline"]:
        disc_counts[d] += 1
disc_labels = wc["discipline_taxonomy"]

print("Works per discipline:")
for d, c in disc_counts.most_common():
    label = disc_labels[d]["label"]
    print(f"  {label}: {c}")
print(f"\nTotal works: {len(wc['works'])}")
print("(Sum > 89 because some works have 2 disciplines)")
