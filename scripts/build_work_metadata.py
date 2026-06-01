"""
Curated work catalog for MarxOS — all major works across 文集(10卷) + 选集(4卷).
Hand-annotated with discipline, concepts, quotes, and cross-edition page references.

資本論三卷作为单条 work，书信按年代分组。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load article_map for page number reference
with open(ROOT / "rag/article_map_core.json", encoding="utf-8") as f:
    am = json.load(f)

# ── All works, organized by 文集 volume ──────────────────────────

WORKS = []

def W(wid, title, author, co_author, year, wtype, discipline, concepts,
      quotes=None, aliases=None,
      wenji_vol=None, wenji_pages=None, wenji_is_full=True,
      xuanji_vol=None, xuanji_pages=None, xuanji_is_full=True,
      note=None):
    """Define a work with cross-edition entries"""
    editions = {}
    if wenji_vol and wenji_pages:
        editions[f"wenji_v{wenji_vol}"] = {
            "source": f"mea{wenji_vol:02d}.pdf",
            "article_title": title,
            "start_page": wenji_pages[0],
            "end_page": wenji_pages[1],
            "entry_type": "primary",
            "is_full_text": wenji_is_full,
        }
    if xuanji_vol and xuanji_pages:
        editions[f"xuanji_v{xuanji_vol}"] = {
            "source": f"mes{xuanji_vol:02d}.pdf",
            "article_title": title,
            "start_page": xuanji_pages[0],
            "end_page": xuanji_pages[1],
            "entry_type": "parallel" if wenji_vol else "primary",
            "is_full_text": xuanji_is_full,
        }
    if note:
        for ek in editions:
            editions[ek]["note"] = note

    work = {
        "work_id": wid,
        "title": title,
        "aliases": aliases or [],
        "author": author,
        "co_author": co_author,
        "writing_year": year,
        "work_type": wtype,
        "discipline": discipline,
        "concepts": concepts,
        "quotes": quotes or [],
        "editions": editions,
    }
    WORKS.append(work)


# ═══════════════════════════════════════════════════════════════════
# 文集 第一卷 (mea01.pdf): 1843—1848年著作
# ═══════════════════════════════════════════════════════════════════

W("critique-hegel-law-intro", "《黑格尔法哲学批判》导言",
  "卡·马克思", None, "1843", "essay",
  ["philosophy", "scientific_socialism"],
  ["宗教批判", "无产阶级历史使命", "哲学与革命", "德国解放"],
  quotes=["宗教是人民的鸦片", "批判的武器当然不能代替武器的批判"],
  aliases=["黑格尔法哲学批判导言"],
  wenji_vol=1, wenji_pages=(3,18),
  xuanji_vol=1, xuanji_pages=(1,16))

W("on-the-jewish-question", "论犹太人问题",
  "卡·马克思", None, "1843", "essay",
  ["philosophy", "scientific_socialism"],
  ["政治解放", "人类解放", "宗教与国家", "市民社会", "人权批判"],
  aliases=["犹太人问题"],
  wenji_vol=1, wenji_pages=(21,55))

W("outlines-critique-political-economy", "国民经济学批判大纲",
  "弗·恩格斯", None, "1843", "essay",
  ["political_economy"],
  ["政治经济学批判", "私有制", "竞争", "价值", "资本与劳动对立"],
  aliases=["政治经济学批判大纲"],
  wenji_vol=1, wenji_pages=(56,86),
  xuanji_vol=1, xuanji_pages=(17,48))

W("condition-england-18th-century", "英国状况 十八世纪",
  "弗·恩格斯", None, "1844", "essay",
  ["scientific_socialism", "philosophy"],
  ["英国工业革命", "社会状况", "阶级形成"],
  wenji_vol=1, wenji_pages=(87,108))

W("economic-philosophic-manuscripts-1844", "1844年经济学哲学手稿",
  "卡·马克思", None, "1844", "manuscript",
  ["philosophy", "political_economy"],
  ["异化劳动", "私有财产", "共产主义", "黑格尔辩证法批判", "人的本质"],
  quotes=["劳动为富人生产了奇迹般的东西，但是为工人生产了赤贫"],
  aliases=["经济学哲学手稿", "巴黎手稿"],
  wenji_vol=1, wenji_pages=(109,248),
  xuanji_vol=1, xuanji_pages=(49,63), xuanji_is_full=False,
  note="选集为节选（异化劳动和私有财产）")

W("holy-family", "神圣家族",
  "卡·马克思", "弗·恩格斯", "1844", "book",
  ["philosophy"],
  ["青年黑格尔派批判", "唯物主义史", "群众史观", "思辨哲学批判"],
  quotes=["历史活动是群众的活动"],
  aliases=["对批判的批判所做的批判"],
  wenji_vol=1, wenji_pages=(249,359), wenji_is_full=False,
  note="文集为节选")

W("condition-working-class-england", "英国工人阶级状况",
  "弗·恩格斯", None, "1845", "book",
  ["scientific_socialism", "political_economy"],
  ["工人阶级", "工业革命", "城市化", "工人运动"],
  wenji_vol=1, wenji_pages=(363,498), wenji_is_full=False,
  xuanji_vol=1, xuanji_pages=(65,132), xuanji_is_full=False,
  note="文集和选集均为节选")

W("theses-feuerbach", "关于费尔巴哈的提纲",
  "卡·马克思", None, "1845", "thesis",
  ["philosophy"],
  ["实践", "唯物主义", "感性活动", "社会关系", "改变世界"],
  quotes=["哲学家们只是用不同的方式解释世界，而问题在于改变世界",
          "人的本质在其现实性上是一切社会关系的总和"],
  aliases=["费尔巴哈提纲", "费尔巴哈论纲"],
  wenji_vol=1, wenji_pages=(499,506),
  xuanji_vol=1, xuanji_pages=(133,140))

W("german-ideology", "德意志意识形态",
  "卡·马克思", "弗·恩格斯", "1845-1846", "manuscript",
  ["philosophy"],
  ["历史唯物主义", "意识形态", "分工", "所有制形式", "市民社会"],
  quotes=["不是意识决定生活，而是生活决定意识"],
  wenji_vol=1, wenji_pages=(507,591), wenji_is_full=False,
  xuanji_vol=1, xuanji_pages=(141,215), xuanji_is_full=False,
  note="文集和选集均为第一卷第一章《费尔巴哈》节选")

W("poverty-philosophy", "哲学的贫困",
  "卡·马克思", None, "1847", "book",
  ["political_economy", "philosophy"],
  ["政治经济学方法", "分工", "竞争", "垄断", "剩余价值萌芽"],
  aliases=["答蒲鲁东先生的《贫困的哲学》"],
  wenji_vol=1, wenji_pages=(595,692), wenji_is_full=False,
  xuanji_vol=1, xuanji_pages=(217,275), xuanji_is_full=False,
  note="文集和选集均为节选")

W("communists-and-karl-heinzen", "共产主义者和卡尔·海因岑",
  "弗·恩格斯", None, "1847", "article",
  ["scientific_socialism"],
  ["共产主义宣传", "小资产阶级激进主义批判"],
  wenji_vol=1, wenji_pages=(75,85),
  xuanji_vol=1, xuanji_pages=(276,294))

W("principles-communism", "共产主义原理",
  "弗·恩格斯", None, "1847", "essay",
  ["scientific_socialism"],
  ["共产主义", "无产阶级革命", "废除私有制"],
  wenji_vol=1, wenji_pages=(673,693),
  xuanji_vol=1, xuanji_pages=(295,312))

W("speech-on-poland", "关于波兰的演说",
  "卡·马克思", "弗·恩格斯", "1847", "speech",
  ["scientific_socialism"],
  ["波兰问题", "民族解放", "国际主义"],
  aliases=["论波兰"],
  wenji_vol=1, wenji_pages=(694,700),
  xuanji_vol=1, xuanji_pages=(313,316))

W("wage-labour-capital", "雇佣劳动与资本",
  "卡·马克思", None, "1847", "essay",
  ["political_economy"],
  ["工资", "资本", "剩余价值萌芽", "生产关系"],
  wenji_vol=1, wenji_pages=(711,742),
  xuanji_vol=1, xuanji_pages=(317,359))

W("speech-free-trade", "关于自由贸易问题的演说",
  "卡·马克思", None, "1848", "speech",
  ["political_economy"],
  ["自由贸易", "保护关税", "资本主义批判"],
  aliases=["关于自由贸易的演说"],
  wenji_vol=1, wenji_pages=(744,759),
  xuanji_vol=1, xuanji_pages=(360,375))


# ═══════════════════════════════════════════════════════════════════
# 文集 第二卷 (mea02.pdf): 1848—1859年著作
# ═══════════════════════════════════════════════════════════════════

W("communist-manifesto", "共产党宣言",
  "卡·马克思", "弗·恩格斯", "1848", "manifesto",
  ["scientific_socialism", "philosophy"],
  ["阶级斗争", "资产阶级", "无产阶级", "共产主义", "政党"],
  quotes=["一个幽灵，共产主义的幽灵，在欧洲游荡",
          "至今一切社会的历史都是阶级斗争的历史",
          "无产者在这个革命中失去的只是锁链，他们获得的将是整个世界"],
  aliases=["共产主义宣言"],
  wenji_vol=2, wenji_pages=(3,67),
  xuanji_vol=1, xuanji_pages=(376,435))

W("crisis-and-counterrevolution", "危机和反革命",
  "卡·马克思", None, "1848", "article",
  ["scientific_socialism"],
  ["革命危机", "反革命", "1848年革命"],
  wenji_vol=2, wenji_pages=(68,71),
  xuanji_vol=1, xuanji_pages=(436,439))

W("bourgeoisie-and-counterrevolution", "资产阶级和反革命",
  "卡·马克思", None, "1848", "article",
  ["scientific_socialism"],
  ["资产阶级", "反革命", "德国革命"],
  wenji_vol=2, wenji_pages=(72,76),
  xuanji_vol=1, xuanji_pages=(440,444))

W("class-struggles-france", "1848年至1850年的法兰西阶级斗争",
  "卡·马克思", None, "1850", "book",
  ["scientific_socialism"],
  ["阶级斗争", "无产阶级专政", "革命", "法国"],
  quotes=["阶级斗争必然要导致无产阶级专政"],
  aliases=["法兰西阶级斗争"],
  wenji_vol=2, wenji_pages=(77,187),
  xuanji_vol=1, xuanji_pages=(445,552))

W("address-central-authority", "共产主义者同盟中央委员会告同盟书",
  "卡·马克思", "弗·恩格斯", "1850", "essay",
  ["scientific_socialism"],
  ["不断革命", "无产阶级独立组织"],
  wenji_vol=2, wenji_pages=(188,199),
  xuanji_vol=1, xuanji_pages=(553,564))

W("peasant-war-germany", "德国农民战争",
  "弗·恩格斯", None, "1850", "book",
  ["scientific_socialism", "philosophy"],
  ["农民战争", "宗教改革", "德国历史", "阶级分析"],
  wenji_vol=2, wenji_pages=(200,283))

W("conditions-war-1852", "1852年神圣同盟对法战争的条件与前景",
  "弗·恩格斯", None, "1851", "manuscript",
  ["scientific_socialism"],
  ["军事理论", "神圣同盟", "法国"],
  wenji_vol=2, wenji_pages=(347,350))

W("revolution-counterrevolution-germany", "德国的革命和反革命",
  "弗·恩格斯", None, "1851-1852", "article",
  ["scientific_socialism"],
  ["德国革命", "1848年革命", "阶级分析", "民族问题"],
  aliases=["德国的革命与反革命"],
  wenji_vol=2, wenji_pages=(351,460),
  xuanji_vol=1, xuanji_pages=(565,662))

W("eighteenth-brumaire", "路易·波拿巴的雾月十八日",
  "卡·马克思", None, "1852", "book",
  ["scientific_socialism", "philosophy"],
  ["国家机器", "阶级分析", "农民阶级", "波拿巴主义"],
  quotes=["人们自己创造自己的历史，但并不是随心所欲地创造"],
  aliases=["路易·波拿巴的雾月十八", "雾月十八日"],
  wenji_vol=2, wenji_pages=(461,578),
  xuanji_vol=1, xuanji_pages=(663,774))

W("speech-peoples-paper", "在《人民报》创刊纪念会上的演说",
  "卡·马克思", None, "1856", "speech",
  ["scientific_socialism", "philosophy"],
  ["无产阶级历史使命", "技术进步"],
  wenji_vol=2, wenji_pages=(579,581),
  xuanji_vol=1, xuanji_pages=(775,777))

W("credit-mobilier", "法国的动产信用公司",
  "卡·马克思", None, "1856", "article",
  ["political_economy"],
  ["信用制度", "金融资本", "法国经济"],
  wenji_vol=2, wenji_pages=(582,587))

W("preface-critique-political-economy", "《政治经济学批判》序言",
  "卡·马克思", None, "1859", "essay",
  ["philosophy", "political_economy"],
  ["历史唯物主义", "经济基础", "上层建筑", "社会形态"],
  quotes=["物质生活的生产方式制约着整个社会生活、政治生活和精神生活的过程",
          "不是人们的意识决定人们的存在，相反，是人们的社会存在决定人们的意识"],
  wenji_vol=2, wenji_pages=(588,594))

W("engels-review-critique-political-economy", "卡尔·马克思《政治经济学批判。第一分册》",
  "弗·恩格斯", None, "1859", "article",
  ["political_economy"],
  ["政治经济学批判", "马克思经济学"],
  wenji_vol=2, wenji_pages=(595,610))

# China/India articles group
W("revolution-china-europe", "中国革命和欧洲革命",
  "卡·马克思", None, "1853", "article",
  ["scientific_socialism"],
  ["中国革命", "欧洲革命", "殖民主义"],
  wenji_vol=2, wenji_pages=(611,618),
  xuanji_vol=1, xuanji_pages=(778,785))

W("russian-trade-china", "俄国的对华贸易",
  "卡·马克思", None, "1857", "article",
  ["scientific_socialism"],
  ["对华贸易", "俄国扩张", "殖民主义"],
  wenji_vol=2, wenji_pages=(619,622),
  xuanji_vol=1, xuanji_pages=(786,789))

W("english-atrocities-china", "英人在华的残暴行动",
  "卡·马克思", None, "1857", "article",
  ["scientific_socialism"],
  ["英国殖民主义", "鸦片战争", "中国"],
  wenji_vol=2, wenji_pages=(623,626),
  xuanji_vol=1, xuanji_pages=(790,793))

W("persia-and-china", "波斯和中国",
  "弗·恩格斯", None, "1857", "article",
  ["scientific_socialism"],
  ["波斯", "中国", "殖民主义"],
  wenji_vol=2, wenji_pages=(627,633),
  xuanji_vol=1, xuanji_pages=(794,800))

W("opium-trade", "鸦片贸易史",
  "卡·马克思", None, "1858", "article",
  ["scientific_socialism"],
  ["鸦片贸易", "英国殖民主义", "中国"],
  wenji_vol=2, wenji_pages=(634,641),
  xuanji_vol=1, xuanji_pages=(801,808))

W("anglo-chinese-treaty", "英中条约",
  "卡·马克思", None, "1858", "article",
  ["scientific_socialism"],
  ["英中条约", "不平等条约", "殖民主义"],
  wenji_vol=2, wenji_pages=(642,647),
  xuanji_vol=1, xuanji_pages=(809,814))

W("china-britain-treaty", "中国和英国的条约",
  "卡·马克思", None, "1858", "article",
  ["scientific_socialism"],
  ["不平等条约", "中英关系", "殖民主义"],
  wenji_vol=2, wenji_pages=(648,653),
  xuanji_vol=1, xuanji_pages=(815,820))

W("russian-success-far-east", "俄国在远东的成功",
  "弗·恩格斯", None, "1858", "article",
  ["scientific_socialism"],
  ["俄国扩张", "远东", "殖民主义"],
  wenji_vol=2, wenji_pages=(654,658),
  xuanji_vol=1, xuanji_pages=(821,825))

W("new-war-china", "新的对华战争",
  "卡·马克思", None, "1859", "article",
  ["scientific_socialism"],
  ["第二次鸦片战争", "英国殖民主义", "中国"],
  wenji_vol=2, wenji_pages=(659,674),
  xuanji_vol=1, xuanji_pages=(826,842))

W("trade-with-china", "对华贸易",
  "卡·马克思", None, "1859", "article",
  ["political_economy"],
  ["对华贸易", "国际贸易", "殖民主义"],
  wenji_vol=2, wenji_pages=(675,678),
  xuanji_vol=1, xuanji_pages=(843,847))

W("british-rule-india", "不列颠在印度的统治",
  "卡·马克思", None, "1853", "article",
  ["scientific_socialism"],
  ["英国殖民主义", "印度", "亚洲社会"],
  wenji_vol=2, wenji_pages=(679,685),
  xuanji_vol=1, xuanji_pages=(848,855))

W("future-results-british-rule-india", "不列颠在印度统治的未来结果",
  "卡·马克思", None, "1853", "article",
  ["scientific_socialism"],
  ["英国殖民主义", "印度", "亚洲社会"],
  wenji_vol=2, wenji_pages=(686,693),
  xuanji_vol=1, xuanji_pages=(856,863))


# ═══════════════════════════════════════════════════════════════════
# 文集 第三卷 (mea03.pdf): 1864—1883年著作
# ═══════════════════════════════════════════════════════════════════

W("inaugural-address-iwa", "国际工人协会成立宣言",
  "卡·马克思", None, "1864", "essay",
  ["scientific_socialism", "political_economy"],
  ["国际工人协会", "第一国际", "工人阶级解放"],
  wenji_vol=3, wenji_pages=(3,15))

W("on-proudhon", "论蒲鲁东",
  "卡·马克思", None, "1865", "essay",
  ["scientific_socialism", "political_economy"],
  ["蒲鲁东批判", "小资产阶级社会主义"],
  wenji_vol=3, wenji_pages=(16,24))

W("value-price-profit", "工资、价格和利润",
  "卡·马克思", None, "1865", "essay",
  ["political_economy"],
  ["工资", "利润", "剩余价值", "劳动价值论"],
  aliases=["工资价格和利润", "价值价格与利润"],
  wenji_vol=3, wenji_pages=(25,77),
  xuanji_vol=2, xuanji_pages=(17,69))

W("civil-war-france", "法兰西内战",
  "卡·马克思", None, "1871", "book",
  ["scientific_socialism"],
  ["巴黎公社", "无产阶级专政", "国家机器"],
  quotes=["工人阶级不能简单地掌握现成的国家机器，并运用它来达到自己的目的"],
  aliases=["法兰西内战"],
  wenji_vol=3, wenji_pages=(95,223))

W("political-indifferentism", "政治冷淡主义",
  "卡·马克思", None, "1873", "article",
  ["scientific_socialism"],
  ["巴枯宁主义批判", "无产阶级政治行动"],
  wenji_vol=3, wenji_pages=(339,345))

W("refugee-literature", "流亡者文献",
  "弗·恩格斯", None, "1874-1875", "article",
  ["scientific_socialism"],
  ["巴枯宁主义", "俄国革命", "工人运动"],
  wenji_vol=3, wenji_pages=(346,388))

W("russian-social-conditions", "论俄国的社会问题",
  "弗·恩格斯", None, "1875", "article",
  ["scientific_socialism"],
  ["俄国", "农村公社", "革命前景"],
  wenji_vol=3, wenji_pages=(389,416))

W("critique-gotha-programme", "哥达纲领批判",
  "卡·马克思", None, "1875", "critique",
  ["scientific_socialism", "political_economy"],
  ["按需分配", "过渡时期", "无产阶级专政", "劳动"],
  quotes=["各尽所能，按需分配",
          "在资本主义社会和共产主义社会之间，有一个从前者变为后者的革命转变时期"],
  aliases=["德国工人党纲领批注"],
  wenji_vol=3, wenji_pages=(417,450),
  xuanji_vol=3, xuanji_pages=(352,380))

W("socialism-utopian-scientific", "社会主义从空想到科学的发展",
  "弗·恩格斯", None, "1880", "book",
  ["scientific_socialism", "philosophy"],
  ["科学社会主义", "唯物史观", "剩余价值", "空想社会主义"],
  aliases=["社会主义从空想到科学"],
  wenji_vol=3, wenji_pages=(487,567))

W("letter-to-otechestvennye-zapiski", "给《祖国纪事》杂志编辑部的信",
  "卡·马克思", None, "1877", "letter",
  ["scientific_socialism", "philosophy"],
  ["历史唯物主义", "俄国道路"],
  wenji_vol=3, wenji_pages=(463,467))

W("circular-letter-bebel-liebknecht", "给倍倍尔、李卜克内西、白拉克等人的通告信",
  "卡·马克思", "弗·恩格斯", "1879", "letter",
  ["scientific_socialism"],
  ["阶级斗争", "党的领导"],
  wenji_vol=3, wenji_pages=(468,486))


# ═══════════════════════════════════════════════════════════════════
# 文集 第四卷 (mea04.pdf): 1884—1895年恩格斯著作
# ═══════════════════════════════════════════════════════════════════

W("marx-and-neue-rheinische-zeitung", "马克思和《新莱茵报》",
  "弗·恩格斯", None, "1884", "article",
  ["scientific_socialism"],
  ["马克思生平", "新莱茵报", "1848年革命"],
  wenji_vol=4, wenji_pages=(3,12))

W("origin-family-private-property-state", "家庭、私有制和国家的起源",
  "弗·恩格斯", None, "1884", "book",
  ["scientific_socialism", "philosophy"],
  ["家庭", "私有制", "国家", "原始社会", "妇女解放"],
  aliases=["家庭私有制和国家的起源"],
  wenji_vol=4, wenji_pages=(13,198),
  xuanji_vol=4, xuanji_pages=(669,709), xuanji_is_full=False,
  note="选集为摘选")

W("marx-and-rodbertus", "马克思和洛贝尔图斯",
  "弗·恩格斯", None, "1885", "essay",
  ["political_economy"],
  ["洛贝尔图斯", "剩余价值理论", "马克思经济学"],
  wenji_vol=4, wenji_pages=(199,214))

W("decline-feudalism-rise-nation-states", "论封建制度的瓦解和民族国家的产生",
  "弗·恩格斯", None, "1884", "manuscript",
  ["scientific_socialism", "philosophy"],
  ["封建制度", "民族国家", "君主专制"],
  wenji_vol=4, wenji_pages=(215,225))

W("history-communist-league", "关于共产主义者同盟的历史",
  "弗·恩格斯", None, "1885", "article",
  ["scientific_socialism"],
  ["共产主义者同盟", "马克思生平", "恩格斯生平"],
  wenji_vol=4, wenji_pages=(226,246))

W("prussian-peasant-history", "关于普鲁士农民的历史",
  "弗·恩格斯", None, "1885", "article",
  ["scientific_socialism"],
  ["普鲁士", "农民问题", "农奴制"],
  wenji_vol=4, wenji_pages=(247,259))

W("ludwig-feuerbach", "路德维希·费尔巴哈和德国古典哲学的终结",
  "弗·恩格斯", None, "1886", "book",
  ["philosophy"],
  ["费尔巴哈", "德国古典哲学", "黑格尔", "唯物主义", "辩证法"],
  aliases=["费尔巴哈和德国古典哲学的终结", "费尔巴哈论"],
  quotes=["全部哲学，特别是近代哲学的重大的基本问题，是思维和存在的关系问题"],
  wenji_vol=4, wenji_pages=(261,313))

W("american-labor-movement", "美国工人运动",
  "弗·恩格斯", None, "1887", "article",
  ["scientific_socialism"],
  ["美国工人运动", "无产阶级政党"],
  wenji_vol=4, wenji_pages=(316,325))

W("protection-and-free-trade", "保护关税制度和自由贸易",
  "弗·恩格斯", None, "1888", "article",
  ["political_economy"],
  ["保护关税", "自由贸易", "美国经济"],
  wenji_vol=4, wenji_pages=(333,350))

W("foreign-policy-russian-tsarism", "俄国沙皇政府的对外政策",
  "弗·恩格斯", None, "1889-1890", "article",
  ["scientific_socialism"],
  ["俄国", "沙皇制度", "对外政策", "欧洲外交"],
  wenji_vol=4, wenji_pages=(351,394))

W("critique-social-democratic-program-1891", "1891年社会民主党纲领草案批判",
  "弗·恩格斯", None, "1891", "critique",
  ["scientific_socialism"],
  ["社会民主党", "爱尔福特纲领", "民主共和国"],
  aliases=["爱尔福特纲领批判"],
  wenji_vol=4, wenji_pages=(407,422))

W("socialism-germany", "德国的社会主义",
  "弗·恩格斯", None, "1891-1892", "article",
  ["scientific_socialism"],
  ["德国社会主义", "工人运动"],
  wenji_vol=4, wenji_pages=(423,441))

W("primitive-christianity", "论原始基督教的历史",
  "弗·恩格斯", None, "1894", "article",
  ["scientific_socialism", "philosophy"],
  ["原始基督教", "宗教批判", "早期基督教"],
  wenji_vol=4, wenji_pages=(473,503))

W("peasant-question-france-germany", "法德农民问题",
  "弗·恩格斯", None, "1894", "essay",
  ["scientific_socialism"],
  ["农民问题", "土地问题", "合作社", "工农联盟"],
  aliases=["法德农民问题"],
  wenji_vol=4, wenji_pages=(507,531))

W("introduction-class-struggles-france-1895", "卡·马克思《1848年至1850年的法兰西阶级斗争》一书导言",
  "弗·恩格斯", None, "1895", "essay",
  ["scientific_socialism"],
  ["革命策略", "议会斗争", "街垒战"],
  aliases=["《法兰西阶级斗争》导言", "恩格斯晚年导言"],
  wenji_vol=4, wenji_pages=(532,554),
  xuanji_vol=4, xuanji_pages=(571,603))


# ═══════════════════════════════════════════════════════════════════
# 文集 第五卷 — 资本论 第一卷 (mea05.pdf)
# 文集 第六卷 — 资本论 第二卷 (mea06.pdf)
# 文集 第七卷 — 资本论 第三卷 (mea07.pdf)
# ═══════════════════════════════════════════════════════════════════

W("capital-vol1", "资本论 第一卷",
  "卡·马克思", None, "1867", "book",
  ["political_economy", "philosophy"],
  ["商品", "货币", "剩余价值", "资本积累", "原始积累", "价值形式"],
  quotes=["资本来到世间，从头到脚，每个毛孔都滴着血和肮脏的东西"],
  aliases=["资本论第一卷", "资本论第1卷"],
  wenji_vol=5, wenji_pages=(7,887),
  xuanji_vol=2, xuanji_pages=(81,300), xuanji_is_full=False,
  note="选集为节选")

W("capital-vol2", "资本论 第二卷",
  "卡·马克思", None, "1885", "book",
  ["political_economy"],
  ["资本循环", "资本周转", "社会总资本再生产", "流通"],
  aliases=["资本论第二卷"],
  wenji_vol=6, wenji_pages=(7,672),
  xuanji_vol=2, xuanji_pages=(301,435), xuanji_is_full=False,
  note="选集为节选")

W("capital-vol3", "资本论 第三卷",
  "卡·马克思", None, "1894", "book",
  ["political_economy"],
  ["利润", "平均利润率", "地租", "生息资本", "各种收入"],
  aliases=["资本论第三卷"],
  wenji_vol=7, wenji_pages=(7,1027),
  xuanji_vol=2, xuanji_pages=(436,655), xuanji_is_full=False,
  note="选集为节选")


# ═══════════════════════════════════════════════════════════════════
# 文集 第八卷 — 资本论手稿选编 (mea08.pdf)
# ═══════════════════════════════════════════════════════════════════

W("grundrisse-introduction", "《政治经济学批判》导言",
  "卡·马克思", None, "1857", "manuscript",
  ["political_economy", "philosophy"],
  ["生产", "分配", "交换", "消费", "政治经济学方法"],
  aliases=["1857年导言"],
  wenji_vol=8, wenji_pages=(5,36),
  xuanji_vol=2, xuanji_pages=(683,712))

W("grundrisse-selections", "《政治经济学批判（1857—1858年手稿）》摘选",
  "卡·马克思", None, "1857-1858", "manuscript",
  ["political_economy", "philosophy"],
  ["货币", "资本", "前资本主义所有制", "机器体系", "共产主义"],
  aliases=["1857-1858年经济学手稿", "大纲"],
  wenji_vol=8, wenji_pages=(37,209))

W("manuscripts-1861-1863-selections", "《政治经济学批判（1861—1863年手稿）》摘选",
  "卡·马克思", None, "1861-1863", "manuscript",
  ["political_economy"],
  ["生产劳动", "危机", "机器", "剩余价值理论"],
  aliases=["1861-1863年经济学手稿", "剩余价值理论"],
  wenji_vol=8, wenji_pages=(213,418))

W("capital-manuscript-1863-1865", "《资本论（1863—1865年手稿）》摘选",
  "卡·马克思", None, "1863-1865", "manuscript",
  ["political_economy"],
  ["直接生产过程", "流通", "再生产"],
  aliases=["1863-1865年手稿"],
  wenji_vol=8, wenji_pages=(423,754))


# ═══════════════════════════════════════════════════════════════════
# 文集 第九卷 — 反杜林论 + 自然辩证法 (mea09.pdf)
# ═══════════════════════════════════════════════════════════════════

W("anti-duhring", "反杜林论",
  "弗·恩格斯", None, "1876-1878", "book",
  ["philosophy", "political_economy", "scientific_socialism"],
  ["唯物主义", "辩证法", "政治经济学", "社会主义", "杜林批判"],
  aliases=["欧根·杜林先生在科学中实行的变革"],
  wenji_vol=9, wenji_pages=(3,398),
  xuanji_vol=3, xuanji_pages=(477,607), xuanji_is_full=False,
  note="选集为节选")

W("dialectics-nature", "自然辩证法",
  "弗·恩格斯", None, "1873-1882", "manuscript",
  ["philosophy"],
  ["辩证法", "自然科学", "物质运动", "劳动", "猿到人"],
  quotes=["劳动创造了人本身"],
  aliases=["自然辩证法（节选）"],
  wenji_vol=9, wenji_pages=(399,563))


# ═══════════════════════════════════════════════════════════════════
# 文集 第十卷 — 书信选编 (mea10.pdf)
# 按年代分组，不逐封信建条目
# ═══════════════════════════════════════════════════════════════════

W("letters-1842-1848", "书信选编 1842—1848年",
  "卡·马克思", "弗·恩格斯", "1842-1848", "letter",
  ["philosophy", "scientific_socialism"],
  ["早期通信", "共产主义者同盟"],
  wenji_vol=10, wenji_pages=(3,55), wenji_is_full=True)

W("letters-1849-1859", "书信选编 1849—1859年",
  "卡·马克思", "弗·恩格斯", "1849-1859", "letter",
  ["political_economy", "scientific_socialism"],
  ["经济危机", "革命", "政治经济学研究"],
  wenji_vol=10, wenji_pages=(57,216), wenji_is_full=True)

W("letters-1860-1870", "书信选编 1860—1870年",
  "卡·马克思", "弗·恩格斯", "1860-1870", "letter",
  ["political_economy", "scientific_socialism"],
  ["第一国际", "资本论写作", "工人运动"],
  wenji_vol=10, wenji_pages=(217,346), wenji_is_full=True)

W("letters-1871-1883", "书信选编 1871—1883年",
  "卡·马克思", "弗·恩格斯", "1871-1883", "letter",
  ["scientific_socialism"],
  ["巴黎公社", "哥达纲领", "德国党"],
  wenji_vol=10, wenji_pages=(347,496), wenji_is_full=True)

W("letters-1884-1895", "书信选编 1884—1895年",
  "弗·恩格斯", None, "1884-1895", "letter",
  ["scientific_socialism", "philosophy"],
  ["晚年通信", "第二国际", "历史唯物主义"],
  aliases=["恩格斯晚年书信"],
  wenji_vol=10, wenji_pages=(497,706), wenji_is_full=True)


# ═══════════════════════════════════════════════════════════════════
# 选集第三卷 额外篇目 (mes03.pdf) — article_map 完全损坏，手动标注
# ═══════════════════════════════════════════════════════════════════

# wenji_v3 already covers these: 法兰西内战, 哥达纲领批判, 社会主义从空想到科学,
# 反杜林论, 自然辩证法. 选集第三卷 额外篇目:

W("on-authority", "论权威",
  "弗·恩格斯", None, "1872", "article",
  ["scientific_socialism"],
  ["权威", "革命", "自治"],
  xuanji_vol=3, xuanji_pages=(274,277))

W("housing-question", "论住宅问题",
  "弗·恩格斯", None, "1872-1873", "article",
  ["scientific_socialism", "political_economy"],
  ["住宅问题", "城市", "资本主义"],
  xuanji_vol=3, xuanji_pages=(188,273))

W("letter-to-bebel-1875", "给奥·倍倍尔的信",
  "弗·恩格斯", None, "1875", "letter",
  ["scientific_socialism"],
  ["哥达纲领", "德国社会民主党"],
  xuanji_vol=3, xuanji_pages=(341,351))

W("bakunin-state-anarchism-summary", "巴枯宁《国家制度和无政府状态》一书摘要",
  "卡·马克思", None, "1874-1875", "manuscript",
  ["scientific_socialism"],
  ["巴枯宁批判", "无政府主义", "无产阶级专政"],
  xuanji_vol=3, xuanji_pages=(329,340))

W("french-worker-party-program", "法国工人党纲领导言（草案）",
  "卡·马克思", None, "1880", "essay",
  ["scientific_socialism"],
  ["法国工人党", "纲领"],
  xuanji_vol=3, xuanji_pages=(649,652))

W("letter-to-zasulich", "给维·伊·查苏利奇的复信",
  "卡·马克思", None, "1881", "letter",
  ["scientific_socialism", "philosophy"],
  ["俄国农村公社", "历史唯物主义"],
  xuanji_vol=3, xuanji_pages=(653,666))

W("marx-graveside-speech", "在马克思墓前的讲话",
  "弗·恩格斯", None, "1883", "speech",
  ["scientific_socialism"],
  ["马克思生平"],
  quotes=["正像达尔文发现有机界的发展规律一样，马克思发现了人类历史的发展规律"],
  xuanji_vol=3, xuanji_pages=(667,668))


# ═══════════════════════════════════════════════════════════════════
# 选集第四卷 额外篇目 (mes04.pdf) — article_map 完全损坏，手动标注
# ═══════════════════════════════════════════════════════════════════

W("marx-new-rhine-paper", "马克思和《新莱茵报》",
  "弗·恩格斯", None, "1884", "article",
  ["scientific_socialism"],
  ["新莱茵报", "马克思"],
  xuanji_vol=4, xuanji_pages=(1,7))

W("marx-rodbertus", "马克思和洛贝尔图斯",
  "弗·恩格斯", None, "1885", "essay",
  ["political_economy"],
  ["剩余价值", "洛贝尔图斯"],
  xuanji_vol=4, xuanji_pages=(8,15))

W("history-communist-league-sel", "关于共产主义者同盟的历史",
  "弗·恩格斯", None, "1885", "article",
  ["scientific_socialism"],
  ["共产主义者同盟"],
  xuanji_vol=4, xuanji_pages=(16,35))

W("ludwig-feuerbach-sel", "路德维希·费尔巴哈和德国古典哲学的终结",
  "弗·恩格斯", None, "1886", "book",
  ["philosophy"],
  ["费尔巴哈", "德国古典哲学", "黑格尔"],
  xuanji_vol=4, xuanji_pages=(36,97))

W("peasant-question-france-germany-sel", "法德农民问题",
  "弗·恩格斯", None, "1894", "essay",
  ["scientific_socialism"],
  ["农民问题", "合作社"],
  xuanji_vol=4, xuanji_pages=(98,118))

W("letters-selected-xuanji-4", "书信选编",
  "卡·马克思", "弗·恩格斯", "1842-1895", "letter",
  ["philosophy", "political_economy", "scientific_socialism"],
  ["历史唯物主义书信", "晚年书信"],
  xuanji_vol=4, xuanji_pages=(119,668), xuanji_is_full=True,
  note="选集精选102封书信")


# ═══════════════════════════════════════════════════════════════════
# 选集第二卷 额外篇目 (mes02.pdf)
# ═══════════════════════════════════════════════════════════════════

W("engels-review-capital-vol1", "卡·马克思《资本论》第一卷书评",
  "弗·恩格斯", None, "1868", "article",
  ["political_economy"],
  ["资本论", "书评"],
  xuanji_vol=2, xuanji_pages=(70,78))


# ── Build final JSON ─────────────────────────────────────────────

# Sort by 文集 volume then page
def sort_key(w):
    for ek, ev in w["editions"].items():
        if ek.startswith("wenji"):
            vol = int(ek.split("_v")[1])
            return (vol, ev["start_page"] or 0)
    # Works only in 选集
    for ek, ev in w["editions"].items():
        if ek.startswith("xuanji"):
            vol = int(ek.split("_v")[1]) + 10  # after 文集 volumes
            return (vol, ev["start_page"] or 0)
    return (99, 0)

WORKS.sort(key=sort_key)

EDITION_VOLUMES = {
    "wenji": {
        "name": "马克思恩格斯文集", "publisher": "人民出版社", "year": 2009,
        "volumes": 10, "isbn": "9787010076534",
        "volume_map": {
            "1": {"source": "mea01.pdf", "cover": "1843—1848年著作"},
            "2": {"source": "mea02.pdf", "cover": "1848—1859年著作"},
            "3": {"source": "mea03.pdf", "cover": "1864—1883年著作"},
            "4": {"source": "mea04.pdf", "cover": "1884—1895年著作"},
            "5": {"source": "mea05.pdf", "cover": "资本论 第一卷"},
            "6": {"source": "mea06.pdf", "cover": "资本论 第二卷"},
            "7": {"source": "mea07.pdf", "cover": "资本论 第三卷"},
            "8": {"source": "mea08.pdf", "cover": "资本论手稿选编"},
            "9": {"source": "mea09.pdf", "cover": "反杜林论 / 自然辩证法"},
            "10": {"source": "mea10.pdf", "cover": "书信选编"},
        },
    },
    "xuanji": {
        "name": "马克思恩格斯选集", "publisher": "人民出版社", "year": 2012,
        "edition": "第3版", "volumes": 4, "isbn": "9787010106861",
        "volume_map": {
            "1": {"source": "mes01.pdf", "cover": "1843—1859年著作"},
            "2": {"source": "mes02.pdf", "cover": "政治经济学专卷"},
            "3": {"source": "mes03.pdf", "cover": "1864—1883年著作"},
            "4": {"source": "mes04.pdf", "cover": "1884—1895年著作 / 书信选编"},
        },
    },
}

output = {
    "$schema": "work_catalog_v2",
    "description": "MarxOS 著作元数据目录 — 覆盖《文集》10卷 +《选集》4卷全部主要著作",
    "version": "2.0.0",
    "editions": EDITION_VOLUMES,
    "discipline_taxonomy": {
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
            "sub_categories": ["阶级斗争", "无产阶级革命", "国家理论", "共产主义理论", "政党理论", "农民问题", "民族问题"],
        },
    },
    "works": WORKS,
}

out_path = ROOT / "rag/work_catalog.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Stats
total = len(WORKS)
wenji_only = sum(1 for w in WORKS if any(k.startswith("wenji") for k in w["editions"]) and not any(k.startswith("xuanji") for k in w["editions"]))
xuanji_only = sum(1 for w in WORKS if any(k.startswith("xuanji") for k in w["editions"]) and not any(k.startswith("wenji") for k in w["editions"]))
both = total - wenji_only - xuanji_only
all_concepts = set()
for w in WORKS:
    all_concepts.update(w["concepts"])
phil = sum(1 for w in WORKS if "philosophy" in w["discipline"])
econ = sum(1 for w in WORKS if "political_economy" in w["discipline"])
socs = sum(1 for w in WORKS if "scientific_socialism" in w["discipline"])
with_quotes = sum(1 for w in WORKS if w.get("quotes"))

print(f"Wrote {total} works to {out_path}")
print(f"  文集 + 选集 both: {both}")
print(f"  文集 only: {wenji_only}")
print(f"  选集 only: {xuanji_only}")
print(f"  philosophy: {phil}, political_economy: {econ}, scientific_socialism: {socs}")
print(f"  unique concepts: {len(all_concepts)}")
print(f"  works with quotes: {with_quotes}")
