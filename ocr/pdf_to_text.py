from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import re

# 定义文本清洗函数
def clean_text(text):

    # 去掉多余换行
    text = text.replace("\n", "")

    # 去掉多余空格
    text = re.sub(r"\s+", "", text)

    return text
# 1 初始化OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=False
)

# 2 PDF转图片
pages = convert_from_path(
    "data/capital.pdf",
    first_page=220,
    last_page=220,
    poppler_path=r"D:\EdgeDownload\poppler-26.02.0\Library\bin"
)

# 3 OCR识别
# 3. OCR识别
for i, page in enumerate(pages):
    image_path = f"page_{i}.png"
    page.save(image_path, "PNG")

    result = ocr.ocr(image_path, cls=True)

    print("\n===== OCR结果 =====\n")

    all_text = ""

    for line in result[0]:
        text = line[1][0]
        print(text)
        all_text += text

    cleaned_text = clean_text(all_text)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("保存完成")