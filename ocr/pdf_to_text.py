from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from rag.clean_ocr_text import clean_text


# 1. Initialize OCR.
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    use_gpu=False,
)

# 2. Convert PDF page to image.
pages = convert_from_path(
    "data/capital.pdf",
    first_page=220,
    last_page=220,
    poppler_path=r"D:\EdgeDownload\poppler-26.02.0\Library\bin",
)

# 3. Run OCR.
for i, page in enumerate(pages):
    image_path = f"page_{i}.png"
    page.save(image_path, "PNG")

    result = ocr.ocr(image_path, cls=True)

    print("\n===== OCR result =====\n")

    all_text = ""

    for line in result[0]:
        text = line[1][0]
        print(text)
        all_text += text

    cleaned_text = clean_text(all_text)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("Saved")
