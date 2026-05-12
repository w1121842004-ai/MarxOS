from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

result = ocr.ocr("test.png")

for line in result[0]:
    print(line[1][0])