import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ======================
# 目标网页
# ======================

BASE_PAGE = "https://www.marxists.org/chinese/pdf/me-old.htm"


# ======================
# 保存目录
# ======================

SAVE_DIR = "data/marx_engels全集"

os.makedirs(SAVE_DIR, exist_ok=True)


# ======================
# 请求网页
# ======================

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(BASE_PAGE, headers=headers)

html = response.text


# ======================
# 解析HTML
# ======================

soup = BeautifulSoup(html, "html.parser")


# ======================
# 找所有 PDF 链接
# ======================

pdf_links = []

for a in soup.find_all("a"):

    href = a.get("href")

    if isinstance(href, str) and href.endswith(".pdf"):

        full_url = urljoin(BASE_PAGE, href)

        pdf_links.append(full_url)


print(f"\n发现 PDF 数量：{len(pdf_links)}\n")


# ======================
# 下载 PDF
# ======================

for url in pdf_links:

    filename = url.split("/")[-1]

    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path):

        print(f"已存在：{filename}")

        continue

    print(f"正在下载：{filename}")

    try:

        pdf_response = requests.get(
            url,
            headers=headers,
            timeout=60
        )

        if pdf_response.status_code == 200:

            with open(save_path, "wb") as f:

                f.write(pdf_response.content)

            print(f"下载完成：{filename}")

        else:

            print(f"下载失败：{filename}")

    except Exception as e:

        print(f"错误：{filename}")
        print(e)


print("\n===== 全部完成 =====")