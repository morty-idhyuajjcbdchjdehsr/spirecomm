import requests
from bs4 import BeautifulSoup
import json
import re
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URL = "https://slay-the-spire.fandom.com"
CATEGORIES = {

    "Class":"/wiki/Category:Class",
    "Enemies": "/wiki/Category:Monster",
    "Elites":"/wiki/Category:Elite",
    "Bosses":"/wiki/Category:Boss_Monster",
    "IronClad Cards": "/wiki/Category:Ironclad_Cards",
    "Silent Cards": "/wiki/Category:Silent_Cards",
    "Defect Cards": "/wiki/Category:Defect_Cards",
    "Watcher Cards": "/wiki/Category:Watcher_Cards",
    "Status Cards":"/wiki/Category:Status_Cards",
    "Curse Cards":"/wiki/Category:Curse_Cards",
    "Colorless Cards":"/wiki/Colorless_Cards"
}

def fetch_page(url):
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"[ERROR] Failed to fetch: {url}")
        return None
    return BeautifulSoup(resp.text, "html.parser")

def extract_links_from_category(category_url):
    soup = fetch_page(BASE_URL + category_url)
    links = []
    if not soup:
        return links

    for a in soup.select("a.category-page__member-link"):
        href = a.get("href")
        title = a.get("title")
        if href and title:
            links.append((title, BASE_URL + href))
    return links

# def extract_main_text(soup):
#     # Get the main content text
#     content_div = soup.find("div", {"class": "mw-parser-output"})
#     if not content_div:
#         return ""
#
#     paragraphs = content_div.find_all("p")
#     text = ""
#     for p in paragraphs:
#         cleaned = p.get_text().strip()
#         if len(cleaned) > 30:
#             text += cleaned + "\n"
#     return text.strip()

def extract_main_text(soup):
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        return ""

    text_parts = []

    for element in content_div.children:
        # 提取段落
        if element.name == "p":
            para = element.get_text().strip()
            if para and len(para) > 30:
                text_parts.append(para)

        # 提取小标题
        elif element.name in ["h2", "h3"]:
            heading = element.get_text().strip()
            if heading and not heading.lower().startswith("navigation"):
                text_parts.append(f"\n【{heading}】")

        # 提取表格信息
        elif element.name == "table":
            table_text = extract_table_text(element)
            if table_text:
                text_parts.append(table_text)

        # 提取列表
        elif element.name in ["ul", "ol"]:
            list_items = element.find_all("li")
            list_text = "\n".join(f"- {li.get_text().strip()}" for li in list_items if li.get_text().strip())
            if list_text:
                text_parts.append(list_text)

    return "\n\n".join(text_parts).strip()

def extract_table_text(table):
    rows = table.find_all("tr")
    table_lines = []

    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) != len(headers):
            continue
        line = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells))
        table_lines.append(f"- {line}")

    return "\n".join(table_lines) if table_lines else ""

def generate_qa(title, content):
    return {
        "conversations": [
            {
                "role": "user",
                "content": f"What knowledge is there about '{title}' in Slay the Spire?"
            },
            {
                "role": "assistant",
                "content": content
            }
        ]
    }



def main():
    dataset = []
    for cat_name, cat_url in CATEGORIES.items():
        print(f"Fetching category: {cat_name}")
        links = extract_links_from_category(cat_url)
        for title, url in tqdm(links):
            soup = fetch_page(url)
            if not soup:
                continue
            content = extract_main_text(soup)
            if content:
                qa = generate_qa(title, content)
                dataset.append(qa)

    # 保存为 JSONL 格式
    with open("dataset/slay_the_spire_knowledge_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据集生成完成，共 {len(dataset)} 条问答样本。")
    print(f"dataset[0]:{dataset[0]['conversations'][1]['content']}")

if __name__ == "__main__":
    main()
