import requests
from bs4 import BeautifulSoup, Tag, NavigableString
import json
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
BASE_URL = "https://slay-the-spire.fandom.com"
CATEGORY_URL = "/wiki/Category:Boss_Monster"
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
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return None
        return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

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

def extract_table_text(table):
    rows = table.find_all("tr")
    table_lines = []
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) != len(headers):
            continue
        line = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells))
        table_lines.append(line)

    return table_lines

def extract_value_text(value):
    """提取 pi-data-value 中所有文本内容，使用换行符连接"""
    texts = []
    # 先提取 div 的 own text（不含子标签）
    if value.contents:
        for content in value.contents:
            # print(content)
            if isinstance(content, str):
                texts.append(content)
            elif content.name == "p":
                texts.append('\n'+content.get_text(strip=False)+'\n')
            else:
                texts.append(content.get_text(strip=False))

    return "".join(texts)

def extract_infoboxes(soup):
    """提取所有 portable-infobox 为结构化字段"""
    infoboxes = []

    for aside in soup.find_all("aside", class_="portable-infobox"):
        box = {}

        # 标题
        title_el = aside.find("h2", class_="pi-title")
        if title_el:
            box["title"] = title_el.text.strip()


        # 字段
        fields = {}
        for item in aside.select("div.pi-item.pi-data"):
            label = item.find("h3", class_="pi-data-label")
            value = item.find("div", class_="pi-data-value")
            if label and value:
                clean_label = label.get_text(strip=True)
                # 多段落合并
                # if value.find("p"):
                #     paragraphs = value.find_all("p")
                #     clean_value = "\n".join(p.get_text(strip=True) for p in paragraphs)
                # else:
                #     clean_value = value.get_text(strip=True)
                clean_value = extract_value_text(value)
                fields[clean_label] = clean_value
        if fields:
            box["fields"] = fields

        infoboxes.append(box)

    return infoboxes

def extract_structured_content(soup, page_url):
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        return {}

    structured = {
        "title": soup.find("h1").text.strip(),
        "summary": "",
        "sections": {},
        "url": page_url
    }

    # ✅ 添加 Infobox 提取结果为 sections["Infobox"]
    infoboxes = extract_infoboxes(soup)
    if infoboxes:
        structured["sections"]["Infobox"] = infoboxes

    # ✅ 提取首段 summary
    summary_parts = []
    for child in content_div.children:
        # print(f"child is: \n{child}\n\n")
        if isinstance(child, Tag) and child.name=="p":
            for c in child.children:
                    # print(f"sub child is: \n{c}\n\n")
                    if c.name in ["aside"]:
                        continue
                    if c.name == "ul":
                        for li in c.find_all("li"):
                            text = li.get_text(strip=False)
                            if text:
                                summary_parts.append(text)
                    else:
                        text = c.get_text(strip=True)
                        if text:
                            summary_parts.append(text)
        elif child.name =="ul":
            for li in child.find_all("li"):
                text = li.get_text(strip=False)
                if text:
                    summary_parts.append(text)
        elif child.name in ["table","h2","h3"]:
            break
    structured["summary"] =" ".join(summary_parts)

    # ✅ 提取正文：支持主标题 + 子标题嵌套
    current_section = None
    current_subsection = None
    section_buffer = {}

    def add_buffer():

        if not current_section or not buffer:
            return

        if current_subsection:
            if not isinstance(section_buffer.get(current_section), dict):
                section_buffer[current_section] = {}
            section_buffer[current_section][current_subsection] = buffer[:]
        else:
            section_buffer[current_section] = buffer[:]

    buffer = []
    for element in content_div.children:
        if element.name == "h2":
            add_buffer()
            current_section = element.get_text(strip=True).replace("[edit]", "").replace("[]", "").strip()
            current_subsection = None
            buffer = []
        elif element.name == "h3":
            add_buffer()
            current_subsection = element.get_text(strip=True).replace("[edit]", "").replace("[]", "").strip()
            buffer = []
        elif element.name in ["p", "li"]:
            text = element.get_text(strip=False)
            if text and len(text) > 10:
                buffer.append(text)
        elif element.name == "ul":
            for li in element.find_all("li"):
                text = li.get_text(strip=False)
                if text:
                    buffer.append(text)
        elif element.name == "table":
            rows = extract_table_text(element)
            buffer.extend(rows)

    add_buffer()  # flush last

    structured["sections"].update(section_buffer)

    return structured

def main():
    dataset = []
    for cat_name, cat_url in CATEGORIES.items():
        print(f"Fetching category: {cat_name}")
        links = extract_links_from_category(cat_url)
        # print(f"发现 {len(links)} 个页面，开始抓取...")


        for title, url in tqdm(links):
            soup = fetch_page(url)
            if not soup:
                continue
            print(f"extracting url:{url}")
            structured = extract_structured_content(soup, url)
            if structured:
                dataset.append(structured)
            # break # 只执行一次

    # 保存 JSON 文件
    with open("dataset/slay_the_spire_structured_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"抓取完成，共 {len(dataset)} 条结构化数据。")

if __name__ == "__main__":
    main()
