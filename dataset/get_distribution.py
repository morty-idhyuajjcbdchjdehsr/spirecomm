import json
from collections import defaultdict

def analyze_sections(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    section_counter = defaultdict(int)
    subsection_counter = defaultdict(lambda: defaultdict(int))

    for entry in data:
        sections = entry.get("sections", {})
        for section_name, section_content in sections.items():


            if isinstance(section_content, dict):  # 有子标题
                for subsection_name in section_content:
                    subsection_counter[section_name][subsection_name] += 1
            else:
                section_counter[section_name] += 1

    # 打印结果
    print("\n=== 📘 Section 分布 ===")
    for section, count in sorted(section_counter.items(), key=lambda x: -x[1]):
        print(f"{section}: {count}")

    print("\n=== 📙 Subsection 分布 ===")
    for section, subsections in subsection_counter.items():
        print(f"\n▶ {section}")
        for subsec, count in sorted(subsections.items(), key=lambda x: -x[1]):
            print(f"   - {subsec}: {count}")

if __name__ == "__main__":
    analyze_sections("slay_the_spire_structured_knowledge.json")

