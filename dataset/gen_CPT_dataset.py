import json

def convert_card_to_text(entry: dict) -> str:
    """将结构化 entry 转换为单条连续文本"""
    lines = []
    lines.append(f"Title: {entry.get('title','').strip()}")

    # summary
    if entry.get("summary"):
        lines.append("\nSummary:\n" + entry["summary"].strip())

    # Infobox
    if "Infobox" in entry.get("sections", {}):
        infoboxes = entry["sections"]["Infobox"]
        for ibox in infoboxes:
            fields = ibox.get("fields", {})
            if fields:
                lines.append("\nInfobox:")
                for k, v in fields.items():
                    lines.append(f"- {k}: {v.strip()}")

    # other sections
    for sec_name, sec_content in entry.get("sections", {}).items():
        if sec_name == "Infobox":
            continue
        lines.append(f"\n### {sec_name}")
        if isinstance(sec_content, list):
            for item in sec_content:
                lines.append(f"- {item.strip()}")
        elif isinstance(sec_content, dict):
            # subsection 结构
            for sub_name, sub_items in sec_content.items():
                lines.append(f"\n#### {sub_name}")
                for item in sub_items:
                    lines.append(f"- {item.strip()}")

    return "\n".join(lines).strip()


def convert_structured_to_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            text = convert_card_to_text(entry)
            if not text.strip():
                continue
            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成，共 {len(dataset)} 条，保存到 {output_path}")


if __name__ == "__main__":
    convert_structured_to_jsonl(
        "slay_the_spire_structured_knowledge.json",
        "slay_the_spire_cpt.jsonl"
    )
