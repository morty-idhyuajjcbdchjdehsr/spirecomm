import json
import re

def slugify(text):
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text).strip().lower()

def generate_natural_question(title, section, subsection=None):
    # 优先使用 subsection 表达
    if subsection:
        if subsection.lower().startswith("move"):
            return f"What moves does {section} perform in combat?"
        if subsection.lower().startswith("pattern"):
            return f"What is the combat pattern of {section}?"
        if section.lower().startswith("cards"):
            return f"What is {subsection} for {title}?"
        if section.lower() in ["strategy", "strategies","gameplay","pattern"]:
            return f"What is the {section} for {title} regarding {subsection}?"
        if section.lower().startswith("enemies that"):
            that =  section.replace("enemies","")
            return f"What is {subsection}{that}?"
        if section.lower().startswith("act"):
            return f"What is '{subsection}' {title} in {section}?"

        return f"What does the '{subsection}' section say in '{section}' of {title}?"
    else:
        if section.lower() == "behavior":
            return f"How does {title} behave during combat?"
        if section.lower() == "strategy":
            return f"What is the strategy to defeat {title}?"
        if section.lower().startswith("act"):
            return f"What is {section} {title}?"

        return f"What is '{section}' of {title}?"
        # return f"What does the '{section}' section tell us about {title}?"

def extract_qa(entry):
    title = entry.get("title", "")
    sections = entry.get("sections", {})
    qa_pairs = []

    summary = entry.get("summary", "")

    # 1. Summary 问答
    if summary:
        qa_pairs.append((
             f"What is {title} in Slay the Spire?",
             summary))

    # 2. Infobox 字段问答
    for box in sections.get("Infobox", []):
        fields = box.get("fields", {})
        for key, value in fields.items():
            # zh_q = f"{title} 的 {key} 是多少？"
            en_q = f"What is the {key} of {title}?"
            # qa_pairs.append({"question": zh_q, "answer": value})
            qa_pairs.append(( en_q, value))

    # 3. Section 和 Subsection
    for section, content in sections.items():
        if section == "Infobox":
            continue  # 忽略 infobox

        if isinstance(content, list):
            answer = "\n".join(content).strip()
            if answer:
                question = generate_natural_question(title, section)
                qa_pairs.append((question, answer))

        elif isinstance(content, dict):
            for subsec, subcontent in content.items():
                answer = "\n".join(subcontent).strip()
                if answer:
                    question = generate_natural_question(title, section, subsec)
                    qa_pairs.append((question, answer))

    return qa_pairs

def convert_to_chatml(entries):
    data = []
    for entry in entries:
        qa_pairs = extract_qa(entry)
        for q, a in qa_pairs:
            data.append({
                "conversations": [
                    {"role": "user", "content": "Answer the question about Slay the Spire:\n"+q},
                    {"role": "assistant", "content": a}
                ]
            })
        # break #只执行一次
    return data

def main():
    input_file = "dataset/slay_the_spire_structured_knowledge.json"
    output_file = "dataset/slay_the_spire_qa_en_only.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    qa_data = convert_to_chatml(entries)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(qa_data)} English-only QA pairs → {output_file}")

if __name__ == "__main__":
    main()
