import json

input_path = r'C:\Users\32685\Desktop\spirecomm\dataset\dataset_DeepSeek-V3.jsonl'  # 修改为你的模型名文件
output_path = r'C:\Users\32685\Desktop\spirecomm\dataset\dataset_DeepSeek-V3.jsonl'  # 输出文件路径

with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)

        # 删除第一条 system 消息中的 🔁
        if "conversations" in data and len(data["conversations"]) > 0:
            first_msg = data["conversations"][0]
            if first_msg.get("role") == "system":
                first_msg["content"] = first_msg["content"].replace("🔁", "")

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print("处理完成，结果保存在:", output_path)