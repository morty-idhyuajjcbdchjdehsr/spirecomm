import copy
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # cnm


# 自定义你的修改逻辑
def modify_assistant_message(messages: []) -> str:
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)

    t_messages = copy.deepcopy(messages)
    t_messages.append({"role": "user", "content": "add explanation to your response,"
                                                  "limit your explanation to 100 words."})
    response = llm.invoke(t_messages)
    print(f"content is:\n{response.content}\n\n")
    explanation = response.content
    return explanation


input_file = "dataset_human_act2.jsonl"
output_file = "dataset_human_act2_modified.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for index,line, in enumerate(fin):
        item = json.loads(line)
        messages = item.get("conversations", [])

        # 遍历找到 role 为 "assistant" 的消息
        for message in messages:
            if message.get("role") == "assistant":
                message["role"] = "ai"
                response_text = message["content"]
                # print(f"ori text is:\n{response_text}\n")
                message["content"] = modify_assistant_message(messages)
                message["role"] = "assistant"

        # 写入修改后的数据
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"data {index} write over.")
        # break

print("处理完成，已保存至", output_file)
