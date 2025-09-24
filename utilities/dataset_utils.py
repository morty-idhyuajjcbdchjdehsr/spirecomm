import json
import os
from typing import Dict

def is_file_empty(file_path):
    """
    判断文件是否为空
    :param file_path: 文件路径
    :return: 如果文件为空返回True，否则返回False
    """
    # 方法1: 检查文件大小
    # if os.path.getsize(file_path) == 0:
    #     return True

    # 方法2: 尝试读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content.strip():  # 去除空白字符后检查是否为空
            return True

    return False


def copy_file_content(source_path, destination_path):
    """
    将源文件内容复制到目标文件
    :param source_path: 源文件路径
    :param destination_path: 目标文件路径
    """
    try:
        with open(source_path, 'r', encoding='utf-8') as source_file:
            content = source_file.read()

        with open(destination_path, 'a', encoding='utf-8') as dest_file:
            dest_file.write(content)

        # print(f"文件内容已从 '{source_path}' 复制到 '{destination_path}'")

    except FileNotFoundError:
        # print("文件不存在")
        pass
    except Exception as e:
        # print(f"复制文件时发生错误: {e}")
        pass


def clear_file(file_path):
    """
    清空文件内容
    :param file_path: 文件路径
    """
    try:
        # 以写入模式打开文件，这会自动清空文件内容
        with open(file_path, 'w', encoding='utf-8') as file:
            pass  # 不写入任何内容

        # print(f"文件 '{file_path}' 已清空")
        return True

    except Exception as e:
        # print(f"清空文件时发生错误: {e}")
        return False

def export_dataset_item_for_battle_llm(
        llm,
        role: str,
        battle_agent_sys_prompt: str,
        humanM: str,
        result: Dict,
        floor: int,
        dataset_dir: str = r"C:\Users\32685\Desktop\spirecomm\dataset"
):
    """
    导出对话数据到数据集/缓冲文件

    Args:
        llm: 当前使用的 LLM 实例
        role: 当前角色 (如 "Ironclad")
        battle_agent_sys_prompt: 系统提示词
        humanM: 用户输入
        result: 调用结果字典 (必须包含 result["messages"][-2].content)
        floor: 当前楼层数
        dataset_dir: 数据集存储目录
    """
    from langchain_openai import ChatOpenAI  # 避免循环依赖

    if not isinstance(llm, ChatOpenAI):
        return  # 只处理 ChatOpenAI 的情况

    item = {
        "conversations": [
            {"role": "system", "content": battle_agent_sys_prompt},
            {"role": "user", "content": humanM},
            {"role": "assistant", "content": result["messages"][-2].content}
        ]
    }

    m_name = llm.model_name.replace("\\", "-").replace('/', '-')
    buffer_act1 = os.path.join(dataset_dir, "dataset_buffer_act1.jsonl")
    buffer_act2 = os.path.join(dataset_dir, "dataset_buffer_act2.jsonl")

    if 0 <= floor <= 16:
        # 写入第一层 buffer
        with open(buffer_act1, 'a', encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elif 16 < floor <= 33:
        # buffer1 非空 → 转存到 act1
        if not is_file_empty(buffer_act1):
            copy_file_content(buffer_act1, os.path.join(dataset_dir, f"dataset_{m_name}_{role}_act1.jsonl"))
            clear_file(buffer_act1)

        # 写入第二层 buffer
        with open(buffer_act2, 'a', encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    else:
        # buffer2 非空 → 转存到 act2
        if not is_file_empty(buffer_act2):
            copy_file_content(buffer_act2, os.path.join(dataset_dir, f"dataset_{m_name}_{role}_act2.jsonl"))
            clear_file(buffer_act2)

        # 写入 act3
        with open(os.path.join(dataset_dir, f"dataset_{m_name}_{role}_act3.jsonl"), 'a', encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def export_dataset_item(
        dataset_name: str,
        llm,
        sys_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        dataset_dir: str = r"C:\Users\32685\Desktop\spirecomm\dataset"
):
    """
    通用对话导出方法：直接写入指定数据集文件（不分阶段）

    Args:
        dataset_name: 数据集名称（例如 "training_set"）
        llm: 当前使用的 LLM 实例（需要有 model_name 属性）
        sys_prompt: 系统提示词
        user_prompt: 用户输入
        assistant_prompt: ai回答
        dataset_dir: 数据集存储目录
    """
    item = {
        "conversations": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
        ]
    }

    # 处理模型名，避免非法字符
    m_name = getattr(llm, "model_name", "unknown").replace("\\", "-").replace("/", "-")

    # 输出路径
    dataset_path = os.path.join(dataset_dir, f"{dataset_name}.jsonl")

    # 写入文件
    os.makedirs(dataset_dir, exist_ok=True)
    with open(dataset_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dataset_path