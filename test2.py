import itertools
import datetime
import logging
import sys
import threading

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


import tkinter as tk
import time


def read_file():
    with open('output.txt', 'r') as file:
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)  # 如果文件没有新内容，等待
                continue
            text_widget.insert(tk.END, line)  # 插入文本
            text_widget.yview(tk.END)  # 滚动到最新文本





if __name__ == "__main__":
    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("实时文本显示")

    # 创建 Text 组件来显示输出
    text_widget = tk.Text(root, wrap=tk.WORD, height=50, width=80)
    text_widget.pack()

    # 启动读取文件的线程
    thread = threading.Thread(target=read_file, daemon=True)
    thread.start()

    # 启动 Tkinter 主循环
    root.mainloop()
