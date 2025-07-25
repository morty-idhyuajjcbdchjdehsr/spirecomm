from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

chat = ChatOllama(
    model="my-mistral",  # 你在服务器上运行的模型名称
    base_url="http://211.71.15.50:11434"  # 你的服务器 IP 和端口
)

response = chat.invoke([
    HumanMessage(content="Hello! What is the capital of France?")
])

print(response.content)
