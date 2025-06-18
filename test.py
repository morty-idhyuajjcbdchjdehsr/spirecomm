import itertools
import datetime
import json
import logging
import sys

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass


if __name__ == "__main__":


    llm = ChatOpenAI(model="doubao-seed-1-6-flash-250615", temperature=0.3)
    print(ChatOpenAI(model="doubao-seed-1-6-flash-250615", temperature=0.3).model_name)
    print(ChatOllama(model="gemma3:1b", temperature=1.0, top_p=0.95, top_k=64).model)
    print(isinstance(llm,ChatOpenAI))


    # print(response["messages"])
    # output_dict = output_parser.parse(response["messages"][-1].content)
    # print(output_dict.get('cardName'))