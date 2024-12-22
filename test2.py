import itertools
import datetime
import logging
import sys

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass


if __name__ == "__main__":
    agent = SimpleAgent()
    agent.init_llm_env()
    agent.change_class(PlayerClass.IRONCLAD)
    print(agent.search_card("Decay"))