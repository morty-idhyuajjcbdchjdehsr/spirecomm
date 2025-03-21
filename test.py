import itertools
import datetime
import json
import logging
import sys

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass


if __name__ == "__main__":
    agent = SimpleAgent()
    agent.change_class(PlayerClass.IRONCLAD)
    agent.init_llm_env()
    agent.init_battle_llm()
    agent.init_choose_card_llm()
    # agent.get_play_card_action()



    config = {"configurable": {"thread_id": agent.thread_id}}
    response = agent.choose_card_agent.invoke({"messages": [HumanMessage(content=""" now you need to choose card rewards from **Available Cards**, and the below is the current situation: 
                - **Current Deck:** [Strike+, Strike, Strike, Defend, Defend, Defend+, Defend, Bash, Headbutt, Flex, Twin Strike, Wild Strike+, Spot Weakness, Power Through, Blood for Blood, Sever Soul, Havoc]
                - **Player's Health:** 11/80
                - **Available Cards:** [Brutality, Offering, Bludgeon]
                - **Relic Bowl:** False
                
                now make your choice and provide the reason.                
"""
    )]}, config)

    for response1 in response["messages"]:
        print(type(response1).__name__ + " " + response1.__str__())

    response_text = response["messages"][-1].content
    print(response_text)

    start = response_text.rfind('```json') + len('```json\n')
    end = response_text.rfind('```')
    json_text = response_text[start:end].strip()
    # print(json_text)
    json_data = json.loads(json_text)
    # print(json_data.get('cardName'))







    # print(response["messages"])
    # output_dict = output_parser.parse(response["messages"][-1].content)
    # print(output_dict.get('cardName'))