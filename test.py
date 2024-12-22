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
    # agent.get_play_card_action()
    output_parser = agent.battle_output_parser


    config = {"configurable": {"thread_id": agent.thread_id}}
    response = agent.battle_agent.invoke({"messages": [HumanMessage(content="""You are currently in a combat, and the below is the context:
        **Floor**: 14, 
        **Turn Number**: 3, 
        **Current HP**: 47/80,
        **Block**: 0,
        **Energy Available**: 3,
        **Relics**:[ Burning Blood, Neow's Lament, Tiny Chest,  ],
        **Hand pile**(the cards in your hand): [ Havoc(1,False), Strike(1,True), Strike(1,True), Bash+(2,True), Anger+(0,True)],
        **Enemy Lists**:[ FungiBeast( 7/26 ,Intent.ATTACK 9*1,[ Spore Cloud, Strength,  ]), FungiBeast( 28/28 ,Intent.BUFF,[ Spore Cloud, Strength ])],
        **Draw Pile**(the cards in draw pile): [ Spot Weakness+(1,True), Anger+(0,True), Strike(1,True), Dropkick(1,True), Defend+(1,False), Defend(1,False), Strike+(1,True), Strike(1,True), Hemokinesis(1,True) ],
        **Discard Pile**(the cards in discard pile):[  ],
        **Player Status**(list of player status):[ No Block, Strength,  ]
    
        remember, the cardName you output should not contain parentheses,and can contain '+',which stand for
        upgraded card. for example:
            "Strike","Havoc+","Warcry"
        now take your action and give the response."""
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