import json
import re
import time
from collections import deque

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
import os
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from spirecomm.spire.card import Card, CardType
from spirecomm.spire.character import Monster, Intent
from spirecomm.spire.relic import Relic

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"


def get_lists_str(lists):
    ret = "[ "
    for index, item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists) - 1:
            ret += ", "
    ret += " ]"
    return ret

def get_card_lists_str(lists):
    ret = "[ "
    for index, item in enumerate(lists):
        type = ""
        if item.type == CardType.ATTACK:
            type = "ATTACK"
        if item.type == CardType.SKILL:
            type = "SKILL"
        if item.type == CardType.POWER:
            type = "POWER"
        if item.type == CardType.STATUS:
            type = "STATUS"
        if item.type == CardType.CURSE:
            type = "CURSE"
        ret += f"{item.name}({item.cost},{type})"
        if index != len(lists) - 1:
            ret += ", "
    ret += " ]"
    return ret

class State(TypedDict):
    messages: Annotated[list, add_messages]
    floor: int
    turn: int
    current_hp: int
    max_hp: int
    block: int
    energy: int
    relics: list
    hand: list
    monsters: list
    drawPile: list
    discardPile: list
    powers: list
    orbs: list
    potion: list

    num_cards: int
    current_action: str
    available_cards: list



class HandSelectAgent:
    def __init__(self,battle_rounds_info, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):

        self.error_invoke_cnt = 0
        self.total_invoke_cnt = 0
        self.card_indices = None
        self.chosen_cards = None
        self.available_cards = None
        self.num_cards = None

        self.router2_cnt = 0
        self.humanM = None

        self.explanation = None
        self.card_Index = -1

        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        card_index_schema = ResponseSchema(
            name="cardIndices",
            description="The list of index of cards you choose from **Available Cards**, "
                        "index ranges from 0 to len(available_cards)-1",
            type="List[int]"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you take the action."
        )
        # 将所有 schema 添加到列表中
        response_schemas = [
            card_index_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.output_parser = output_parser

        tools1 = []
        tool_node1 = ToolNode(tools1)
        graph_builder = StateGraph(State)
        graph_builder.add_node("Tool", tool_node1)
        graph_builder.add_node("LLM", self.llm_1)
        graph_builder.add_node("Check", self.outputChecker)

        graph_builder.add_edge(START, "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", "Check"])
        graph_builder.add_edge("Tool", "LLM")
        graph_builder.add_conditional_edges("Check", self.router2, ["LLM", END])

        self.graph = graph_builder.compile()

        self.previous_rounds_info = battle_rounds_info

    def llm_1(self, state: State):

        outputFormat = self.output_parser.get_format_instructions()
        current_action = state["current_action"]
        num_cards = state["num_cards"]
        Task = {
            "DiscardAction":"Now your task is to Discard one Card from your Hand Pile.",
            "ArmamentsAction":"Now you have Played Card 'Armaments',your task is to Choose one Card "
                              "from your Hand Pile to upgrade",
            "RetainCardsAction":"Now your task is to Retain one Card from your Hand Pile",
            "BetterDiscardPileToHandAction":"Now your task is to Choose one Card from your DiscardPile to HandPile.",
            "PutOnDeckAction":"Now your task is to Choose one Card from your HandPile "
                              "to put on the Top of your DrawPile.",
            "GamblingChipAction":"Now you have triggered the effect of Gambling Chip. You need to discard any "
                                 "number of cards from your HandPile,then draw that many.",
            "RecycleAction":"Now you have Played Card 'Recycle',you need to choose one card "
                            "from your HandPile to exhaust and gain energies equal to its cost.",
            "BetterDrawPileToHandAction":"Now you need to Choose 2 cards from your DrawPile to your HandPlie.",
            "DiscardPileToTopOfDeckAction":"Now you need to Choose one card from your DiscardPile to put on the top"
                                           " of your DrawPile.",
            "ExhaustAction":"Now you need to Choose one Card from your HandPile to exhaust.",
            "SetupAction":"You have played Card 'Setup',now you need to "
                          "put a card from your handPile on top of your draw pile.It costs 0 until played.",
            "DualWieldAction":"You have played Card 'Dual Wield'.now you need to Choose an Attack or Power card from "
                              "your HandPile. It will then "
                              "Add a copy of that card into your handPile."
        }

        system_msg_2 = f"""You are an AI designed to play *Slay the Spire* as {self.role} and make optimal card choices 
during combat.  {Task[current_action]}.
You will be provided the following context.

### Battle situation
provide situation of current battle to help you make card choice.

- **Floor**: 'floor'
- **Turn Number**: 'turn_number'
- **Current HP**: 'current_hp' / 'max_hp'
- **Block**: 'block'
- **Energy Available**: 'energy'
- **Relics**: [ Relic ]
- **Enemy List**: [ Enemy ]  Enemy format: "enermy_name( enermy_hp,enemy_intent,enemy_block,[enemy_status])"
- **Hand Pile**: [ Card ]  Card format: "card_name( card_cost,is_card_has_target,card_type )"
- **Draw Pile**: [ Card ]
- **Discard Pile**: [ Card ]
- **Player Status**: [ player_status ]
- **Potion**: [ Potion ] Potion format: "potion_name(is_potion_has_target)"

### Previous Actions (This Turn):
A list of prior actions during this turn:  
[ 
  {{ turn: int, operation: str }}, //first action in this turn
  {{ turn: int, operation: str }} // Example: {{ turn: 3, operation: choose card 'Defend'  }}
  .......
  {{ turn: int, operation: str }}, // last action in this turn
]

### Available Cards:
list of cards to choose from
[ Card ]

### Notice:
things you should be aware of

### Response Format:
{outputFormat}
"""

        self.sys_prompt = system_msg_2
        messages = [{"role": "system", "content": system_msg_2}] + state["messages"]
        return {
            **state,  # 保留原 state 的所有属性
            "messages": [self.llm.invoke(messages)]
        }

    def router1(self, state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "Tool"
        else:
            return "Check"

    def outputChecker(self, state: State):
        response_text = state["messages"][-1].content
        # print("response_text is:",response_text)
        start = response_text.rfind('```json') + len('```json\n')
        end = response_text.rfind('```')
        json_text = response_text[start:end].strip()

        # 得到最终的 json格式文件
        try:
            jsonfile = json.loads(json_text)
            self.card_indices = jsonfile.get('cardIndices')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate your answer!"}]
            }
        available_cards = state["available_cards"]
        num_cards = state["num_cards"]

        self.chosen_cards = []

        if isinstance(self.card_indices,list):
            if num_cards!=99 and len(self.card_indices) != num_cards:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": f"You should provide {num_cards} cardIndex,"
                                                             " please regenerate your answer!"}]
                }
            if len(self.card_indices) > len(available_cards):
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": f"You have provided {len(self.card_indices)} cardIndex,"
                                                             f"which exceed the length of available_cards({len(available_cards)})"
                                                             " please regenerate your answer!"}]
                }

            for index in self.card_indices:
                if isinstance(index,int):
                    if 0 <= index < len(available_cards):
                        self.chosen_cards.append(available_cards[index])
                    else:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": f"Your provided cardIndex {index} is out of range"
                                                                     f"(0~{len(available_cards)-1}),"
                                                                     " please regenerate your answer!"}]
                        }
                else:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": f"Your provided cardIndex should be Int,"
                                                                 " please regenerate your answer!"}]
                    }
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "cardIndices should be a list of Int,"
                                                         " please regenerate your answer!"}]
            }


        # check pass!
        return {
            **state,
            "messages": [AIMessage(content="output check pass!!")]
        }

    def router2(self, state: State):
        messages = state["messages"]
        last_message = messages[-1]
        self.total_invoke_cnt += 1
        if type(last_message) == AIMessage:
            return END
        else:
            self.error_invoke_cnt += 1
            self.router2_cnt += 1
            with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 2:

                self.card_indices = None
                self.chosen_cards = None
                self.explanation = 'router2 reach recursion limit! use algorithm to choose card!'
                return END
            return "LLM"

    def invoke(self, floor: int, turn: int,
               current_hp: int,
               max_hp: int,
               block: int,
               energy: int,
               relics: list,
               hand: list,
               monsters: list,
               drawPile: list,
               discardPile: list,
               powers: list,
               orbs: list,
               potion: list,


               num_cards:int,
               current_action:str,
               available_cards:list,
               config=None):
        start_time = time.time()  # 记录开始时间

        # 获取当前turn的所有action
        previous_rounds_info = '['
        for item in list(self.previous_rounds_info):

            tmp_turn = item['turn']
            tmp_floor = item['floor']
            if tmp_turn == turn and tmp_floor == floor:
                tmp_str = f"{{ turn:{tmp_turn},operation:{item['operation']} }}"
                previous_rounds_info += (tmp_str + '\n')
        previous_rounds_info += ']'

        self.router2_cnt = 0

        notice = ""
        if current_action=="DiscardAction":
            for card in available_cards:
                if card.name=="Tactician":
                    notice += ("You have card 'Tactician' in Available Cards,"
                               "If this card is discarded, you will gain 1 energy.\n")
                if card.name=="Tactician+":
                    notice += ("You have card 'Tactician+' in Available Cards,"
                               "If this card is discarded, you will gain 2 energy.\n")
                if card.name=="Reflex":
                    notice += ("You have card 'Reflex' in Available Cards,"
                               "If this card is discarded, you will draw 2 cards.\n")
                if card.name=="Reflex+":
                    notice += ("You have card 'Reflex+' in Available Cards,"
                               "If this card is discarded, you will draw 3 cards.\n")



        template_string = """        
Battle Situation:
        **Floor**: {floor}, 
        **Turn Number**: {turn}, 
        **Current HP**: {hp},
        **Block**: {block},
        **Energy Available**: {energy},
        **Relics**:{relics},
        **Hand pile**(the cards in your hand): {hand},
        **Enemy Lists**:{monsters},
        **Draw Pile**(the cards in draw pile): {drawPile},
        **Discard Pile**(the cards in discard pile):{discardPile},
        **Player Status**(list of player status):{pStatus}
        **Potion**:{potion}
        **Orbs**(if you are DEFECT): {orbs}

Previous actions in this turn:
{previous_rounds_info}

Available Cards:
{available_cards}

Notice:
{notice}

now give the response.
"""
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            floor=floor,
            turn=turn,
            hp=f"{current_hp}/{max_hp}",
            block=block,
            energy=energy,
            relics=get_lists_str(relics),
            hand=get_lists_str(hand),
            monsters=get_lists_str(monsters),
            drawPile=get_lists_str(drawPile),
            discardPile=get_lists_str(discardPile),
            pStatus=get_lists_str(powers),
            # output_format=outputFormat,
            orbs=get_lists_str(orbs),
            potion=get_lists_str(potion),
            available_cards=get_card_lists_str(available_cards),
            notice=notice,
            previous_rounds_info=previous_rounds_info
        )
        self.humanM = messages[0].content
        state = State(messages=messages, turn=turn, current_hp=current_hp, max_hp=max_hp,
                      block=block, energy=energy, relics=relics, hand=hand,
                      monsters=monsters, drawPile=drawPile, discardPile=discardPile,
                      powers=powers, orbs=orbs, potion=potion, floor=floor, num_cards=num_cards,current_action=current_action,available_cards=available_cards)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时


        #添加round info
        if self.chosen_cards is None:
            operation = ''
        elif len(self.chosen_cards)==1:
            operation = f"choose Card '{self.chosen_cards[0].name}' for {current_action}"
        else:
            card_str = '['
            for card in self.chosen_cards:
                card_str += f"'{card.name}',"
            card_str += ']'
            operation = f"choose Cards {card_str} for {current_action}"

        round_info = {
            'floor': floor,
            'turn': turn,
            'operation': operation
        }
        self.previous_rounds_info.append(round_info)

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\hand_select_agent.txt', 'a') as file:
            file.write('--------------round start-------------------------\n')
            file.write("System:\n" + self.sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write(f"error rate:{(float(self.error_invoke_cnt) / self.total_invoke_cnt) * 100 :.3f}%\n")
            file.write('--------------round end-------------------------\n')

        return result



