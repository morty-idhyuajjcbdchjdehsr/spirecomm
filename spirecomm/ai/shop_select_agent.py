import json
import time
from collections import deque

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
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

from spirecomm.communication.action import CancelAction, BuyCardAction, BuyPotionAction, BuyRelicAction, ChooseAction
from spirecomm.spire.card import Card, CardRarity
from spirecomm.spire.character import Monster, Intent
from spirecomm.spire.relic import Relic

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"


def get_lists_str_shop(lists):
    str = "[ "
    for item in lists:
        str += f"{item.name}({item.price})"
        if item != lists[-1]:
            str += ", "
    str = str + " ]"
    return str


def get_lists_str(lists):
    ret = "[ "
    for index, item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists) - 1:
            ret += ", "
    ret += " ]"
    return ret

def get_lists_str_with_only_name(lists):
    ret = "[ "
    for index, item in enumerate(lists):
        ret += item.name
        if index != len(lists) - 1:
            ret += ", "
    ret += " ]"
    return ret


class State(TypedDict):
    messages: Annotated[list, add_messages]
    floor: int
    current_hp: int
    max_hp: int
    deck: list
    c_potions: list
    c_relics: list
    gold: int
    cards: list
    relics: list
    potions: list
    purge_cost: int
    purge_available: bool
    potion_full:bool

class ShopSelectAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):

        self.ret = None
        self.action = None
        self.card_index = None
        self.relic_index = None
        self.potion_index = None
        self.sys_prompt = None

        self.router2_cnt = 0
        self.explanation = None


        self.humanM = None

        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        action_schema = ResponseSchema(
            name="action",
            description="return 'card' if you choose to buy one Card, return 'relic' if you decide to buy one Relic,"
                        "return 'potion' if you choose to buy one Potion, return 'purge' if you decide to purge one card,"
                        "return 'leave' if you choose to leave the shop."
        )
        card_index_schema = ResponseSchema(
            name="cardIndex",
            description="The index of the card you choose to buy(Start with 0); if you don't choose a card, just return -1",
            type="Int"
        )
        relic_index_schema = ResponseSchema(
            name="relicIndex",
            description="The index of the relic you choose to buy(Start with 0); if you don't choose a relic, just return -1",
            type="Int"
        )
        potion_index_schema = ResponseSchema(
            name="potionIndex",
            description="The index of the potion you choose to buy(Start with 0); if you don't choose a potion, just return -1",
            type="Int"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you take the action."
        )
        # 将所有 schema 添加到列表中
        response_schemas = [
            action_schema,
            card_index_schema,
            potion_index_schema,
            relic_index_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.output_parser = output_parser

        tools = load_tools(["wikipedia"], llm=self.small_llm)
        tool_node1 = ToolNode(tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("Tool", tool_node1)
        graph_builder.add_node("LLM", self.llm_1)
        graph_builder.add_node("Check", self.outputChecker)



        graph_builder.add_edge(START, "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", "Check"])
        graph_builder.add_edge("Tool", "LLM")
        graph_builder.add_conditional_edges("Check", self.router2, ["LLM", END])


        self.graph = graph_builder.compile()

    def llm_1(self, state: State):

        outputFormat = self.output_parser.get_format_instructions()
        system_prompt = f"""
You are an AI designed to play *Slay the Spire* as {self.role} and make optimal shopping choices for Player. 
The shop offers 3 types of products,they are Cards, Relics and Potions.you can also pay to Purge one card from
your Deck. There are 5 choice for you to make:
1.buy one Card  2.buy one Relic  3.buy one Potion  4.Purge One Card  5.leave the shop
You will be provided the following context.

### Player Situation:
    - **Floor**: 'floor'
    - **Current Deck**: [ Card ]
    - **Player's Health**: 'current_hp' / 'max_hp'
    - **Current Relics**: [ Relic ]
    - **Current Potions**: [ Potion ]

### Shop Situation:
    - **Gold**: Int (the current gold you have)
    - **Cards**: [ Card ] Card format: "card_name( card_price )"
    - **Relics**: [ Relic ] 
    - **Potions**: [ Potion ]
    - **Purge Cost**: Int (the cost of Purging one card)
                    
### output format:
    {outputFormat}
"""
        self.sys_prompt = system_prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
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
            self.action = jsonfile.get('action')
            self.card_index = jsonfile.get('cardIndex')
            self.relic_index = jsonfile.get('relicIndex')
            self.potion_index = jsonfile.get('potionIndex')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate it!"}]
            }

        shop_cards = state["cards"]
        shop_relics = state["relics"]
        shop_potions = state["potions"]
        purge_cost = state["purge_cost"]
        gold = state["gold"]
        self.ret = None

        has_sozu = False
        for relic in state["c_relics"]:
            if relic.name == "Sozu":
                has_sozu = True

        if self.action == 'leave':
            self.ret = CancelAction()
        elif self.action == 'card':
            if isinstance(self.card_index,int):
                if 0 <= self.card_index < len(shop_cards):
                    c_card = shop_cards[self.card_index]
                    if gold>= c_card.price:
                        self.ret = BuyCardAction(c_card)
                    else:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": f"Your current gold({gold}) is lower than "
                                                                     f"this Card's price({c_card.price})"
                                                                     " please regenerate your answer!"}]
                        }
                else:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": f"Your provided cardIndex is out of range"
                                                                 f"(0~{len(shop_cards) - 1}),"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "cardIndex should be Int,"
                                                             " please regenerate your answer!"}]
                }


        elif self.action == 'potion':
            if isinstance(self.potion_index,int):
                if 0 <= self.potion_index < len(shop_potions):
                    c_potion = shop_potions[self.potion_index]
                    if gold>= c_potion.price:
                        if not state["potion_full"]:
                            if not has_sozu:
                                self.ret = BuyPotionAction(c_potion)
                            else:
                                return {
                                    **state,  # 保留原 state 的所有属性
                                    "messages": [
                                        {"role": "user", "content": f"You have relic 'Sozu',now you can't buy any potions."
                                                                    " please regenerate your answer!"}]
                                }
                        else:
                            return {
                                **state,  # 保留原 state 的所有属性
                                "messages": [{"role": "user", "content": f"Cannot buy potion because potion slots are full."
                                                                         " please regenerate your answer!"}]
                            }
                    else:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": f"Your current gold({gold}) is lower than "
                                                                     f"this Potion's price({c_potion.price})"
                                                                     " please regenerate your answer!"}]
                        }
                else:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": f"Your provided potionIndex is out of range"
                                                                 f"(0~{len(shop_potions) - 1}),"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "potionIndex should be Int,"
                                                             " please regenerate your answer!"}]
                }
        elif self.action == 'relic':
            if isinstance(self.relic_index, int):
                if 0 <= self.relic_index < len(shop_relics):
                    c_relic = shop_relics[self.relic_index]
                    if gold>=c_relic.price:
                        self.ret = BuyRelicAction(c_relic)
                    else:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": f"Your current gold({gold}) is lower than "
                                                                     f"this Relic's price({c_relic.price})"
                                                                     " please regenerate your answer!"}]
                        }
                else:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": f"Your provided relicIndex is out of range"
                                                                 f"(0~{len(shop_relics) - 1}),"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "relicIndex should be Int,"
                                                             " please regenerate your answer!"}]
                }
        elif self.action == 'purge':
            if state["purge_available"] and state["gold"] >= purge_cost:
                self.ret = ChooseAction(name="purge")
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "Now you can't purge!please regenerate your answer!"}]
                }

        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "You should choose your action from ['card','potion','relic',"
                                                         "'purge','leave'],"
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
        if type(last_message) == AIMessage:
            return END
        else:
            self.router2_cnt += 1
            with open(r'C:\Users\32685\Desktop\spirecomm\choose_card_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 3:
                self.action = 'algorithm'
                self.explanation = 'router2 reach recursion limit! use algorithm to choose!'
                self.ret = None
                return END
            return "LLM"

    def invoke(self, floor: int,
               current_hp: int,
               max_hp: int,
               deck: list,
               c_potions:list,
               c_relics:list,
               gold:int,
               cards:list,
               relics:list,
               potions:list,
               purge_cost:int,
               purge_available:bool,
               potion_full:bool,

               config=None):
        start_time = time.time()  # 记录开始时间
        self.router2_cnt = 0

        template_string = """ 
Player Situation:
    - **Floor**:{floor}
    - **Current Deck**: {deck}
    - **Player's Health**: {hp}
    - **Current Relics**: {c_relics}
    - **Current Potions**: {c_potions}
    
Shop Situation:
    - **Gold**: {gold}

    - **Cards**: {cards} 
    - **Relics**: {relics}
    - **Potions**: {potions}
    - **Purge Cost**: {purge_cost}

now give your response.
                """
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            floor=floor,
            hp=f"{current_hp}/{max_hp}",
            deck=get_lists_str(deck),
            c_relics=get_lists_str_with_only_name(c_relics),
            c_potions=get_lists_str_with_only_name(c_potions),
            gold = gold,
            cards = get_lists_str_shop(cards),
            relics = get_lists_str_shop(relics),
            potions = get_lists_str_shop(potions),
            purge_cost = purge_cost,
        )
        self.humanM = messages[0].content
        state = State(messages=messages, floor=floor, current_hp=current_hp, max_hp=max_hp,
                      deck=deck, c_relics=c_relics,c_potions=c_potions,gold=gold,cards=cards,relics=relics,
                      potions=potions,purge_cost=purge_cost,purge_available=purge_available,potion_full=potion_full)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\shop_select_agent.txt', 'a', encoding="utf-8") as file:
            file.write('--------------round start-------------------------\n')
            file.write("System:\n" + self.sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write('--------------round end-------------------------\n')
        return result



