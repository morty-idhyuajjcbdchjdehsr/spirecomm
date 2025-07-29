import json
import time
from collections import deque
from dis import Instruction

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

from spirecomm.spire.card import Card, CardRarity, CardType
from spirecomm.spire.character import Monster, Intent
from spirecomm.spire.relic import Relic

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

def get_lists_str(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret

def get_lists_str_with_name(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        ret += item.name
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret

def get_lists_str_for_card(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        rarity = ""
        if item.rarity == CardRarity.BASIC:
            rarity = "BASIC"
        elif item.rarity == CardRarity.COMMON:
            rarity = "COMMON"
        elif item.rarity == CardRarity.UNCOMMON:
            rarity = "UNCOMMON"
        elif item.rarity == CardRarity.RARE:
            rarity = "RARE"
        elif item.rarity == CardRarity.SPECIAL:
            rarity = "SPECIAL"
        else:
            rarity = "CURSE"
        ret += f"{item.name}({rarity},{index})"
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret



class State(TypedDict):
    messages: Annotated[list, add_messages]
    relics:list
    current_hp:int
    max_hp:int
    deck:list
    intent:str
    available_cards:list


class SimpleGridChoiceAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):


        self.small_llm_sys = None
        self.humanM = None
        self.router2_cnt = None
        self.current_deck = None
        self.explanation = None
        self.cardIndex = None
        self.grid_choice_agent_sys_prompt = None
        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        card_indexes_schema = ResponseSchema(
            name="cardIndex",
            description="The index of chosen card from Available Cards(starts with 0)",
            type="Int"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you choose the card."
        )
        response_schemas = [
            card_indexes_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.grid_choice_output_parser = output_parser

        tools = load_tools(["wikipedia"], llm=self.small_llm)
        tool_node1 = ToolNode(tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("Tool", tool_node1)
        graph_builder.add_node("LLM", self.llm_1)
        graph_builder.add_node("Check", self.outputChecker)
        graph_builder.add_node("Suggest", self.suggestionAdder)

        graph_builder.add_edge(START, "Suggest")
        graph_builder.add_edge("Suggest", "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", "Check"])
        graph_builder.add_edge("Tool", "LLM")
        graph_builder.add_conditional_edges("Check", self.router2, ["LLM", END])


        self.graph = graph_builder.compile()

    def llm_1(self, state: State):

        outputFormat = self.grid_choice_output_parser.get_format_instructions()
        Instruction = {
            'upgrade':"""choose the best card to upgrade.
                         1.Consider Card Rarity,Prioritize upgrading higher rarity cards, 
                            as they often provide more powerful effects.
                         2.Focus on Specific Archetypes,Consider cards that align with the current strategy 
                         or archetype you are pursuing (e.g., aggressive, defensive, combo).""",
            'purge':"""1.purge Curse card 
                       2.purge Low-Impact card to improve your deck.(e.q. "Strike","Defend"....)""",
            'transform':"""transform Low-Impact card to improve your deck.(e.q. "Strike","Defend"....)""",
        }
        system_prompt = f"""
                        You are an expert at playing Slay the Spire, and now you are playing as the role {self.role}.
                        now you need to assist in choosing one card from **Available Cards** for **{state["intent"]}**.

                        ### Context:
                        Information provided for you to make better choice.
                        - **Relics**: [Relic],
                        - **Current Deck**: [Card] 
                        - **Player's Health**: 'current_hp' / 'max_hp'
                        
                        ### Available Cards
                        Cards to choose from
                         [ "card_name(card_rarity, card_index)" ]  
                        Where:
                        - `card_rarity`: "BASIC","COMMON","UNCOMMON","RARE","SPECIAL","CURSE" 
                        - `card_index`: int

                        ### Instructions:
                        {Instruction[state["intent"]]}

                        ### output format:
                        {outputFormat}
                        """
        self.grid_choice_agent_sys_prompt = system_prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        return {
            **state,  # 保留原 state 的所有属性
            "messages": [self.llm.invoke(messages)]
        }


    def suggestionAdder(self, state: State):

        if state["intent"]!='upgrade': #只对升级添加建议
            return {
                **state,
                "messages":[]
            }

        context = """
- **Relics**: [Relic],
- **Current Deck:** [Card] 
- **Player's Health:** 'current_hp' / 'max_hp'
- **Available Cards**: 
 [ "card_name(card_rarity, card_index)" ]  
    Where:
    - `card_rarity`: "BASIC","COMMON","UNCOMMON","RARE","SPECIAL","CURSE" 
    - `card_index`: int
"""
        msg = {
            'purge':"""
You are an AI strategist for *Slay the Spire*. Your task is to help guide the purge process by recommending 
which card in the deck should be purged to make the deck more efficient and effective.    

## **Analysis Goals**:
1. **Identify Curse Cards**: Cards in type of CURSE.
2. **Identify Redundant or Weak Cards**: Cards that provide minimal value in combat or overlap too much 
    with other cards.
3. **Prioritize Removal of Low-Impact Cards**: For example, a “Strike” card when there are stronger damage-dealing 
    cards or “Defend” when the deck is more offensive.
""",

            'upgrade':"""
You are an AI strategist for *Slay the Spire*. Your task is to help guide the upgrade process by selecting 
the most impactful card to upgrade, based on the analysis of the deck and the available resources.

###Analysis Goals:
1. **Identify Key Strengths and Weaknesses** in the player’s current deck.
2. **Prioritize the Upgrade of Cards** that maximize combat effectiveness:
   - High damage output (for offense).
   - Strong defensive cards (to mitigate incoming damage).
   - Synergistic upgrades (cards that enhance deck synergy or relic synergies).
3. **Avoid Upgrading Cards** that may not provide a significant benefit (e.g., cards already powerful enough or redundant).
4. **Account for Relics** that modify how upgrades function, like the "Chemical X" relic affecting cost reductions.
""",

            'transform':"""
You are an AI strategist for *Slay the Spire*. Your task is to help guide the transformation process by evaluating 
which card should be transformed to improve the deck's overall effectiveness.

## **Analysis Goals**:
1. **Identify Redundant or Weak Cards**: 
 Cards that provide minimal value in combat or overlap too much with other cards.
2.**Minimize Risk**:
 Avoid transforming cards that are already highly valuable or are essential for the deck’s strategy.
""",
        }


        outputformat = """
You output should include a brief analysis of deck and cards you recommended.
Give **top 3** recommended cards!!   
```json
{
  "deck_analysis": "brief analysis of the deck"
  "recommended_cards": [
    {
      "card_name": "name of card",
      "card_index" "index of card from **Available Cards**",
      "reasoning": "why you recommend this card",
    },
    ....
  ]
}
        """
        system_msg = f"""
{msg[state["intent"]]}

###context:
{context}

### output format:
{outputformat}
"""
        self.small_llm_sys = system_msg
        human_msg = f""" Context:
                - **Relics:** {get_lists_str(state["relics"])}
                - **Current Deck:** {get_lists_str_with_name(state["deck"])}
                - **Player's Health:** {state["current_hp"]}/{state["max_hp"]}
                - **Available Cards**: {get_lists_str_for_card(state["available_cards"])}
                now give your response. """

        messages = [{"role": "system", "content": system_msg}] + [
            HumanMessage(content=human_msg)]
        response = self.small_llm.invoke(messages)
        suggestion_content = response.content

        return {
            **state,  # 保留原 state 的所有属性
            "messages": [AIMessage(content="can you help me analyse the deck and give me some guidance?")] + [
                HumanMessage(content=suggestion_content)]
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
            self.cardIndex = jsonfile.get('cardIndex')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate it!"}]
            }

        self.current_deck = state["deck"]
        available_cards = state["available_cards"]
        chosen_card = None
        if isinstance(self.cardIndex,int) and 0<=self.cardIndex<len(available_cards):
            chosen_card = available_cards[self.cardIndex]
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": f"Your card_Index is out of range(index ranging from 0 to {len(available_cards)-1}),"
                                                         " please regenerate it!"}]
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
            with open(r'C:\Users\32685\Desktop\spirecomm\output\simple_grid_choice_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 2:
                self.cardIndex = 0
                self.explanation = 'router2 reach recursion limit! use algorithm to choose card!'
                return END
            return "LLM"

    def invoke(self, relics:list,
                current_hp:int,
                max_hp:int,
                deck:list,
                intent:str,
                available_cards:list,
                config=None):
        start_time = time.time()  # 记录开始时间
        self.router2_cnt = 0
        template_string = """ 
                        Context
                        - **Relics**:{relics}
                        - **Current Deck:** {deck}
                        - **Player's Health:** {hp}
                        
                        Available Cards:
                        {available_cards}
                        
                        now give your response.
                        """
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            relics=get_lists_str(relics),
            hp=f"{current_hp}/{max_hp}",
            deck=get_lists_str_with_name(deck),
            available_cards=get_lists_str_for_card(available_cards),
        )

        self.humanM = messages[0].content
        state = State(messages=messages, relics=relics, current_hp=current_hp, max_hp=max_hp,
                      deck=deck, intent=intent, available_cards=available_cards)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\output\simple_grid_choice_agent.txt', 'a', encoding="utf-8") as file:
            file.write('--------------round start-------------------------\n')
            # file.write("System for small llm:\n"+ self.small_llm_sys+ '\n' )
            file.write("System:\n" + self.grid_choice_agent_sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write('--------------round end-------------------------\n')
        return result



