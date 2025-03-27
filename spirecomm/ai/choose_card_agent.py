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

from spirecomm.spire.card import Card, CardRarity
from spirecomm.spire.character import Monster, Intent
from spirecomm.spire.relic import Relic

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"


def get_lists_str_with_r(lists):
    str = "[ "
    for item in lists:
        str += item.name
        rarity = ''
        if item.rarity == CardRarity.BASIC:
            rarity = 'BASIC'
        if item.rarity == CardRarity.COMMON:
            rarity = 'COMMON'
        if item.rarity == CardRarity.UNCOMMON:
            rarity = 'UNCOMMON'
        if item.rarity == CardRarity.RARE:
            rarity = 'RARE'
        if item.rarity == CardRarity.SPECIAL:
            rarity = 'SPECIAL'
        if item.rarity == CardRarity.CURSE:
            rarity = 'CURSE'

        str += '(' + rarity +')'
        if item != lists[-1]:
            str += ", "
    str = str + " ]"
    return str

def get_lists_str(lists):
    str = "[ "
    for item in lists:
        str += item.name
        if item != lists[-1]:
            str += ", "
    str = str + " ]"
    return str



class State(TypedDict):
    messages: Annotated[list, add_messages]
    floor: int
    current_hp: int
    max_hp: int
    deck: list
    reward_cards:list
    relic_bowl:bool


class ChooseCardAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):

        self.strategy = None
        self.current_deck = None
        self.router2_cnt = 0
        self.explanation = None
        self.card_name = None
        self.choose_card_agent_sys_prompt = None
        self.humanM = None

        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        card_name_schema = ResponseSchema(
            name="cardName",
            description="The name of the card you choose.**should not contain '()'**."
                        "if you decide to choose no card,return ''.if you choose to use relic 'Bowl',return 'Bowl'"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you choose the card."
        )
        response_schemas = [
            card_name_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.choose_card_output_parser = output_parser


        tools = load_tools(["wikipedia"],llm=self.small_llm)
        tool_node1 = ToolNode(tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("Tool", tool_node1)
        graph_builder.add_node("LLM", self.llm_1)
        graph_builder.add_node("Check", self.outputChecker)
        graph_builder.add_node("Suggest", self.suggestionAdder)
        graph_builder.add_node("Strategy",self.strategyGenerator)

        graph_builder.add_edge(START, "Suggest")
        graph_builder.add_edge("Suggest", "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", "Check"])
        graph_builder.add_edge("Tool", "LLM")
        graph_builder.add_conditional_edges("Check", self.router2, ["LLM", "Strategy"])
        graph_builder.add_edge("Strategy",END)

        self.graph = graph_builder.compile()


    def llm_1(self, state: State):

        outputFormat = self.choose_card_output_parser.get_format_instructions()
        system_prompt = f"""
                        You are an expert at playing Slay the Spire, and now you need to play Slay the Spire 
                        as the role {self.role}.now you need to assist in choosing card rewards.
                         Your goal is to maximize the player's chances of success by selecting the most beneficial cards 
                         based on the current context. Before choosing, please invoke the tool to search the content of 
                         cards on wikipedia.

                        ### Context:
                        - **Floor**: [Current floor]
                        - **Current Deck:** [List of cards currently in the deck]
                        - **Player's Health:** [Current health points]
                        - **Available Cards:** [List of cards available for selection]
                        - **Relic Bowl:** [whether you have the relic 'Bowl']

                        ### Considerations:
                        1.**Deck Size Management**: Maintain a streamlined deck. A larger deck can dilute card effectiveness and 
                          make it harder to draw key cards. If the available cards do not significantly improve the deck  , 
                          consider skipping the card selection. Don't grab too many cards of the same type.
                        2. **Card Rarity:** Evaluate the rarity of the available cards. Prioritize higher rarity cards for their
                            unique abilities and potential impact on the gameplay.
                        3. **Upgraded:** Prioritize upgraded cards( 'card_name+' ) for their upgraded abilities.
                        3. **Synergy with Archetypes:** Assess how well each card fits into a specific archetype or deck strategy.
                          Look for cards that complement and enhance the current build, creating powerful synergies.
                        4. **Health Management:** Prioritize cards that can help regain health or mitigate damage .
                        5. **Scaling:** Consider cards that scale well into the later acts of the game.


                        ### choice：
                        1.choose one card from **Available Cards**
                        2.skip the card selection
                        3.use the relic "Bowl" to improve max-hp 
                        
                        ### attention
                        Do NOT always choose to add one card.Adding too many low-level cards will do much more harm to
                        your deck than good.Skipping the card could be a greater choice in some cases.If you have relic
                        "Bowl", you can weigh the trade-off between adding a card and increasing max HP.

                        ### output format:
                        {outputFormat}
                        """
        self.choose_card_agent_sys_prompt = system_prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        return {
            **state,  # 保留原 state 的所有属性
            "messages": [self.llm.invoke(messages)]
        }

    def strategyGenerator(self,state:State):
        system_msg = """
You are an advanced *Slay the Spire* deck strategy analyst. Your task is to analyze the current deck and provide a comprehensive 
**game plan** that describes the overall strategy for playing this deck effectively. The game plan should include 
a high-level approach to playing the deck and detailed tactical instructions on how to execute the strategy.

### **1. Output Format**:
Your response should be structured in the following format:
**deck Analysis**: 
"A high-level description of how this deck should be played, including its core strategy, strengths, weaknesses,
 and general approach to fights.",
 
 refine your response and limit it up to 100 words.

"""

        human_msg = f""" Context:
                        - **Current Deck:** {get_lists_str(self.current_deck)}

                        now give your response. """

        messages = [{"role": "system", "content": system_msg}] + [
            HumanMessage(content=human_msg)]
        response = self.small_llm.invoke(messages)
        self.strategy = response.content
        return {**state,
                "messages":[]}

    def suggestionAdder(self, state: State):

        system_msg = """
You are an advanced *Slay the Spire* deck-building AI assistant. Your task is to analyze the given deck 
and provide guidance on selecting post-battle card rewards. Your analysis should determine whether 
adding a card is beneficial and which option best aligns with the deck’s current strategy.

Context Format:
- **Current Deck:** [List of cards currently in the deck] - 
- **Floor**: [Current floor]
- **Player's Health:** [Current health points]
- **Available Cards:** [List of cards available for selection] 

### Considerations:
1.**Deck Size Management**: Maintain a streamlined deck. A larger deck can dilute card effectiveness and 
make it harder to draw key cards. If the available cards do not significantly improve the deck  , 
consider skipping the card selection. Don't grab too many cards of the same type.
2. **Card Rarity:** Evaluate the rarity of the available cards. Prioritize higher rarity cards for their
unique abilities and potential impact on the gameplay.
3. **Upgraded:** Prioritize upgraded cards( 'card_name+' ) for their upgraded abilities.
4. **Synergy with Archetypes:** Assess how well each card fits into a specific archetype or deck strategy.
Look for cards that complement and enhance the current build, creating powerful synergies.
5. **Health Management:** Prioritize cards that can help regain health or mitigate damage .
6. **Scaling:** Consider cards that scale well into the later acts of the game.                        

###  Output Format**
Your response should be structured in JSON format as follows:
```json
{
  "deck_analysis": {
    "current_strategy": "Briefly describe the deck's current core strategy.",
    "missing_elements": ["List what the deck lacks to improve its strategy."]
  },
  "card_evaluation": [
    {
      "card_name": "Card A",
      "justification": "Explain why this card is good or bad for the deck.",
      "synergies": ["List cards in the deck that this card synergizes with, if any."],
      "anti_synergies": ["List cards in the deck that this card conflicts with, if any."],

    },
    .......
  ],
  "skip_evaluation":{
    "skip_value": "low | medium | high",
    "skip_justification": "Explain why skipping is or isn't valuable."
  }
}

  
"""

        human_msg = f""" Context:
                - **Current Deck:** {get_lists_str(state["deck"])}
                - **Floor**: {state["floor"]}
                - **Player's Health:** {state["current_hp"]}
                - **Available Cards:** {get_lists_str_with_r(state["reward_cards"])}
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
            self.card_name = jsonfile.get('cardName')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate it!"}]
            }
        
        self.current_deck = state["deck"]
        if self.card_name=="" or self.card_name=="Bowl":
            return {
                **state,
                "messages": [AIMessage(content="output check pass!!")]
            }
        else:
            reward_cards = state["reward_cards"]
            card_to_choose = next((card for card in reward_cards if card.name == self.card_name), None)
            if card_to_choose is None:
                if self.card_name.count('(') > 0:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "Your provided card_name should not contain '()',"
                                                                 " please regenerate it!"}]
                    }
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "Your provided card_name is not in reward_cards,"
                                                             " please regenerate it!"}]
                }
            else:
                self.current_deck.append(card_to_choose)

        # check pass!
        return {
            **state,
            "messages": [AIMessage(content="output check pass!!")]
        }

    def router2(self, state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if type(last_message) == AIMessage:
            return "Strategy"
        else:
            self.router2_cnt += 1
            with open(r'C:\Users\32685\Desktop\spirecomm\choose_card_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 2:
                self.card_name = ''
                self.explanation = 'router2 reach recursion limit! use algorithm to choose card!'
                return "Strategy"
            return "LLM"

    def invoke(self, floor: int,
               current_hp: int,
               max_hp: int,
               deck: list,
               reward_cards: list,
               relic_bowl: bool,
               config=None):
        start_time = time.time()  # 记录开始时间
        self.router2_cnt = 0

        template_string = """ 
                Context
                - **Floor**:{floor}
                - **Current Deck:** {deck}
                - **Player's Health:** {hp}
                - **Available Cards:** {reward_cards}
                - **Relic Bowl:** {relic_bowl}
                
                now give your response.
                """
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            floor=floor,
            hp=f"{current_hp}/{max_hp}",
            deck = get_lists_str(deck),
            reward_cards=get_lists_str_with_r(reward_cards),
            relic_bowl=relic_bowl
        )
        self.humanM = messages[0].content
        state = State(messages=messages, floor=floor, current_hp=current_hp, max_hp=max_hp,
                      deck=deck,reward_cards=reward_cards,relic_bowl=relic_bowl)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\choose_card_agent.txt', 'a',encoding="utf-8") as file:
            file.write('--------------round start-------------------------\n')
            file.write("System:\n" + self.choose_card_agent_sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write("strategy:\n"+self.strategy+'\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write('--------------round end-------------------------\n')
        return result



