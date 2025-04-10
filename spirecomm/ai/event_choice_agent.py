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


def get_lists_str_for_options(lists):
    str = "[\n"
    for item in lists:
        str += '"'+item.text+'"'
        if item != lists[-1]:
            str += ",\n"
    str = str + "\n]"
    return str


def get_lists_str(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret


class State(TypedDict):
    messages: Annotated[list, add_messages]
    floor: int
    current_hp: int
    max_hp: int
    deck: list
    relics:list
    event_name:str
    event_text:str
    event_options:list


class EventChoiceAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)):


        self.router2_cnt = 0
        self.explanation = None
        self.option_index = None
        self.sys_prompt = None
        self.humanM = None

        self.role = role
        self.llm = llm


        option_index_schema = ResponseSchema(
            name="optionIndex",
            description="The index of the event option you choose.(starts with 0)",
            type="Int"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you make this choice."
        )
        response_schemas = [
            option_index_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.output_parser = output_parser

        tools = load_tools(["wikipedia"], llm=self.llm)
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
You are an expert at playing Slay the Spire, and now you play as the role {self.role}.
now you need to assist in making event choice.Your goal is to maximize the player's chances 
of success by selecting the most beneficial event choice based on the current context. 
                         
### Event INFO:
- **Event Name**: 'name of event'
- **Event text**: 'text of event'
- **Event Options**: [ Option ] 
                        
### Context:
- **Floor**: 'floor'
- **Current Deck:** [ Card ]
- **Player's Health:** 'current_hp' / 'max_hp'
- **Relics**: [Relic],

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
            self.option_index = jsonfile.get('optionIndex')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate it!"}]
            }

        options = state["event_options"]
        chosen_option = None

        if isinstance(self.option_index,int) and 0 <= self.option_index < len(options):
            chosen_option = options[self.option_index]
            if chosen_option.disabled:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "Your event_option is disabled,"
                                                             " please regenerate it!"}]
                }
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your provided option_index is out of range,"
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
            with open(r'C:\Users\32685\Desktop\spirecomm\choose_card_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 2:
                self.option_index = 0
                self.explanation = 'router2 reach recursion limit! use algorithm to choose card!'
                return END
            return "LLM"

    def invoke(self, floor: int,
               current_hp: int,
               max_hp: int,
               deck: list,
               relics:list,
               event_name: str,
               event_text: str,
               event_options: list,
               config=None):
        start_time = time.time()  # 记录开始时间
        self.router2_cnt = 0

        template_string = """ 
Event:
- **Event Name**: {event_name}
- **Event text**: {event_text}
- **Event Options**: {event_options}
                
Context
- **Floor**:{floor}
- **Current Deck:** {deck}
- **Player's Health:** {hp}
- **Relics:** {relics}

now give your response.
"""
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            floor=floor,
            hp=f"{current_hp}/{max_hp}",
            deck=get_lists_str(deck),
            relics=get_lists_str(relics),
            event_name = event_name,
            event_text = event_text,
            event_options = get_lists_str_for_options(event_options)
        )
        self.humanM = messages[0].content
        state = State(messages=messages, floor=floor, current_hp=current_hp, max_hp=max_hp,
                      deck=deck, relics=relics, event_name=event_name,event_text=event_text,event_options=event_options)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\event_agent.txt', 'a', encoding="utf-8") as file:
            file.write('--------------round start-------------------------\n')
            # file.write("System:\n" + self.sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write('--------------round end-------------------------\n')
        return result



