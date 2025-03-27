import json
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

from spirecomm.spire.card import Card
from spirecomm.spire.character import Monster, Intent
from spirecomm.spire.relic import Relic

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"


def get_lists_str(lists):
    str = "[ "
    for item in lists:
        str += (item.__str__())
        if item != lists[-1]:
            str += ", "
    str = str + " ]"
    return str


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


class BattleAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):
        self.deck_analysis = ''
        self.ori_suggestion = None
        self.end_turn_cnt = None
        self.battle_agent_sys_prompt = None
        self.router2_cnt = 0
        self.humanM = None
        self.explanation = None
        self.target_index = None
        self.card_Index = -1
        self.is_to_end_turn = None
        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        is_to_end_turn_schema = ResponseSchema(
            name="isToEndTurn",
            description="return 'No' if you have chosen one card to play, return 'Yes' if you decide to end the turn"
        )
        card_index_schema = ResponseSchema(
            name="cardIndex",
            description="The index of the card you choose from Hand Pile(Start with 0); if you don't choose a card, just return -1",
            type="Int"
        )
        target_index_schema = ResponseSchema(
            name="targetIndex",
            description="The index of your card's target in enemy list(Start with 0); if your card's attribute 'is_card_has_target' is False, just return -1.",
            type="Int"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you take the action."
        )
        # 将所有 schema 添加到列表中
        response_schemas = [
            is_to_end_turn_schema,
            card_index_schema,
            target_index_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        self.battle_output_parser = output_parser

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

        self.last_two_rounds = deque(maxlen=2)

    def llm_1(self, state: State):

        outputFormat = self.battle_output_parser.get_format_instructions()
        system_msg_2 = f"""You are an AI designed to play *Slay the Spire* as {self.role} and make optimal card choices during combat. 
### Basic Game Rules:
At the beginning of a turn, you will be given MAX_ENERGY and draw cards from the Draw Pile. 
You can only play cards from your Hand Pile, and each card costs a certain amount of energy. 
A turn consists of multiple actions. 
On each action, your job is to choose **one** card to play (if energy allows) or **end the turn**.

### deck Analysis:
To improve decision-making, you are provided with Analysis of your current deck.

### Context:
info of the current action
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

### Previous Two actions:
To improve decision-making, you are provided with the previous two actions:
[ 
  {{ turn: int, operation: str }}, // previous two
  {{ turn: int, operation: str }}, // previous one 
]

### Notice:
things you should be aware of in the combat.

### Response Format:
{outputFormat}
"""

        self.battle_agent_sys_prompt = system_msg_2
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
            self.is_to_end_turn = jsonfile.get('isToEndTurn')
            # card_name = jsonfile.get('cardName')
            self.card_Index = jsonfile.get('cardIndex')
            self.target_index = jsonfile.get('targetIndex')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate it!"}]
            }

        available_monsters = [monster for monster in state["monsters"] if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        playable_cards = [card for card in state["hand"] if card.is_playable]
        hand_cards = state["hand"]
        zero_cost_card = 0
        for card in playable_cards:
            if card.cost == 0:
                zero_cost_card = 1

        if self.is_to_end_turn == 'Yes':
            self.end_turn_cnt += 1
            if self.end_turn_cnt == 1:
                if state["energy"] == 0 and zero_cost_card:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "There are 0 cost cards in your Hand Pile,"
                                                                 "you can play them even if your energy is 0."
                                                                 "are you sure to end the turn?"
                                                                 "please regenerate the answer."}]
                    }
                if state["energy"] > 0 and len(playable_cards) > 0:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "you have unused energy and there are still"
                                                                 "playable cards,are you sure to end the turn?"
                                                                 "please regenerate the answer."}]
                    }
            return {
                **state,
                "messages": [AIMessage(content="output check pass!!")]
            }

        if self.card_Index is not None and 0 <= self.card_Index < len(hand_cards):
            card_to_play1 = hand_cards[self.card_Index]
            if not card_to_play1.is_playable:
                if card_to_play1.cost > state["energy"]:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "Your chosen card's cost is greater than your energy!,"
                                                                 " please regenerate it!"}]
                    }

                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": "Your chosen card is not playable,"
                                                             " please regenerate it!"}]
                }
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your card_Index is out of range,"
                                                         " please regenerate it!"}]
            }
        target1 = None
        if self.target_index is not None and self.target_index != -1 and 0 <= self.target_index < len(available_monsters):
            target1 = available_monsters[self.target_index]

        if card_to_play1 is not None:
            if target1 is None:
                if card_to_play1.has_target:
                    if self.target_index == -1:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": "Your chosen card must have a target,"
                                                                     " please regenerate it!"}]
                        }
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "Your target_index is out of range,"
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
            with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'a') as file:
                file.write('cnt is:' + str(self.router2_cnt) + '\n')
            if self.router2_cnt >= 2:
                self.is_to_end_turn = 'NO'
                self.card_Index = -1
                self.target_index = -1
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
               deck_analysis: str,
               config=None):
        start_time = time.time()  # 记录开始时间

        last_two_rounds_info = '['
        for item in list(self.last_two_rounds):
            last_two_rounds_info += (str(item) + '\n')
        last_two_rounds_info += ']'

        self.router2_cnt = 0
        self.end_turn_cnt = 0
        self.deck_analysis = deck_analysis

        # 人工添加建议：
        suggestion_content = ''
        suggestion_content += '**Notice**:'
        no_attack_flag = 1
        total_damage = 0
        low_hp_flag = 0
        low_hp_m_list = []

        for monster in monsters:
            if (monster.intent == Intent.ATTACK or monster.intent == Intent.ATTACK_BUFF or
                    monster.intent == Intent.ATTACK_DEBUFF or monster.intent == Intent.ATTACK_DEFEND):
                no_attack_flag = 0

            if monster.current_hp < 10:
                low_hp_flag = 1
                low_hp_m_list.append(monster)

            total_damage += monster.move_hits * monster.move_adjusted_damage

            if monster.monster_id == "GremlinNob":
                suggestion_content += ("\nYou are facing Elite enemy GremlinNob,With the exception of the first turn, "
                                       "where it has yet to apply  Enrage, playing Skills will make the Gremlin Nob "
                                       "much more threatening. Since most  Block-granting cards are also Skills, "
                                       "it can be worth more to not play them and take the damage instead. "
                                       "Before using a Skill to mitigate damage, "
                                       "consider how much longer the fight might take.")

        if no_attack_flag == 1:
            suggestion_content += ("\nenemies are not in attacking intention this round,"
                                   "you should prioritize dealing damage or buffing yourself.")
        if low_hp_flag:
            suggestion_content += ("\nEnemy is in low hp,check the maximum damage you can deal to see"
                                   "if you can eliminate it.")
        if len(monsters) > 1:
            suggestion_content += ("\nYou are facing multiply enemies,you should prioritize"
                                   "AOE card which can affect them all.")

        if total_damage - block >= 10:
            suggestion_content += f"\nYou are facing huge incoming damage, which will make you lose {total_damage - block} hp"

        zero_cost_card_flag = 0
        for card in hand:
            if card.cost == 0:
                zero_cost_card_flag = 1
        if zero_cost_card_flag == 1:
            suggestion_content += ("\nYou have 0 cost cards in your Hand Pile,"
                                   "you could consider prioritizing them as they cost no energy.")

        template_string = """       
{deck_analysis}        

context:
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
        **Orbs**(if you are DEFECT): {orbs}

Previous two operations Info:
        {last_two_rounds_info}

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
            last_two_rounds_info=last_two_rounds_info,
            notice=suggestion_content,
            deck_analysis=deck_analysis
        )
        self.humanM = messages[0].content
        state = State(messages=messages, turn=turn, current_hp=current_hp, max_hp=max_hp,
                      block=block, energy=energy, relics=relics, hand=hand,
                      monsters=monsters, drawPile=drawPile, discardPile=discardPile,
                      powers=powers, orbs=orbs)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 添加round信息到队列
        available_monsters = [monster for monster in monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        playable_cards = [card for card in hand if card.is_playable]
        card_to_play = None
        if 0 <= self.card_Index < len(playable_cards):
            card_to_play = playable_cards[self.card_Index]
        target1 = None
        if self.target_index is not None and self.target_index != -1 and 0 <= self.target_index < len(available_monsters):
            target1 = available_monsters[self.target_index]

        operation = ""
        if self.is_to_end_turn == 'Yes':
            operation += "END turn"
        elif card_to_play is not None:
            operation += f"choose card '{card_to_play.name}'"
            if card_to_play.has_target and target1 is not None:
                operation += f" towards '{target1.name}(target_index={self.target_index})'"
        round_info = f"{{ turn:{turn},operation:{operation} }}"
        self.last_two_rounds.append(round_info)

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'a') as file:
            file.write('--------------round start-------------------------\n')
            # file.write("System:\n" + self.battle_agent_sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write('--------------round end-------------------------\n')

        return result


if __name__ == "__main__":
    agent = BattleAgent()
    # responses = agent.invoke(
    #     floor=1,
    #     turn=1,
    #     current_hp=75,
    #     max_hp=75,
    #     block=0,
    #     energy=3,
    #     relics=[Relic(name="Cracked Core",relic_id=1)],
    #     hand=[ Card(name='Strike',cost=1, has_target=True), Card(name='Zap',cost=1,has_target=False),Card(name='Strike',cost=1, has_target=True), Card(name='Dualcast',cost=1, has_target=False), Card(name='Strike',cost=1, has_target=True) ],
    #     monsters=[ Monster(name='JawWorm',intent='',max_hp=42,current_hp=42,block=0,powers=[]) ],
    #     drawPile=[ Card(name='Strike',cost=1, has_target=True), Card(name='Defend',cost=1, has_target=False), Card(name='Defend',cost=1, has_target=False), Card(name='Defend',cost=1, has_target=False) ],
    #     discardPile=[],
    #     powers=[],
    #     orbs=[  ],
    # )
    # for response in responses["messages"]:
    #     print(type(response).__name__+":\n "+response.content.__str__())
    #     # print("\n\n")
    #
    # print("self.target_index:",str(agent.target_index))
    # print("self.is_to_end_turn:"+str(agent.is_to_end_turn))
    # print("self.target_index:"+str(agent.target_index))
    # print("self.explanation:"+str(agent.explanation))
