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
        self.ori_suggestion = None
        self.end_turn_cnt = None
        self.battle_agent_sys_prompt = None
        self.router2_cnt = 0
        self.humanM = None
        self.explanation = None
        self.target_index = None
        self.card_Index = None
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
            description="The index of the card you choose from Hand Pile; if you don't choose a card, just return -1",
            type="Int"
        )
        target_index_schema = ResponseSchema(
            name="targetIndex",
            description="The index of your card's target in enemy list; if your card's attribute 'is_card_has_target' is False, just return -1.",
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
        graph_builder.add_node("Suggest", self.suggestionAdder)

        graph_builder.add_edge(START, "Suggest")
        graph_builder.add_edge("Suggest", "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", "Check"])
        graph_builder.add_edge("Tool", "LLM")
        graph_builder.add_conditional_edges("Check", self.router2, ["LLM", END])

        self.graph = graph_builder.compile()

        self.last_two_rounds = deque(maxlen=2)

    def llm_1(self, state: State):
        basic_game_rules = """At the beginning of one turn, you will be given energy in the value of MAX_ENERGY,and
                                    you will draw cards from **Draw Pile**.If your current energy is greater than the cost of 
                                    the card, then you can play this card (NOTICE that even if your energy is 0, 
                                    you can still play a 0 cost cards).One turn will be separated into several operations.
                                    On each operation,you choose one card from **Hand pile** to play(you can ONLY play card 
                                    from **Hand Pile**!!!).In the end of one turn, the remaining energy and block will be cleared.
                                    """

        general_guidance = """- before you choose a card, please figure out your combat strategy first based on the enemy you encounter.
                            - if you don't have the relic 'Runic Dome', and your enemy's intent is UNKNOWN, it means the enemy doesn't
                              attack this turn. Then you should prioritize attacking or enhance yourself rather than using 
                              defensive cards.
                            - Focus on maximizing damage if the enemy’s HP is low, or if you have a strong offensive option available,
                              or you think you can terminate the enemy in this turn.
                            - If enemy has high block now, you can prefer not to attack it in this turn, because in next turn it's 
                              block will be removed.
                            - Consider using defensive cards(cards that gain blocks or weaken the enemy) when enemy has an attack 
                              intention and the damage is bigger than your block. when facing multiple enemies, prioritize 
                              weaken the enemy with biggest damage.
                            - prioritize Power card,as it can benefit you in all turns after.
                            - when facing multiple enemies, AOE cards should be prioritized.
                            - When you are about to attack or defend, you should prioritize non-basic cards(cards that are not 
                              "Defend" or "Strike")
                            - For cards of the same type, prioritize those with best overall effects (evaluated based on its 
                              value, additional effects, etc.)
                            - card with 0 cost can be chosen whatever your available energy is, so please prioritize it.
                            - when you have 0 energy,don't easily end your turn, you can still play 0 cost card.
                            - spend your energy at most.Don't easily leave unused energy in one turn.
                            - Take into account any status effects that may alter the effectiveness of your cards 
                              (e.g., *Vulnerable*, *Frailty*, etc.).
                            - If you cannot play any card or you feel that playing a card will do much harm than good, 
                              choose to end the turn.
                            - If there are any synergy effects (e.g., combo cards or cards that strengthen with specific conditions)
                              , consider playing those first.
                            - Evaluate potential future turns: if you need to conserve status effects that 
                              can be applied in future turns, account for that as well.
                            - exhausting the status card is good for you, do it when you can."""

        outputFormat = self.battle_output_parser.get_format_instructions()
        system_msg = f"""
                            You are an AI designed to play *Slay the Spire* as the role {self.role} and make optimal 
                            card choices during combat. Please read the Basic Game Rules below first.
                            Basic Game Rules:
                            {basic_game_rules}
                            
                            On each operation, you need to choose one card to play or decide to 
                            end the turn. you will be given info of previous two operations and context of this operation.
                            Based on the context, please make your choice by combining user guidance
                            ,game rules and the info of previous two operations.

                            Context format:
                               **Floor**: 'floor' (current floor in game)
                               **Turn Number**: 'turn_number' (current turn in this combat)
                               **Current HP**: 'current_hp' / 'max_hp'
                               **Block**: 'block' (current block you have),
                               **Energy Available**: 'energy' (how much energy is available for playing cards),
                               **Relics**: [ Relic ],(the relics you have)
                               **Enemy Lists**: [ Enemy ]  (a list of enemy,each Enemy is in format: 
                                "enermy_name( enermy_hp,enemy_intent,enemy_block,[enemy_status])"  )
                               **Hand pile**: [ Card ] (list of cards available in the player’s hand, each Card is in
                                  format: "card_name( card_cost,is_card_has_target )" )
                               **Draw Pile**: [ Card ] (list of cards in draw pile)
                               **Discard Pile**: [ Card ](list of cards in discard pile)
                               **Player Status**: [ player_status ] (list of player status)
                            
                            Previous two operations Info Format:
                            [ {{ turn: int, operation: str }}, ...  ]
                            
                            General_guidance:
                            {general_guidance}
                            
                            Response format:
                            {outputFormat}
                            
                            Attention:
                            - Before giving your response,please check the chosen card's attribute 'is_card_has_target', 
                              if it's True,then you need to appoint a target for the card, which means 'targetIndex' 
                              in your response shouldn't be -1.
                            - your response shouldn't contain annotation '//'
                            """
        self.battle_agent_sys_prompt = system_msg
        messages = [{"role": "system", "content": system_msg}] + state["messages"]
        return {
            **state,  # 保留原 state 的所有属性
            "messages": [self.llm.invoke(messages)]
        }

    def suggestionAdder(self, state: State):
        basic_game_rules = """At the beginning of one turn, you will be given energy in the value of MAX_ENERGY,and
                            you will draw cards from **Draw Pile**.If your current energy is greater than the cost of 
                            the card, then you can play this card (NOTICE that even if your energy is 0, 
                            you can still play a 0 cost cards).One turn will be separated into several operations.
                            On each operation,you choose one card from **Hand pile** to play(you can ONLY play card 
                            from **Hand Pile**!!!).In the end of one turn, the remaining energy and block will be cleared.
                            """

        system_msg = f"""
                    You are an AI designed to play *Slay the Spire* as the role {self.role} and analyse the current combat
                    situation to generate guidance about making card choices. Please read the Basic Game Rules below first. 
                    Basic Game Rules:
                    {basic_game_rules}
                    
                    you will be given info of previous two operations and context of this operation.
                    you will also be given a Ready made guidance.
                    Based on them,your job is to analyze various aspects of combat situation(including Enemy,cards,etc)
                    , and completion the Ready made guidance.
                    
                    Context Format:
                    **Floor**: 'floor' (current floor in game)
                    **Turn Number**: 'turn_number' (current round in the combat)
                    **Current HP**: 'current_hp' / 'max_hp'
                    **Block**: 'block' (current block you have),
                    **Energy Available**: 'energy' (how much energy is available for playing cards),
                    **Relics**: [ Relic ],(the relics you have)
                    **Enemy Lists**: [ Enemy ]  (a list of enemy,each Enemy is in format: 
                                "enermy_name( enermy_hp,enemy_intent,enemy_block,[enemy_status])"  )
                    **Hand pile**: [ Card ] (list of cards available in the player’s hand, each Card is in
                                format: "card_name( card_cost,is_card_has_target )" )
                    **Draw Pile**: [ Card ] (list of cards in draw pile)
                    **Discard Pile**: [ Card ](list of cards in discard pile)
                    **Player Status**: [ player_status ] (list of player status)
                    
                    Previous two operations Info Format:
                    [ {{ turn: int, operation: str }}, ...  ]
                    
                    
                    Response:
                    completion the Ready made guidance. you should add 2 parts into the guidance:
                    1. Introduction of the enemy and strategy to deal with it.
                    2. analysis of current state of User
                    your response should follow the format: 
                    **Guidance**:
                            xxxxxxxxxxxxx
                    Refine your response. **limit your response to 100 words!!**.
                    """

        suggestion_content = '**Guidance**:'

        # 人工添加建议：
        monsters = state["monsters"]
        hand = state["hand"]
        current_hp = state["current_hp"]

        # suggestion_content += "\n spend your energy at most.Don't easily leave unused energy in one turn."
        # suggestion_content += ("\nFor cards of the same type, prioritize cards with best overall effects "
        #                        "(evaluated based on its"
        #                        "value, additional effects, etc.)")
        suggestion_content += ("When you are about to attack or defend, you should prioritize"
                               " non-basic cards(cards that are not"
                              "'Defend' or 'Strike'")
        # suggestion_content += ("When facing multiple enemies which are leader and minions, prioritize"
        #                        "dealing with the leader.")

        no_attack_flag = 1
        for monster in monsters:
            if monster.intent == Intent.ATTACK or monster.intent == Intent.ATTACK_BUFF or monster.intent == Intent.ATTACK_DEBUFF or monster.intent == Intent.ATTACK_DEFEND:
                no_attack_flag = 0
                break
        if no_attack_flag == 1:
            suggestion_content += ("\nenemies are not in attacking intention this round,"
                                   "you should prioritize dealing damage or buffering yourself.")
        zero_cost_card_flag = 0
        for card in hand:
            if card.cost == 0:
                zero_cost_card_flag = 1
        if zero_cost_card_flag == 1:
            suggestion_content += ("\nYou have 0 cost cards in your Hand Pile,"
                                   "you could consider prioritizing them as they cost no energy.")
        low_hp_flag = 0
        for monster in monsters:
            if monster.current_hp < 10:
                low_hp_flag = 1
                break
        if low_hp_flag:
            suggestion_content += ("\nEnemy is in low hp,check the maximum damage you can deal to see"
                                   "if you can eliminate it.")
        if len(monsters) > 1:
            suggestion_content += ("\nYou are facing multiply enemies,you should prioritize"
                                   "AOE card which can affect them all.")

        self.ori_suggestion = suggestion_content
        with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'a') as file:
            file.write("\nOriginal Suggestion is:\n" + self.ori_suggestion)

        messages = [{"role": "system", "content": system_msg}] + [
            HumanMessage(content=self.humanM + '\n' + suggestion_content)]
        response = self.small_llm.invoke(messages)
        suggestion_content = response.content

        return {
            **state,  # 保留原 state 的所有属性
            "messages": [AIMessage(content="can you help me analyse the condition and give me some guidance?")] + [
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

        if 0 <= self.card_Index < len(hand_cards):
            card_to_play1 = hand_cards[self.card_Index]
            if not card_to_play1.is_playable:
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
        if self.target_index != -1 and 0 <= self.target_index < len(available_monsters):
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
               config=None):
        start_time = time.time()  # 记录开始时间

        last_two_rounds_info = '['
        for item in list(self.last_two_rounds):
            last_two_rounds_info += (str(item) + '\n')
        last_two_rounds_info += ']'

        self.router2_cnt = 0
        self.end_turn_cnt = 0

        template_string = """ 
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
            last_two_rounds_info=last_two_rounds_info
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
        if self.target_index != -1 and 0 <= self.target_index < len(available_monsters):
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
            file.write("System:\n" + self.battle_agent_sys_prompt + '\n')
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
