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
    for index,item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret

def get_lists_str_for_m(lists):
    ret = "[ \n\t"
    for index,item in enumerate(lists):
        ret += (item.__str__())
        if index != len(lists)-1:
            ret += ",\n\t"
    ret += " \n\t]"
    return ret

def get_lists_str_with_only_name(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        ret += item.name
        if index != len(lists)-1:
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
    potion:list


class BattleAgent:
    def __init__(self, role="DEFECT", llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
                 small_llm=ChatOllama(model="mistral:7b", temperature=0)):
        self.cnt = 0
        self.total_invoke_time = 0
        self.error_invoke_cnt = 0
        self.total_invoke_cnt = 0
        self.potion_index = None
        self.action = None
        self.deck_analysis = ''
        self.ori_suggestion = None
        self.end_turn_cnt = None
        self.battle_agent_sys_prompt = None
        self.router2_cnt = 0
        self.humanM = None
        self.explanation = None
        self.target_index = None
        self.card_Index = -1

        self.role = role
        self.llm = llm
        self.small_llm = small_llm

        action_schema = ResponseSchema(
            name="action",
            description="return 'card' if you choose one card to play, return 'end' if you decide to end the turn"
                        "return 'potion' if you choose one potion to use."
        )
        card_index_schema = ResponseSchema(
            name="cardIndex",
            description="The index of the card you choose from Hand Pile(Start with 0); if you don't choose a card, just return -1",
            type="Int"
        )
        potion_index_schema = ResponseSchema(
            name="potionIndex",
            description="The index of the potion you choose(Start with 0); if you don't choose a potion, just return -1",
            type="Int"
        )
        target_index_schema = ResponseSchema(
            name="targetIndex",
            description="The index of your target in enemy list(Start with 0); if your card's attribute 'is_card_has_target' is False, just return -1."
                        "If your potion's attribute 'is_potion_has_target' is False, just return -1.",
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

        self.previous_rounds_info = deque(maxlen=5)

    def llm_1(self, state: State):

        outputFormat = self.battle_output_parser.get_format_instructions()
        system_msg_2 = f"""You are an AI designed to play *Slay the Spire* as {self.role} and make optimal card choices during combat. 
### Basic Game Rules:
At the beginning of a turn, you will be given MAX_ENERGY and draw cards from the Draw Pile. 
You can only play cards from your Hand Pile, and each card costs a certain amount of energy. 
A turn consists of multiple actions. 
On each action, your job is to choose **one** card to play (if energy allows) or **end the turn** or 
 choose **one** potion to use (if have).

### deck Analysis:
To improve decision-making, you are provided with Analysis of your current deck.

### Combat situation:
info of the current combat.
- **Floor**: 'floor'
- **Turn Number**: 'turn_number'
- **Current HP**: 'current_hp' / 'max_hp'
- **Block**: 'block'
- **Energy Available**: 'energy'
- **Relics**: [ Relic ]
- **Enemy List**: [ Enemy ]  Enemy format: "enermy_name( enermy_hp,enemy_intent,enemy_block,[enemy_status])"
- **Draw Pile**: [ Card ] 
- **Discard Pile**: [ Card ]
- **Player Status**: [ player_status ]
- **Potion**: [ Potion ] Potion format: "potion_name(is_potion_has_target)"

### Previous actions in this turn:
To improve decision-making, you are provided with the previous actions in this turn:
[ 
  {{ turn: int, operation: str }}, //first action in this turn
  .......
  {{ turn: int, operation: str }}, // last action in this turn
]

### Notice:
things you should be aware of in the combat.

### Hand Pile:
list of cards to choose from
     [ Card ]  Card format: "card_name( card_cost,is_card_has_target,card_type )"

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
            self.action = jsonfile.get('action')
            self.card_Index = jsonfile.get('cardIndex')
            self.potion_index = jsonfile.get('potionIndex')
            self.target_index = jsonfile.get('targetIndex')
            self.explanation = jsonfile.get('explanation')
        except Exception as e:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "Your output does not meet my required JSON format,"
                                                         " please regenerate your answer!"}]
            }

        available_monsters = [monster for monster in state["monsters"] if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        playable_cards = [card for card in state["hand"] if card.is_playable]
        hand_cards = state["hand"]
        potions = state["potion"]
        zero_cost_card = 0
        card_to_play1 = None
        potion_to_use = None
        for card in playable_cards:
            if card.cost == 0:
                zero_cost_card = 1

        #------check action ----------
        if self.action == 'end':
            # end turn
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
                # 相信ai的能力就注释掉这个
                if state["energy"] > 0 and len(playable_cards) > 0:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "you have unused energy and there are still"
                                                                 "playable cards,are you sure to end the turn?You need "
                                                                 "to make full use of your energy!!!If there "
                                                                 "is really no need to play more cards or "
                                                                 "it is harmful to play more cards,insist on"
                                                                 " your choice."
                                                                 "please regenerate the answer."}]
                    }
            return {
                **state,
                "messages": [AIMessage(content="output check pass!!")]
            }

        elif self.action == 'card':
            # play card
            if isinstance(self.card_Index,int) and 0 <= self.card_Index < len(hand_cards):
                card_to_play1 = hand_cards[self.card_Index]
                if not card_to_play1.is_playable:
                    if card_to_play1.cost > state["energy"]:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [
                                {"role": "user", "content": f"Your chosen card's cost({card_to_play1.cost}) is greater than your energy({state["energy"]})!,"
                                                            " please regenerate your answer!"}]
                        }

                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "Your chosen card is not playable,"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": f"Your card_Index is out of range(index ranging from 0 to {len(hand_cards)-1}),"
                                                             " please regenerate your answer!"}]
                }
        elif self.action == 'potion':
            # use potion
            if isinstance(self.potion_index,int) and 0 <= self.potion_index < len(potions):
                potion_to_use = potions[self.potion_index]
                if not potion_to_use.can_use:
                    return {
                        **state,  # 保留原 state 的所有属性
                        "messages": [{"role": "user", "content": "Your chosen potion can not be used,"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    **state,  # 保留原 state 的所有属性
                    "messages": [{"role": "user", "content": f"Your potion_index is out of range(index ranging from 0 to {len(potions)}),"
                                                             " please regenerate your answer!"}]
                }
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": "You should choose your action from ['card','potion','end'],"
                                                         " please regenerate your answer!"}]
            }


        target1 = None
        if isinstance(self.target_index,int) and 0 <= self.target_index < len(
                available_monsters):
            target1 = available_monsters[self.target_index]
        elif self.target_index == -1:
            target1 = None
        else:
            return {
                **state,  # 保留原 state 的所有属性
                "messages": [{"role": "user", "content": f"Your target_index is out of range(index ranging from 0 to {len(available_monsters)-1}),"
                                                         " please regenerate your answer!"}]
            }

        if card_to_play1 is not None:
            if target1 is None:
                if card_to_play1.has_target:
                    if self.target_index == -1:
                        return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": "Your chosen card must have a target(targetIndex can't be -1),"
                                                                     " please regenerate your answer!"}]
                        }

        if potion_to_use is not None:
            if target1 is None:
                if potion_to_use.requires_target:
                    return {
                            **state,  # 保留原 state 的所有属性
                            "messages": [{"role": "user", "content": "Your chosen potion must have a target(targetIndex can't be -1),"
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
                self.action = 'algorithm'
                self.card_Index = -1
                self.potion_index = -1
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
               potion:list,
               room:str,
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
        self.end_turn_cnt = 0
        self.deck_analysis = deck_analysis

        # 人工添加建议：
        suggestion_content = ''
        suggestion_content += 'Notice:'

        no_attack_flag = 1
        total_damage = 0
        low_hp_flag = 0
        low_hp_m_list = []
        Sentry_flag = 0
        Cultist_flag = 0
        Strength = 0
        Artifact_flag = 0


        for power in powers:
            if power.power_name == "Strength":
                Strength = power.amount
                suggestion_content += (f"\nYou have Strength {power.amount}, If you want to Attack,"
                                       f"prioritize cards with multiple hits (e.q. 'Twin Strike',"
                                       f"'Sword Boomerang')")
            if power.power_name == "Artifact":
                Artifact_flag = 1

            if power.power_name == "Confusion":
                suggestion_content += (f"\nYou have status 'Confusion', now the costs of your cards are "
                                       f"randomized on draw, from 0 to 3.")

        for relic in relics:
            if relic.name == "Runic Dome":
                suggestion_content += ("\nYou have the Runic Dome relic, which provides energy each turn "
                                       "but prevents you from seeing enemy intents. This means you won't "
                                       "know whether enemies will attack, defend, or use debuffs.")

        for index, monster in enumerate(monsters):

            for power in monster.powers:
                # poison
                if power.power_name == "Poison":
                    poison = power.amount
                    if poison >= monster.current_hp:
                        suggestion_content += ("\n{}'s poison is greater than its HP,you could focus on other"
                                               "enemies.".format(
                            monster.monster_id + '(target_index=' + str(index) + ')'))

                if power.power_name == "Thorns":
                    suggestion_content += ("\n{} has power 'Thorns'.when attacked, it will deal {} damage back.so "
                                           "be careful using multiple hits attack towards it.".format(
                        monster.monster_id + '(target_index=' + str(index) + ')', power.amount))

            if (monster.intent == Intent.ATTACK or monster.intent == Intent.ATTACK_BUFF or
                    monster.intent == Intent.ATTACK_DEBUFF or monster.intent == Intent.ATTACK_DEFEND or
                    monster.intent == Intent.NONE):
                no_attack_flag = 0

            if (floor <= 16 and monster.current_hp < 10) or (floor > 16 and monster.current_hp < 20):
                low_hp_flag = 1
                low_hp_m_list.append(monster)

            total_damage += monster.move_hits * monster.move_adjusted_damage

            if monster.monster_id == "Cultist" and Cultist_flag == 0:
                suggestion_content += ("\nYou are facing enemy Cultist,who gains Strength after each turn,so it is"
                                       "crucial to eliminating it quickly.")
                Cultist_flag =1

            if monster.monster_id == "AcidSlime_L":
                suggestion_content += ("\nYou are facing enemy AcidSlime_L,When its HP falls below 50%, it splits into "
                                       "two "
                                       "smaller slimes, each with its remaining HP.It is crucial to lower as much "
                                       "HP before it splits."
                                       )

            if monster.monster_id == "GremlinNob":
                suggestion_content += ("\nYou are facing Elite enemy GremlinNob,With the exception of the first turn, "
                                       "where it has yet to apply  Enrage, playing Skills will make the Gremlin Nob "
                                       "much more threatening. Since most  Block-granting cards are also Skills, "
                                       "it can be worth more to not play them and take the damage instead. "
                                       "Before using a Skill to mitigate damage, "
                                       "consider how much longer the fight might take.")

            if monster.monster_id == "Sentry" and len(monsters) == 3 and Sentry_flag == 0:
                suggestion_content += ("\n You are facing Elite enemies Sentry*3.You should prioritize killing  "
                                       "*the first or third* sentry(instead of the second one), to ensure that you never"
                                       " need to block for more than one sentry's damage."
                                       )
                Sentry_flag = 1

            if monster.monster_id == "Lagavulin":
                suggestion_content += ("\nYou are facing Elite enemy Lagavulin,The Lagavulin will awake at the "
                                       "end of its 3rd turn or when any HP damage is taken through the  Block,"
                                       "Use the three turns before the Lagavulin wakes up to prepare for the "
                                       "fight by using Powers, or Bash as the Ironclad.")
            if monster.monster_id == "GremlinLeader":
                suggestion_content += (
                    "\nYou are facing Elite enemy Gremlin Leader and their minions.Any minion from this "
                    "fight (i.e. spawned gremlins or gremlins that come in the fight) will retreat "
                    "and be defeated if the Gremlin Leader is defeated.If you lack considerable damage"
                    " to burst down the Gremlin Leader, killing the gremlins spawned will increase "
                    "the likelyhood of her not attacking (Rallying and Encouraging instead), "
                    "hence giving you turns to continue chipping her health"
                )
            if monster.monster_id == "BookOfStabbing":
                suggestion_content += ("\nYou are facing Elite enemy Book of Stabbing,It is important to try and kill"
                                       " the Book as quickly as possible, because its attacks will only get worse "
                                       "and can become overwhelming.The Book suffers greatly against  Weak,  Thorns, "
                                       "and  Strength reduction due to its scaling being solely said multi-hit attacks "
                                       "and its lack of ability to apply any kind of debuff on the player to reduce "
                                       "their ability to  Block.")

            if monster.monster_id == "TheGuardian":
                suggestion_content += ("\nYou are facing Boss The Guardian.The Guardian is a defensive-oriented boss"
                                       ", known for its Mode Shift ability. After taking 30 damage, it switches from "
                                       "Offensive Mode to Defensive Mode, changing its attack patterns. "
                                       "In Defensive Mode, it gains Block and thorns damage when attacked. "
                                       "In Offensive Mode, it deals high damage with multi-hit attacks."
                                       "Effective strategies include dealing burst damage to trigger Mode Shift quickly"
                                       ", avoiding excessive attacks during Sharp Hide, and using block to mitigate "
                                       "its high-damage attacks. Plan ahead to exploit its transition phases and "
                                       "minimize incoming damage.\n")
                for power in monster.powers:
                    if power.power_name == "Mode Shift":
                        suggestion_content += """The Guardian is in Offensive Mode now, after taking {} damage,
                        it switches to Defensive Mode,consider dealing damage to trigger the switch""".format(
                            power.amount)

                    if power.power_name == "Sharp Hide":
                        suggestion_content += """The Guardian is in Defensive Mode now,it will thorns 3 damage when attacked."""

            if monster.monster_id == "SlimeBoss":
                suggestion_content += ("\nYou are facing Boss Slime Boss.The Slime Boss is an Act 1 boss"
                                       " with a unique Split mechanic. When its HP falls below 50%, it splits into two "
                                       "smaller slimes, each with its remaining HP. It uses Goop Spray to "
                                       "weaken the player and follows up with heavy attacks. The key strategy is to "
                                       "time your damage output carefully—avoid triggering the split when the boss "
                                       "has too much HP left, or you’ll face two strong slimes instead of weaker ones."
                                       " Use area-of-effect (AoE) attacks to handle the split slimes efficiently.")
            if monster.monster_id == "Hexaghost":
                suggestion_content += ("\nYou are facing Boss Hexaghost.Hexaghost is a boss with a unique Burning Hex "
                                       "attack pattern. On its first turn, it unleashes a devastating Inferno attack, "
                                       "dealing six hits based on the player's HP (lower HP means less damage). "
                                       "After that, its attacks follow a six-turn cycle, alternating between weak hits,"
                                       " burns, and another big attack.Prioritize damage output to shorten the fight "
                                       "and manage burn cards efficiently.")
            if monster.monster_id == "TheCollector":
                suggestion_content += ("\nYou are facing Boss The Collector. Prioritize eliminating its minions---"
                                       "TorchHead*2, "
                                       "as leaving the Torch Heads alive can lead to overwhelming "
                                       "damage.")

            if monster.monster_id == "TheChamp":
                suggestion_content += ("\nYou are facing Boss The Champ.The Champ is an Act 2 boss with two distinct "
                                       "phases. In the first phase, it alternates between attacking, blocking, and "
                                       "debuffing the player with Weak and Vulnerable. When its HP drops below 50%, "
                                       "it enters the second phase, immediately purging all debuffs and gaining "
                                       "Strength. In this phase, it becomes significantly more aggressive, "
                                       "using heavy attacks and a powerful Execute, which deals massive damage. "
                                       "Generally, you need to spend the first half of the fight setting up, and then "
                                       "you need to very quickly kill the Champ once his HP drops below half.")

            if monster.monster_id == "BronzeAutomaton":
                suggestion_content += ("\nYou are facing Boss Bronze Automaton.Bronze Automaton is an Act 2 boss that "
                                       "starts the fight by summoning two Orbs, which can steal your card, attack you "
                                       "and provide blocks to The Automaton."
                                       "The Automaton cycles between strong attacks, a multi-hit attack, "
                                       "and Hyper Beam,"
                                       " a devastating attack that deals massive damage but leaves it "
                                       "Stunned (does nothing) the next turn. The Automaton also has Artifact charges, "
                                       "preventing debuffs like Weak and Vulnerable until removed. "
                                       "Its Orbs can be dangerous if left unchecked, and managing them while "
                                       "preparing for Hyper Beam is key to survival. The fight demands balancing "
                                       "offense and defense to outlast its high-damage patterns.")

        if no_attack_flag == 1:
            suggestion_content += ("\nenemies are not in attacking intention this round,"
                                   "you should prioritize dealing damage or buffing yourself.")
        if low_hp_flag:
            suggestion_content += ("\nEnemy is in low hp,check the maximum damage you can deal to see"
                                   "if you can eliminate it.")
        if len(monsters) > 1:
            suggestion_content += ("\nYou are facing multiply enemies,you should prioritize"
                                   "AOE card which can affect them all.")

        if total_damage - block >= 7:
            suggestion_content += (
                f"\nYou are facing huge incoming damage, which will make you lose {total_damage - block} hp."
                f"you should consider mitigate the damage by:"
                f"1. build block, 2.weaken enemy 3.eliminate enemy 4.using potion")
        if block >= total_damage:
            # to do: 考虑壁垒等情况..
            suggestion_content += (f"\nnow your block is greater than incoming damage,there is no need to "
                                   f"build more blocks.")

        zero_cost_card_flag = 0
        status_flag = 0
        for card in hand:
            if card.cost == 0:
                zero_cost_card_flag = 1

            if card.type == CardType.STATUS:
                status_flag = 1

            if card.name == "Body Slam" or card.name == "Body Slam+":
                suggestion_content += ("\nYou have 'Body Slam' in your Hand Pile,"
                                       "this card deals damage based on your current block,DO build block first before"
                                       "playing it. now it can deal {} damage"
                                       .format(block))
            if card.name == "Feed":
                suggestion_content += ("\nYou have 'Feed' in your Hand Pile,which deals 10 damage and exhaust."
                                       "you should use it to **eliminate** the enemy to raise 3 max hp")
            if card.name == "Feed+":
                suggestion_content += ("\nYou have 'Feed' in your Hand Pile,which deals 12 damage and exhaust."
                                       "you should use it to **eliminate** the enemy to raise 4 max hp")
            if card.name == "Self Repair":
                suggestion_content += ("\nYou have 'Self Repair' in your Hand Pile,don't forget to "
                                       "play it to heal 7 HP after combat.")
            if card.name == "Auto-Shields" or card.name == "Auto-Shields+":
                suggestion_content += ("\nYou have 'Auto-Shields' in your Hand Pile,remember it build block "
                                       "only when you have **no block** now.")

            if card.name == "Limit Break" or card.name == "Limit Break+":
                suggestion_content += ("\nYou have 'Limit Break' in your Hand Pile,remember it double your Strength.("
                                       "so don't use it when Strength is 0 ) "
                                       "You current Strength is " + str(Strength))
            if card.name == "Bludgeon":
                suggestion_content += "\nYou have 'Bludgeon' in your Hand Pile,which could deal 32 damage "
            if card.name == "Bludgeon+":
                suggestion_content += "\nYou have 'Bludgeon+' in your Hand Pile,which could deal 42 damage "

            if card.name == "Biased Cognition":
                if Artifact_flag == 0:
                    suggestion_content += ("\nYou have 'Biased Cognition' in your Hand Pile,which gain 4 focus but also"
                                           "cause continuous Focus loss. it is not favorable to use it in the early "
                                           "turns "
                                           "of long fights like bosses and some elites. ")
                else:
                    suggestion_content += ("\nYou have 'Biased Cognition' in your Hand Pile.Meanwhile, you have "
                                           "'Artifact' buff which Negates its Debuff.So, prioritize playing the "
                                           "'Biased Cognition'.")

            if card.name == "Biased Cognition+":
                if Artifact_flag == 0:
                    suggestion_content += ("\nYou have 'Biased Cognition+' in your Hand Pile,which gain 5 focus but "
                                           "also"
                                           "cause continuous Focus loss. it is not favorable to use it in the early "
                                           "turns "
                                           "of long fights like bosses and some elites. ")
                else:
                    suggestion_content += ("\nYou have 'Biased Cognition+' in your Hand Pile.Meanwhile, you have "
                                           "'Artifact' buff which Negates its Debuff.So, prioritize playing the "
                                           "'Biased Cognition'.")
            if card.name == "Flex" or card.name == "Flex+":
                suggestion_content += "\n 'Flex' can give temporary Strength.Before you attack,consider using it."

        # if zero_cost_card_flag == 1:
        #     suggestion_content += "\nYou have 0 cost cards in your Hand Pile."

        if status_flag == 1:
            suggestion_content += ("\nYou have STATUS card in your hand pile,consider exhausting it when"
                                   "having low defence pressure.")

        # if len(potion)>0:
        #     potion_list = get_lists_str(potion)
        #     suggestion_content += (f"\nNow you have these potions:{potion_list},don't forget to use them"
        #                            f"when: 1.facing Boss or Elite 2.facing great damage. ")

        if room == "Boss":
            suggestion_content += ("\nYou are facing Boss enemy,make good use of your **potion** to defeat the enemy"
                                   "")
        if room == "Elite":
            suggestion_content += ("\nYou are facing Elite enemy,make good use of your **potion** to defeat the enemy"
                                   "")

        template_string = """       
{deck_analysis}        

combat situation:
        **Floor**: {floor}, 
        **Turn Number**: {turn}, 
        **Current HP**: {hp},
        **Block**: {block},
        **Energy Available**: {energy},
        **Relics**:{relics},
        **Enemy Lists**:{monsters},
        **Draw Pile**: {drawPile},
        **Discard Pile**:{discardPile},
        **Player Status**:{pStatus}
        **Potion**:{potion}
        **Orbs**(if you are DEFECT): {orbs}
        
Previous actions in this turn:
{previous_rounds_info}

{notice}  

Hand Pile:
{hand}

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
            monsters=get_lists_str_for_m(monsters),
            drawPile=get_lists_str_with_only_name(drawPile),
            discardPile=get_lists_str_with_only_name(discardPile),
            pStatus=get_lists_str(powers),
            # output_format=outputFormat,
            orbs=get_lists_str(orbs),
            previous_rounds_info=previous_rounds_info,
            notice=suggestion_content,
            deck_analysis=deck_analysis,
            potion=get_lists_str(potion)
        )
        self.humanM = messages[0].content
        state = State(messages=messages, turn=turn, current_hp=current_hp, max_hp=max_hp,
                      block=block, energy=energy, relics=relics, hand=hand,
                      monsters=monsters, drawPile=drawPile, discardPile=discardPile,
                      powers=powers, orbs=orbs,potion=potion,floor=floor)
        if config is not None:
            result = self.graph.invoke(state, config)
        else:
            result = self.graph.invoke(state)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        self.total_invoke_time += elapsed_time
        self.cnt +=1

        # 添加round信息到队列
        available_monsters = [monster for monster in monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]

        card_to_play = None
        if isinstance(self.card_Index,int) and 0 <= self.card_Index < len(hand):
            card_to_play = hand[self.card_Index]
        potion_to_use = None
        if isinstance(self.potion_index,int) and 0 <= self.potion_index <len(potion):
            potion_to_use = potion[self.potion_index]


        target1 = None
        if self.target_index is not None and self.target_index != -1 and 0 <= self.target_index < len(
                available_monsters):
            target1 = available_monsters[self.target_index]

        operation = ""
        if self.action == 'end':
            operation += "END turn"
        elif self.action == 'potion':
            operation += f"use potion '{potion_to_use.potion_id}'"
            if potion_to_use.requires_target and target1 is not None:
                operation += f" towards '{target1.name}(target_index={self.target_index})'"
        elif self.action == 'card':
            operation += f"choose card '{card_to_play.name}'"
            if card_to_play.has_target and target1 is not None:
                operation += f" towards '{target1.name}(target_index={self.target_index})'"

        # round_info = f"{{ turn:{turn},operation:{operation} }}"
        round_info = {
            'floor': floor,
            'turn': turn,
            'operation': operation
        }
        self.previous_rounds_info.append(round_info)

        # 输出log
        with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'a') as file:
            file.write('--------------round start-------------------------\n')
            # file.write("System:\n" + self.battle_agent_sys_prompt + '\n')
            for response in result["messages"]:
                file.write(type(response).__name__ + ":\n" + response.content.__str__() + '\n')
            file.write(f"invoke time: {elapsed_time:.6f} s\n")
            file.write(f"average invoke time: {float(self.total_invoke_time)/self.cnt:.6f} s\n")
            file.write(f"error rate:{(float(self.error_invoke_cnt)/self.total_invoke_cnt)*100 :.3f}%\n")
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
