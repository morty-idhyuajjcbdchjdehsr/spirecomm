import json
import os
import time
import random
from collections import deque

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

from spirecomm.ai.battle_agent import BattleAgent
from spirecomm.ai.choose_card_agent import ChooseCardAgent
from spirecomm.ai.grid_choice_agent import SimpleGridChoiceAgent
from spirecomm.spire.game import Game, RoomPhase
from spirecomm.spire.character import Intent, PlayerClass
import spirecomm.spire.card
from spirecomm.spire.map import Node
from spirecomm.spire.screen import RestOption
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


class SimpleAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        self.simple_grid_chice_agent = None
        self.deck_analysis = ''
        self.search_llm = None
        self.choose_card_thread_id = None
        self.map_thread_id = None
        self.battle_thread_id = None
        self.common_agent = None
        self.llm = None
        self.make_map_choice_agent = None
        self.role = None
        self.battle_output_parser = None
        self.choose_card_agent = None
        self.choose_card_output_parser = None
        self.thread_id = ''
        self.battle_agent = None
        self.game = Game()
        self.errors = 0
        self.choose_good_card = False
        self.skipped_cards = False
        self.visited_shop = False
        self.map_route = []
        self.chosen_class = chosen_class
        self.priorities = Priority()
        # self.change_class(chosen_class)
        self.in_game_action_list = deque(maxlen=50)

    def change_class(self, new_class):
        self.chosen_class = new_class
        role = ''
        if self.chosen_class == PlayerClass.THE_SILENT:
            self.priorities = SilentPriority()
            role = "THE_SILENT"
        elif self.chosen_class == PlayerClass.IRONCLAD:
            self.priorities = IroncladPriority()
            role = "IRONCLAD"
        elif self.chosen_class == PlayerClass.DEFECT:
            self.priorities = DefectPowerPriority()
            role = "DEFECT"
        elif self.chosen_class == PlayerClass.WATCHER:
            self.priorities = DefectPowerPriority()
            role = "WATCHER"
        else:
            self.priorities = random.choice(list(PlayerClass))
        self.role = role

    def handle_error(self, error):
        # raise Exception(error)
        with open(r'C:\Users\32685\Desktop\spirecomm\error_log.txt', 'a') as file:
            file.write("error occurs!!:\n"+error.__str__()+"\n\n")

    def get_next_action_in_game(self, game_state):
        self.game = game_state
        # time.sleep(0.07)
        if self.game.choice_available:
            return self.handle_screen()
        if self.game.proceed_available:
            return ProceedAction()
        if self.game.play_available:

            potion_action = None

            # discard SmokeBomb! No surrender!
            for potion in self.game.get_real_potions():
                if potion.can_use:
                    if potion.potion_id == "SmokeBomb":
                        return PotionAction(False, potion=potion)


            if self.game.room_type == "MonsterRoomBoss" and len(self.game.get_real_potions()) > 0:
                potion_action = self.use_next_potion()

            if (self.game.room_type == "MonsterRoom" and len(self.game.get_real_potions()) > 0
                    and self.game.current_hp <= 30):
                potion_action = self.use_next_potion()

            if (self.game.room_type == "MonsterRoomElite" and len(self.game.get_real_potions()) > 0
                    and self.game.current_hp <= 50):
                potion_action = self.use_next_potion()

            if potion_action is not None and not isinstance(self.in_game_action_list[-1], PotionAction):
                return potion_action
            return self.get_play_card_action()
        if self.game.end_available:
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def get_next_action_in_game_new(self, game_state):
        action = self.get_next_action_in_game(game_state)
        self.in_game_action_list.append(action)
        return action

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)

    def is_monster_attacking(self):
        for monster in self.game.monsters:
            if monster.intent.is_attack() or monster.intent == Intent.NONE:
                return True
        return False

    def get_incoming_damage(self):
        incoming_damage = 0
        for monster in self.game.monsters:
            if not monster.is_gone and not monster.half_dead:
                if monster.move_adjusted_damage is not None:
                    incoming_damage += monster.move_adjusted_damage * monster.move_hits
                elif monster.intent == Intent.NONE:
                    incoming_damage += 5 * self.game.act
        return incoming_damage

    def get_low_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = min(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def get_high_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = max(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def many_monsters_alive(self):
        available_monsters = [monster for monster in self.game.monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        return len(available_monsters) > 1

    def get_play_card_action(self):


        available_monsters = [monster for monster in self.game.monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        playable_cards = [card for card in self.game.hand if card.is_playable]
        hand_cards = self.game.hand
        config = {"configurable": {"thread_id": self.battle_thread_id}}
        responses = self.battle_agent.invoke(
            floor=self.game.floor,
            turn=self.game.turn,
            current_hp=self.game.current_hp,
            max_hp=self.game.max_hp,
            block=self.game.player.block,
            energy=self.game.player.energy,
            relics=self.game.relics,
            hand=self.game.hand,
            monsters=available_monsters,
            drawPile=self.game.draw_pile,
            discardPile=self.game.discard_pile,
            powers=self.game.player.powers,
            # output_format=outputFormat,
            orbs=self.game.player.orbs,
            deck_analysis=self.deck_analysis,
            config=config
        )

        is_to_end_turn = self.battle_agent.is_to_end_turn
        card_Index = self.battle_agent.card_Index
        target_index = self.battle_agent.target_index
        explanation = self.battle_agent.explanation

        with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'a') as file:
            # file.write('--------------executing get_play_card_action---------------\n')
            # file.write(self.game.__str__())
            file.write('--------------human message--------------------------------\n')
            file.write(self.battle_agent.humanM+"\n")
            file.write("--------------ai message-----------------------------------\n")
            file.write(responses["messages"][-1].content + "\n")
            file.write("self.card_index:"+str(self.battle_agent.card_Index)+ "\n")
            file.write("self.is_to_end_turn:"+str(self.battle_agent.is_to_end_turn)+ "\n")
            file.write("self.target_index:"+str(self.battle_agent.target_index)+ "\n")
            file.write("self.explanation:"+str(self.battle_agent.explanation)+ "\n")
            # file.write("--------------ai stream------------------------------------\n")
            # for response in responses["messages"]:
            #     file.write(type(response).__name__+" "+response.__str__())
            #     file.write("\n\n")

            # file.write("--------------verify json file-----------------------------------\n")
            # file.write(jsonfile.__str__())

        with open(r'C:\Users\32685\Desktop\spirecomm\state.txt', 'w') as file:
            file.write('--------------current state---------------\n')
            file.write(self.game.__str__())


        card_to_play1 = None
        if 0 <= card_Index < len(hand_cards):
            card_to_play1 = hand_cards[card_Index]

        target1 = None
        if target_index != -1 and 0 <= target_index < len(available_monsters):
            target1 = available_monsters[target_index]

        if is_to_end_turn == 'Yes':
            return EndTurnAction()
        if card_to_play1 is not None:
            if target1 is None:
                if card_to_play1.has_target:
                    with open(r'C:\Users\32685\Desktop\spirecomm\error_log.txt', 'a') as file:
                        file.write(f"\n the card must have a target!!!!!!!!!,targetIndex is {target_index}\n\n")
                    return PlayCardAction(card=card_to_play1, target_monster=available_monsters[0])
                else:
                    return PlayCardAction(card=card_to_play1)
            else:
                return PlayCardAction(card=card_to_play1, target_monster=target1)
        else:
            # just algorithm
            playable_cards = [card for card in self.game.hand if card.is_playable]
            zero_cost_cards = [card for card in playable_cards if card.cost == 0]
            zero_cost_attacks = [card for card in zero_cost_cards if card.type == spirecomm.spire.card.CardType.ATTACK]
            zero_cost_non_attacks = [card for card in zero_cost_cards if card.type != spirecomm.spire.card.CardType.ATTACK]
            nonzero_cost_cards = [card for card in playable_cards if card.cost != 0]
            aoe_cards = [card for card in playable_cards if self.priorities.is_card_aoe(card)]
            if self.game.player.block > self.get_incoming_damage() - (self.game.act + 4):
                offensive_cards = [card for card in nonzero_cost_cards if not self.priorities.is_card_defensive(card)]
                if len(offensive_cards) > 0:
                    nonzero_cost_cards = offensive_cards
                else:
                    nonzero_cost_cards = [card for card in nonzero_cost_cards if not card.exhausts]
            if len(playable_cards) == 0:
                return EndTurnAction()
            if len(zero_cost_non_attacks) > 0:
                card_to_play = self.priorities.get_best_card_to_play(zero_cost_non_attacks)
            elif len(nonzero_cost_cards) > 0:
                card_to_play = self.priorities.get_best_card_to_play(nonzero_cost_cards)
                if len(aoe_cards) > 0 and self.many_monsters_alive() and card_to_play.type == spirecomm.spire.card.CardType.ATTACK:
                    card_to_play = self.priorities.get_best_card_to_play(aoe_cards)
            elif len(zero_cost_attacks) > 0:
                card_to_play = self.priorities.get_best_card_to_play(zero_cost_attacks)
            else:
                # This shouldn't happen!
                return EndTurnAction()
            if card_to_play.has_target:
                available_monsters = [monster for monster in self.game.monsters if
                                      monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
                if len(available_monsters) == 0:
                    return EndTurnAction()
                if card_to_play.type == spirecomm.spire.card.CardType.ATTACK:
                    target = self.get_low_hp_target()
                else:
                    target = self.get_high_hp_target()
                return PlayCardAction(card=card_to_play, target_monster=target)
            else:
                return PlayCardAction(card=card_to_play)

    def use_next_potion(self):
        for potion in self.game.get_real_potions():
            if potion.can_use:
                if potion.requires_target:
                    return PotionAction(True, potion=potion, target_monster=self.get_low_hp_target())
                else:
                    return PotionAction(True, potion=potion)

    def handle_screen(self):
        if self.game.screen_type == ScreenType.EVENT:
            if self.game.screen.event_id in ["Vampires", "Masked Bandits", "Knowing Skull", "Ghosts", "Liars Game",
                                             "Golden Idol", "Drug Dealer", "The Library"]:
                return ChooseAction(len(self.game.screen.options) - 1)
            else:
                return ChooseAction(0)
        elif self.game.screen_type == ScreenType.CHEST:
            return OpenChestAction()
        elif self.game.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            else:
                self.visited_shop = False
                return ProceedAction()
        elif self.game.screen_type == ScreenType.REST:
            return self.choose_rest_option()
        elif self.game.screen_type == ScreenType.CARD_REWARD:
            return self.choose_card_reward()
        elif self.game.screen_type == ScreenType.COMBAT_REWARD:
            for reward_item in self.game.screen.rewards:
                if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                    continue
                elif reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                    continue
                else:
                    return CombatRewardAction(reward_item)
            self.skipped_cards = False
            return ProceedAction()
        elif self.game.screen_type == ScreenType.MAP:
            return self.make_map_choice()
        elif self.game.screen_type == ScreenType.BOSS_REWARD:
            relics = self.game.screen.relics
            best_boss_relic = self.priorities.get_best_boss_relic(relics)
            return BossRewardAction(best_boss_relic)
        elif self.game.screen_type == ScreenType.SHOP_SCREEN:
            if self.game.screen.purge_available and self.game.gold >= self.game.screen.purge_cost:
                return ChooseAction(name="purge")
            for card in self.game.screen.cards:
                if self.game.gold >= card.price and not self.priorities.should_skip(card):
                    return BuyCardAction(card)
            for relic in self.game.screen.relics:
                if self.game.gold >= relic.price:
                    return BuyRelicAction(relic)
            return CancelAction()
        elif self.game.screen_type == ScreenType.GRID:

            with open(r'C:\Users\32685\Desktop\spirecomm\grid.txt', 'w') as file:
                file.write('self.game.cards is:{},self.game.screen.num_cards is:{},self.game.screen.selected_cards is:{}\n\n'
                           .format(self.get_card_list_str(self.game.screen.cards),self.game.screen.num_cards,self.game.screen.selected_cards))

            # self.screen.any_number :use for Watcher to foresee
            if self.game.screen.any_number:
                return CardSelectAction([])

            # 网格选择
            if not self.game.choice_available:
                return ProceedAction()

            # 在这里写选择升级，转化，删除
            available_cards = self.game.screen.cards
            if self.game.screen.for_upgrade or self.game.screen.for_transform or self.game.screen.for_purge:
                chosen_cards = self.make_grid_choice()
                if len(chosen_cards)!=0:
                    return CardSelectAction(chosen_cards)


            if self.game.screen.for_upgrade or self.choose_good_card:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards)
            else:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards, reverse=True)
            num_cards = self.game.screen.num_cards
            return CardSelectAction(available_cards[:num_cards])
        elif self.game.screen_type == ScreenType.HAND_SELECT:
            # 选择手牌
            if not self.game.choice_available:
                return ProceedAction()
            # Usually, we don't want to choose the whole hand for a hand select. 3 seems like a good compromise.
            num_cards = min(self.game.screen.num_cards, 3)
            return CardSelectAction(
                self.priorities.get_cards_for_action(self.game.current_action, self.game.screen.cards, num_cards))
        else:
            return ProceedAction()

    def make_grid_choice(self):

        intent = ''
        if self.game.screen.for_upgrade:
            intent = 'upgrade'
        if self.game.screen.for_purge:
            intent = 'purge'
        if self.game.screen.for_transform:
            intent = 'transform'
        num_cards = self.game.screen.num_cards
        available_cards = self.game.screen.cards


        config = {"configurable": {"thread_id": self.thread_id}}
        responses = self.simple_grid_chice_agent.invoke(
            intent=intent,
            relics=self.game.relics,
            current_hp= self.game.current_hp,
            max_hp= self.game.max_hp,
            deck = self.game.deck,
            available_cards= self.game.screen.cards,
            config=config
        )

        ret = []
        error_ret = []
        card_Index = self.simple_grid_chice_agent.cardIndex
        explanation = self.simple_grid_chice_agent.explanation

        with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'a',encoding="utf-8") as file:
            # file.write('--------------executing get_play_card_action---------------\n')
            # file.write(self.game.__str__())
            file.write('--------------human message--------------------------------\n')
            file.write(self.simple_grid_chice_agent.humanM+"\n")
            file.write("--------------ai message-----------------------------------\n")
            file.write(responses["messages"][-1].content + "\n")
            file.write("self.card_Index:" + str(card_Index) + "\n")
            file.write("self.explanation:" + str(explanation) + "\n")

        chosen_card = available_cards[card_Index]
        ret.append(chosen_card)

        if len(ret) != num_cards:
            # 出bug啦！
            with open(r'C:\Users\32685\Desktop\spirecomm\error_log.txt', 'a') as file:
                file.write(f'wrong index list from "make_grid_choice"')
            return error_ret
        return ret

    def choose_rest_option(self):
        rest_options = self.game.screen.rest_options
        if len(rest_options) > 0 and not self.game.screen.has_rested:
            if RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp / 2:
                return RestAction(RestOption.REST)
            elif RestOption.REST in rest_options and self.game.act != 1 and self.game.floor % 17 == 15 and self.game.current_hp < self.game.max_hp * 0.9:
                return RestAction(RestOption.REST)
            elif RestOption.SMITH in rest_options:
                return RestAction(RestOption.SMITH)
            elif RestOption.LIFT in rest_options:
                return RestAction(RestOption.LIFT)
            elif RestOption.DIG in rest_options:
                return RestAction(RestOption.DIG)
            elif RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp:
                return RestAction(RestOption.REST)
            else:
                return ChooseAction(0)
        else:
            return ProceedAction()

    def count_copies_in_deck(self, card):
        count = 0
        for deck_card in self.game.deck:
            if deck_card.card_id == card.card_id:
                count += 1
        return count

    def choose_card_reward(self):

        config = {"configurable": {"thread_id": self.choose_card_thread_id}}

        # to do: 增加战斗期间的选牌agent
        # if self.game.room_phase == RoomPhase.COMBAT:
        #     response = self.choose_card_agent_in_battle.invoke()
        #     self.game.cancel_available

        responses = self.choose_card_agent.invoke(
            floor=self.game.floor,
            current_hp=self.game.current_hp,
            max_hp=self.game.max_hp,
            deck=self.game.deck,
            reward_cards=self.game.screen.cards,
            relic_bowl=self.game.screen.can_bowl, config=config
        )

        card_name = self.choose_card_agent.card_name
        explanation = self.choose_card_agent.explanation
        self.deck_analysis = self.choose_card_agent.strategy

        with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'a',encoding="utf-8") as file:
            # file.write('--------------executing get_play_card_action---------------\n')
            # file.write(self.game.__str__())
            file.write('--------------human message--------------------------------\n')
            file.write(self.choose_card_agent.humanM+"\n")
            file.write("--------------ai message-----------------------------------\n")
            file.write(responses["messages"][-1].content + "\n")
            file.write("self.card_name:" + str(card_name) + "\n")
            file.write("self.explanation:" + str(explanation) + "\n")
            # file.write("--------------ai stream------------------------------------\n")
            # for response in responses["messages"]:
            #     file.write(type(response).__name__+" "+response.__str__())
            #     file.write("\n\n")


        if card_name == '':
            if self.game.cancel_available:
                self.skipped_cards = True
                return CancelAction()
        elif card_name=="Bowl":
            return CardRewardAction(bowl=True)
        else:
            reward_cards = self.game.screen.cards
            card_to_choose = next((card for card in reward_cards if card.name == card_name), None)
            if card_to_choose is None:
                self.skipped_cards = True
                return CancelAction()
            return CardRewardAction(card_to_choose)


        # algorithm
        reward_cards = self.game.screen.cards
        if self.game.screen.can_skip and not self.game.in_combat:
            pickable_cards = [card for card in reward_cards if
                              self.priorities.needs_more_copies(card, self.count_copies_in_deck(card))]
        else:
            pickable_cards = reward_cards
        if len(pickable_cards) > 0:
            potential_pick = self.priorities.get_best_card(pickable_cards)
            return CardRewardAction(potential_pick)
        elif self.game.screen.can_bowl:
            return CardRewardAction(bowl=True)
        else:
            self.skipped_cards = True
            return CancelAction()

    def get_card_list_str(self, cardlist):
        str = '['
        for card in cardlist:
            str += f"{card.name}"
            if card != cardlist[-1]:
                str += ", "
        str += ']'
        return str

    def generate_map_route(self):
        node_rewards = self.priorities.MAP_NODE_PRIORITIES.get(self.game.act)
        best_rewards = {0: {node.x: node_rewards[node.symbol] for node in self.game.map.nodes[0].values()}}
        best_parents = {0: {node.x: 0 for node in self.game.map.nodes[0].values()}}
        min_reward = min(node_rewards.values())
        map_height = max(self.game.map.nodes.keys())
        for y in range(0, map_height):
            best_rewards[y + 1] = {node.x: min_reward * 20 for node in self.game.map.nodes[y + 1].values()}
            best_parents[y + 1] = {node.x: -1 for node in self.game.map.nodes[y + 1].values()}
            for x in best_rewards[y]:
                node = self.game.map.get_node(x, y)
                best_node_reward = best_rewards[y][x]
                for child in node.children:
                    test_child_reward = best_node_reward + node_rewards[child.symbol]
                    if test_child_reward > best_rewards[y + 1][child.x]:
                        best_rewards[y + 1][child.x] = test_child_reward
                        best_parents[y + 1][child.x] = node.x
        best_path = [0] * (map_height + 1)
        best_path[map_height] = max(best_rewards[map_height].keys(), key=lambda x: best_rewards[map_height][x])
        for y in range(map_height, 0, -1):
            best_path[y - 1] = best_parents[y][best_path[y]]
        self.map_route = best_path

    def make_map_choice(self):

        with open(r'C:\Users\32685\Desktop\spirecomm\mapInfo.txt', 'a') as file:
            file.write('--------------next_nodes---------------\n')
            file.write(self.game.screen.next_nodes.__str__() + "\n")
            file.write('--------------game.map.nodes-----------\n')
            file.write(self.game.map.nodes.__str__() + "\n")

        # 特殊情况
        if self.game.screen.boss_available:
            return ChooseMapBossAction()
        if len(self.game.screen.next_nodes) == 1:
            return ChooseMapNodeAction(self.game.screen.next_nodes[0])

        # llm
        (x, y) = (self.game.screen.current_node.x, self.game.screen.current_node.y)
        current_node = self.game.map.get_node(x, y)
        # 初次进map
        if len(self.game.screen.next_nodes) > 0 and self.game.screen.next_nodes[0].y == 0:
            children = []
            for node in self.game.screen.next_nodes:
                (x,y) = (node.x,node.y)
                tmp = self.game.map.get_node(x, y)
                children.append(tmp)
            current_node = Node(x=x, y=y, symbol=self.game.screen.current_node.symbol)
            current_node.children = children

        # 将树转换为字典
        tree_dict = current_node.to_dict(max_depth=4)
        # 将字典转换为 JSON 字符串
        tree_json = json.dumps(tree_dict, indent=4)

        template_string = """ now you need to choose next level node,
                and the below is the context:
                - **Map Tree**: {tree}
                - **Choice List**: {choice_list}
                - **Current HP**: {hp}
                - **Current Deck**: {deck}
                - **Relics**: {relics}

                now make your choice and provide the reason.
                """
        # outputFormat = self.output_parser.get_format_instructions()
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            hp=f"{self.game.current_hp}/{self.game.max_hp}",
            deck=self.get_card_list_str(self.game.deck),
            choice_list=self.get_lists_str(self.game.screen.next_nodes),
            relics=self.get_lists_str(self.game.relics),
            tree= tree_dict.__str__(),
            # output_format=outputFormat,
        )
        config = {"configurable": {"thread_id": self.map_thread_id}}
        responses = self.make_map_choice_agent.invoke(
            {"messages": messages}, config
        )
        response_text = responses["messages"][-1].content
        start = response_text.rfind('```json') + len('```json\n')
        end = response_text.rfind('```')
        json_text = response_text[start:end].strip()  # json文本
        index = 0 #默认为0

        try:
            # 得到最终的 json格式文件
            jsonfile = json.loads(json_text)
            index = jsonfile.get('index')
        except Exception as e:
            with open(r'C:\Users\32685\Desktop\spirecomm\error_log.txt', 'a') as file:
                file.write(f'unable to parse json_text:{json_text}\n')

        with open(r'C:\Users\32685\Desktop\spirecomm\mapInfo.txt', 'a') as file:
            file.write('--------------tree_json----------------\n')
            file.write(tree_dict.__str__() + "\n")

        with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'a') as file:
            # file.write('--------------executing get_play_card_action---------------\n')
            # file.write(self.game.__str__())
            file.write('--------------human message--------------------------------\n')
            file.write(messages[0].content+"\n")
            file.write("--------------ai message-----------------------------------\n")
            file.write(responses["messages"][-1].content + "\n")
            file.write("--------------ai stream------------------------------------\n")
            for response in responses["messages"]:
                file.write(type(response).__name__+" "+response.__str__())
                file.write("\n\n")

        if len(self.game.screen.next_nodes) > index >= 0:
            return ChooseMapNodeAction(self.game.screen.next_nodes[index])
        else:
            return ChooseMapNodeAction(self.game.screen.next_nodes[0])


        # old version
        # if len(self.game.screen.next_nodes) > 0 and self.game.screen.next_nodes[0].y == 0:
        #     self.generate_map_route()
        #     self.game.screen.current_node.y = -1
        # if self.game.screen.boss_available:
        #     return ChooseMapBossAction()
        # chosen_x = self.map_route[self.game.screen.current_node.y + 1]
        # for choice in self.game.screen.next_nodes:
        #     if choice.x == chosen_x:
        #         return ChooseMapNodeAction(choice)
        # # This should never happen
        # return ChooseAction(0)

    def init_battle_llm(self):
        # small_llm = ChatOpenAI(model="gemini-1.5-flash", temperature=0) # 4s
        # small_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0) #good
        # small_llm = ChatOllama(model="mistral:7b", temperature=0) # 缺少对卡牌的理解
        # small_llm = ChatOpenAI(model="claude-3-haiku-20240307", temperature=0) # 似乎还挺懂 话多
        # small_llm = ChatOpenAI(model="qwen-turbo-latest", temperature=0) # good 3~4s
        # small_llm = ChatOpenAI(model="qwen-plus-latest", temperature=0) # man

        agent = BattleAgent(role=self.role,llm=self.llm,small_llm=self.llm)
        self.battle_agent = agent




    def init_choose_card_llm(self):
        self.choose_card_agent = ChooseCardAgent(role=self.role,llm=self.llm,small_llm=self.llm)

    def init_make_map_choice_llm(self):
        choice_index_schema = ResponseSchema(
            name="index",
            description="The index of the chosen node from Choice List (0~9).",
            type="Int"
        )
        explanation_schema = ResponseSchema(
            name="explanation",
            description="Explanation of why you choose the node."
        )
        response_schemas = [
            choice_index_schema,
            explanation_schema
        ]
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        outputFormat = output_parser.get_format_instructions()

        system_prompt = f"""
        You are an AI designed to assist in making strategic decisions in the game "Slay the Spire" 
        as the role {self.role}.
        Your task is to analyze the current map and suggest the best next level to progress through based on 
        the current context. 
        
        ### Context:
        - **Map Tree**: a current map structure starts from the current node, represented as a tree with a maximum depth of 4:
          format:  {{ "x": self.x,"y": self.y,"name": self.name,"children": []}}
          
        - **Choice List**: [ node ] a list of nodes that represent next level choices
        - **Current HP**: 'current_hp' / 'max_hp'
        - **Current Deck** : [ card ]  a list of cards currently in the deck,
        - **Relics**: [ Relic ],the relics you have
        
        ### Instructions:
        1. Analysis the map tree.Evaluate the possible paths from the current node.
        2. Consider the names of the nodes (e.g., Merchant, Elite, Treasure, etc.) 
            to determine their potential benefits and risks.
        3. Choose the next node to progress to based on the following criteria:
           - Prioritize nodes that offer beneficial encounters (e.g., Merchant, Treasure).
           - Consider the overall strategy and current resources (hp , deck, relics) when making your decision.
           - Avoid nodes that may lead to difficult battles unless necessary for progression.
           
        
        Your response should include:
        - The index of the chosen node from Choice List (0~9).
        - A brief explanation of why this node is the best choice.
        
        ### response format:
        {outputFormat}
        
        """
        # print(system_prompt)
        tools = []
        agent = create_react_agent(self.llm, tools=tools, state_modifier=system_prompt, )
        self.make_map_choice_agent = agent

    def init_common_llm(self):
        # tools = load_tools(["wikipedia"],llm=self.search_llm)
        tools = []
        agent = create_react_agent(model=self.llm, tools=tools)
        self.common_agent = agent

    def init_simple_grid_choice_llm(self):
        agent = SimpleGridChoiceAgent(role=self.role,llm=self.llm,small_llm=self.llm)
        self.simple_grid_chice_agent = agent
    
    @tool("search_card_tool")
    def search_card(card: str) -> str:
        """to get the content of the card"""
        search = TavilySearchResults(max_results=2)
        response = search.invoke(f"what is the content of {card} in Slay the Spire?")
        # print(response)
        # print(response[-1])
        return response[-1].get('content')


    def init_llm_env(self):

        # load_dotenv()  # to do: 隐藏各个api key

        # tavity
        os.environ["TAVILY_API_KEY"] = "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"

        # chatanywhere
        # free
        # os.environ["OPENAI_API_KEY"] = "sk-KCmRtnkbFhG5H17LiQSJ9Y76EjACuiSH0Bgjq83Ld7QiBKs4"
        os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
        os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"


        #silicon
        # os.environ["OPENAI_API_KEY"] = "sk-aqhgalcbwavbbbcjbiuikgznytxmmixcveggxfmxmrjkpxkt"
        # os.environ["OPENAI_API_BASE"] ="https://api.siliconflow.cn/v1"


        #Google
        # os.environ["GEMINI_API_KEY"] = "AIzaSyDUhCQJYnjYpez1v_2BH03Kzw - sDLWYTyI"
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDUhCQJYnjYpez1v_2BH03Kzw-sDLWYTyI"

        #AGICTO
        # os.environ["OPENAI_API_KEY"] = "sk-dXqzgPLzOmxdo24tl4lqL8Ioh9udaG4ctTQMuDq1fDfwS4mM"
        # os.environ["OPENAI_API_BASE"] = "https://api.agicto.cn/v1"

        #aihubmix
        # os.environ["OPENAI_API_KEY"] = "sk-vJnE3cpi4m837up7B34971C0250b42C2818e02C517BeE44e"
        # os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"

        # self.search_llm = ChatOpenAI(model="THUDM/chatglm3-6b", temperature=0)
        self.battle_thread_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
        self.map_thread_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
        self.choose_card_thread_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))
        self.thread_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=10))


        # self.llm = ChatOpenAI(model="gemini-1.5-flash", temperature=0) #便宜
        # self.llm = ChatOpenAI(model="gemini-1.5-flash-latest", temperature=0)
        # self.llm = ChatOpenAI(model="gemini-2.0-flash-exp", temperature=0)  # 便宜
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # good
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo-ca", temperature=0)  # 史
        # self.llm = ChatOpenAI(model="gpt-4o-mini-ca", temperature=0)  # good
        self.llm = ChatOpenAI(model="deepseek-v3", temperature=0.3)  # 神！！！！！！！！！
        # self.llm = ChatOpenAI(model="claude-3-5-haiku-20241022", temperature=0.5) #shi
        # self.llm = ChatOpenAI(model="gemini-2.0-flash", temperature=0)

        # self.llm = ChatOpenAI(model="qwen-turbo-latest", temperature=0) # 便宜又快！！
        # self.llm = ChatOpenAI(model="qwen-plus-latest", temperature=0)
        # self.llm = ChatOpenAI(model="qwen-plus-0919", temperature=0.3)
        # self.llm = ChatOpenAI(model="hunyuan-turbos-latest", temperature=0)  #慢
        # self.llm = ChatOpenAI(model="ERNIE-Speed-128K", temperature=0.7) #shi
        # self.llm = ChatOpenAI(model="glm-4-flashx", temperature=0) #慢
        # self.llm = ChatOpenAI(model="gemma2-9b-it", temperature=0) # 快且免费！但是会爆
        # self.llm = ChatOpenAI(model="gemma2-7b-it", temperature=0)

        # self.llm = ChatOpenAI(model="internlm/internlm2_5-7b-chat", temperature =0) #good grid选择有问题 支持工具 shi
        # self.llm = ChatOpenAI(model="THUDM/chatglm3-6b", temperature =0) # 有点烂
        # self.llm = ChatOpenAI(model="THUDM/glm-4-9b-chat", temperature=0) # 还行，支持工具 还行 10s+
        # self.llm = ChatOpenAI(model="01-ai/Yi-1.5-9B-Chat-16K", temperature=0) # 一般
        # self.llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0) # 还行
        # self.llm = ChatOpenAI(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", temperature=0) # 要钱 慢死了
        # self.llm = ChatOpenAI(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", temperature=0)  # 7b man
        # self.llm = ChatOpenAI(model="internlm/internlm2_5-20b-chat", temperature=0)  # 20b shi
        # self.llm = ChatOpenAI(model="Pro/Qwen/Qwen2.5-7B-Instruct", temperature=0)  # 20b shi

        # self.llm = ChatOllama(model="deepseek-r1:7b", temperature=0) # 有<think>
        # self.llm = ChatOllama(model="mistral:7b", temperature=0) # 还行
        # self.llm = ChatOllama(model="hermes3:3b", temperature=0) # 还不赖
        # self.llm = ChatOllama(model="qwen2.5:3b", temperature=0)  #不赖 10s+
        # self.llm = ChatOllama(model="qwen2.5:1.5b", temperature=0) # shi
        # self.llm = ChatOllama(model="gemma2:2b", temperature=0) #man
        # self.llm = ChatOllama(model="gemma3:1b-it-q4_K_M", temperature=0) #只会e键爬塔

        # self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0,transport='rest') #有限额

        # self.llm = ChatOpenAI(model="gemini-2.0-flash-lite",temperature=0.3)
        # self.llm = ChatOpenAI(model="gemini-2.0-flash", temperature=0.3)
        # self.llm = ChatOpenAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)
        # self.llm = ChatOpenAI(model="DeepSeek-V3", temperature=0.3)



    def get_role_guidelines(self,chosen_class):

        if chosen_class == PlayerClass.IRONCLAD:
            return """
        1. **Strength Build**:
        - **Core Focus**: Maximizing Strength to deal high damage with attacks.
        - **Key Cards**: *Demon Form*, *Flex*, *Limit Break*, *Reaper*, *Sword boomerang*.
        - **Playstyle**: Focus on building Strength as quickly as possible with cards like *Demon Form* 
        (for long-term scaling)  and *Flex* (for instant Strength boosts) and *Limit Break*(Double your Strength). 
         Combine this with strong offensive cards that deal damage multiple times, such as *Sword boomerang* and 
         *Twin Strike*. Your goal is to outscale the enemies over time, becoming stronger and dealing large amounts of 
         damage whileusing *Reaper* to recover large amounts of hp.

        2. **Block Build**:
        - **Core Focus**: Building a solid defense to mitigate damage while slowly chipping away at enemies. 
            This build focuses on using block cards and Barricade to gain permanent blocks and use Body Slam
            to cause damage.
        - **Key Cards**:  *Barricade*, *Body Slam*, *Impervious*.
        - **Playstyle**: In this strategy, you should focus on building up enough block to mitigate enemy damage while 
        using Body Slam to attack enemy. The key is to play *Barricade* as soon as possible,and accumulate Blocks as it
        won't be removed next turn. After you gain enough blocks,you can use Body Slam to turn your block into damage.
        """

        elif chosen_class == PlayerClass.THE_SILENT:
            return """"""
        elif chosen_class == PlayerClass.DEFECT:
            return """
            1.**Focus Build**:
            - **Core Focus**: Increasing Focus to amplify the power of all your Orbs, allowing them to deal more damage 
            or provide more block. This build is about enhancing the effects of Orbs 
            by stacking Focus and using the synergy with your cards that generate and channel Orbs.
            - **Key Cards**: *Defragment*, *Capacitor*, *Coolheaded*, *Charge Battery*, *Echo Form*, *Biased Cognition*.
            - **Playstyle**: The Focus Build revolves around increasing your Focus to make your Orbs significantly more powerful.
            Cards like *Defragment* increase Focus, which in turn improves the potency of Orbs. 
            Once your Focus is high, even weak Orbs like *Frost* or *Lightning* can have devastating effects. 
            With a high Focus, *Echo Form* allows you to double the effectiveness of all your card plays, 
            including Orb triggers, leading to exponential growth in power.
            """
        else:
            return ""

    def get_lists_str(self,lists):
        str = "[ "
        for item in lists:
            str += (item.__str__())
            if item != lists[-1]:
                str += ", "
        str = str + " ]"
        return str
