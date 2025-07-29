import json
import multiprocessing
import threading
import tkinter as tk
from collections import deque
from tkinter import ttk, simpledialog

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from spirecomm.spire.card import Card
from spirecomm.spire.potion import Potion
from spirecomm.spire.relic import Relic


def get_lists_str_with_only_name(lists):
    ret = "[ "
    for index,item in enumerate(lists):
        ret += item.name
        if index != len(lists)-1:
            ret += ", "
    ret += " ]"
    return ret

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


class BattleAgentGUI:
    def __init__(self, root , battle_rounds_info,role):
        self.sys_prompt = None
        self.humanM = None
        self.root = root
        self.root.title("Battle Agent GUI")
        self.root.geometry("1200x800")

        # 初始化战斗变量
        self.floor = 0
        self.turn = 0
        self.current_hp = 0
        self.max_hp = 0
        self.block = 0
        self.energy = 0
        self.relics = []
        self.hand = []
        self.monsters = []
        self.drawPile = []
        self.discardPile = []
        self.powers = []
        self.orbs = []
        self.deck_analysis = ""
        self.potion = []
        self.room = ""
        self.previous_rounds_info = battle_rounds_info
        self.role = role

        # 输出变量
        self.action = None
        self.card_Index = -1
        self.potion_index = -1
        self.target_index = -1
        self.explanation = ""


        self.action_var = tk.IntVar()
        self.create_layout()

    def handle_key_press(self, event):
        """处理键盘按键，数字键1~9选择对应卡牌"""
        if event.char.isdigit():
            card_index = int(event.char) - 1  # 转换为0-based索引
            if 0 <= card_index < len(self.hand):
                self.select_action("card", card_index=card_index)

    def create_layout(self):
        # ===== 模拟调用按钮 =====
        self.debug_button = ttk.Button(self.root, text="Simulate Invoke", command=self.debug_invoke)
        self.debug_button.pack(side="bottom", pady=5)

        # ===== 上方：药水栏 =====
        self.potion_frame = ttk.LabelFrame(self.root, text="Potions", padding=10)
        self.potion_frame.pack(side="top", fill="x")
        self.potion_labels = []
        for idx, potion in enumerate(self.potion):
            btn = ttk.Button(self.potion_frame, text=potion, command=lambda i=idx: self.select_action("potion", potion_index=i))
            btn.pack(side="left", padx=5)
            self.potion_labels.append(btn)

        # ===== 左侧：人物状态栏 =====
        self.player_frame = ttk.LabelFrame(self.root, text="Player", padding=10)
        self.player_frame.pack(side="left", fill="y")
        self.hp_label = ttk.Label(self.player_frame, text=f"HP: {self.current_hp} / {self.max_hp}", padding=5)
        self.block_label = ttk.Label(self.player_frame, text=f"Block: {self.block}", padding=5)
        self.energy_label = ttk.Label(self.player_frame, text=f"Energy: {self.energy}", padding=5)
        self.relics_label = ttk.Label(self.player_frame, text=f"Relics: {get_lists_str_with_only_name(self.relics)}", padding=5)
        for widget in [self.hp_label, self.block_label, self.energy_label, self.relics_label]:
            widget.pack(anchor="w")

        # ===== 下方：卡牌列表 =====
        self.hand_frame = ttk.LabelFrame(self.root, text="Hand Cards", padding=10)
        self.hand_frame.pack(side="bottom", fill="x")
        self.card_buttons = []
        for idx, card in enumerate(self.hand):
            btn = ttk.Button(self.hand_frame, text=card.__str__(), command=lambda i=idx: self.select_action("card", card_index=i))
            btn.pack(side="left", padx=10)
            self.card_buttons.append(btn)

        self.root.bind("<Key>", self.handle_key_press)



        # ===== 右侧：敌人列表 =====
        self.enemy_frame = ttk.LabelFrame(self.root, text="Enemies", padding=10)
        self.enemy_frame.pack(side="right", fill="y")
        self.enemy_labels = []
        for monster in self.monsters:
            label = ttk.Label(self.enemy_frame, text=monster.__str__(), relief="groove", padding=10)
            label.pack(pady=10)
            self.enemy_labels.append(label)

        # ===== End Turn 按钮 =====
        self.end_turn_button = ttk.Button(self.root, text="End Turn", command=lambda: self.select_action("end"))
        self.end_turn_button.pack(side="bottom", pady=5)

        # ===== Flush 按钮 =====
        self.flush_button = ttk.Button(self.root, text="Flush", command=lambda: self.flush())
        self.flush_button.pack(side="bottom", pady=5)


        # 绑定e键到End Turn按钮
        self.root.bind('e', lambda event: self.select_action("end"))
        self.root.bind('E', lambda event: self.select_action("end"))  # 大写E也绑定
        # 绑定f键到flush
        self.root.bind('f', lambda event: self.flush())
        self.root.bind('F', lambda event: self.flush())

    def select_action(self, action_type, card_index=-1, potion_index=-1):
        self.action = action_type
        self.card_Index = card_index
        self.potion_index = potion_index
        self.target_index = -1
        self.explanation = ""

        # Create a custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Action Details")
        dialog.geometry("400x300")
        dialog.resizable(False, False)

        # Make the dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Target selection (only needed for card/potion with monsters that require target)
        target_frame = ttk.Frame(dialog)
        target_frame.pack(pady=5, fill="x")

        self.target_entry = None  # 保存target_entry引用

        if (action_type == "card" and len(self.monsters) > 0 and
                (card_index < len(self.hand) and self.hand[card_index].has_target)):
            ttk.Label(target_frame, text="Target:").pack(side="left")
            self.target_var = tk.StringVar()
            self.target_entry = ttk.Entry(target_frame, textvariable=self.target_var)
            self.target_entry.pack(side="left", padx=5)
            ttk.Label(target_frame, text=f"(1 ~ {len(self.monsters) })").pack(side="left")
        elif (action_type == "potion" and len(self.monsters) > 0 and
              (potion_index < len(self.potion) and self.potion[potion_index].requires_target)):
            ttk.Label(target_frame, text="Target:").pack(side="left")
            self.target_var = tk.StringVar()
            self.target_entry = ttk.Entry(target_frame, textvariable=self.target_var)
            self.target_entry.pack(side="left", padx=5)
            ttk.Label(target_frame, text=f"(1 ~ {len(self.monsters) })").pack(side="left")

        # Explanation
        ttk.Label(dialog, text="Explanation:").pack(anchor="w", padx=10)
        self.explanation_text = tk.Text(dialog, height=8, width=40)
        self.explanation_text.pack(padx=10, pady=5, fill="both", expand=True)

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def on_ok():
            # Process target
            if self.target_entry is not None:
                try:
                    target_str = self.target_var.get().strip()
                    self.target_index = int(target_str)-1 if target_str else 0
                except ValueError:
                    self.target_index = 0

            # Process explanation
            self.explanation = self.explanation_text.get("1.0", "end").strip()

            # print("=== Selection Complete ===")
            # print("Action:", self.action)
            # print("Card Index:", self.card_Index)
            # print("Potion Index:", self.potion_index)
            # print("Target Index:", self.target_index)
            # print("Explanation:", self.explanation)

            dialog.destroy()
            self.action_var.set(1)  # Notify invoke() to continue

        def on_cancel():
            dialog.destroy()
            # self.action_var.set(0)  # Reset without completing the action


        ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
        ok_button.pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=10)

        # 绑定Enter键到OK按钮
        dialog.bind('<Return>', lambda event: on_ok())
        dialog.bind('<Escape>', lambda event: on_cancel())

        # 默认聚焦到Target输入框（如果存在）
        if self.target_entry is not None:
            self.target_entry.focus_set()
        else:
            self.explanation_text.focus_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        # Wait for the dialog to close
        self.root.wait_window(dialog)

    def flush(self):
        self.action = "flush"
        self.action_var.set(1)  # Notify invoke() to continue

    def invoke(self, floor, turn, current_hp, max_hp, block, energy,
               relics, hand, monsters, drawPile, discardPile,
               powers, orbs, deck_analysis, potion, room,config):
        # 更新 Agent 状态
        self.floor = floor
        self.turn = turn
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.block = block
        self.energy = energy
        self.relics = relics
        self.hand = hand
        self.monsters = monsters
        self.drawPile = drawPile
        self.discardPile = discardPile
        self.powers = powers
        self.orbs = orbs
        self.deck_analysis = deck_analysis
        self.potion = potion
        self.room = room

        # 清空并重建界面
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_layout()
        # print("invoke start!")

        self.action = None  # 清空旧状态
        self.card_Index = -1
        self.potion_index = -1
        self.target_index = -1
        self.explanation = ""
        self.action_var.set(0)  # 重置等待变量

        self.root.wait_variable(self.action_var)  # 阻塞等待用户选择完成
        # print("invoke end")

        # 生成数据集
        err = self.generate_dataset(floor, turn, current_hp, max_hp, block, energy,
               relics, hand, monsters, drawPile, discardPile,
               powers, orbs, deck_analysis, potion, room,config)
        if err:
            with open(r'C:\Users\32685\Desktop\spirecomm\output\battle_agent_gui.txt', 'a') as file:
                file.write('--------------round start-------------------------\n')
                # file.write("System:\n" + self.battle_agent_sys_prompt + '\n')
                file.write("Error:\n" + err.__str__()+ '\n')
                file.write('--------------round end-------------------------\n')

        res = {
            "messages":[
                HumanMessage(content="get Answer from human input."),
            ]
        }
        return res

    def generate_dataset(self, floor, turn, current_hp, max_hp, block, energy,
               relics, hand, monsters, drawPile, discardPile,
               powers, orbs, deck_analysis, potion, room,config):

        # to do: 添加校验，没有通过校验直接return
        self.humanM = 'verification fails!'

        available_monsters = [monster for monster in monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        playable_cards = [card for card in hand if card.is_playable]
        hand_cards = hand
        potions = potion
        card_to_play1 = None
        potion_to_use = None
        if self.action == 'end':
            do_nothing = 1
        elif self.action == 'card':
            # play card
            if isinstance(self.card_Index, int) and 0 <= self.card_Index < len(hand_cards):
                card_to_play1 = hand_cards[self.card_Index]
                if not card_to_play1.is_playable:
                    if card_to_play1.cost > energy:
                        return {
                            "messages": [
                                {"role": "user",
                                 "content": f"Your chosen card({card_to_play1.name})'s cost({card_to_play1.cost}) is greater than your energy({energy})!,"
                                            " please regenerate your answer!"}]
                        }

                    return {
                        "messages": [{"role": "user", "content": "Your chosen card is not playable,"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    "messages": [{"role": "user",
                                  "content": f"Your card_Index is out of range(index ranging from 0 to {len(hand_cards) - 1}),"
                                             " please regenerate your answer!"}]
                }
        elif self.action == 'potion':
            # use potion
            if isinstance(self.potion_index, int) and 0 <= self.potion_index < len(potions):
                potion_to_use = potions[self.potion_index]
                if not potion_to_use.can_use:
                    return {
                        "messages": [{"role": "user", "content": "Your chosen potion can not be used,"
                                                                 " please regenerate your answer!"}]
                    }
            else:
                return {
                    "messages": [{"role": "user",
                                  "content": f"Your potion_index is out of range(index ranging from 0 to {len(potions)}),"
                                             " please regenerate your answer!"}]
                }
        else:
            return {
                "messages": [{"role": "user", "content": "You should choose your action from ['card','potion','end'],"
                                                         " please regenerate your answer!"}]
            }

        target1 = None
        if isinstance(self.target_index, int) and 0 <= self.target_index < len(
                available_monsters):
            target1 = available_monsters[self.target_index]
        elif self.target_index == -1:
            target1 = None
        else:
            return {
                "messages": [{"role": "user",
                              "content": f"Your target_index is out of range(index ranging from 0 to {len(available_monsters) - 1}),"
                                         " please regenerate your answer!"}]
            }

        if card_to_play1 is not None:
            if target1 is None:
                if card_to_play1.has_target:
                    if self.target_index == -1:
                        return {
                            "messages": [{"role": "user",
                                          "content": f"Your chosen card(({card_to_play1.name})) must have a target(targetIndex can't be -1),"
                                                     " please regenerate your answer!"}]
                        }

        if potion_to_use is not None:
            if target1 is None:
                if potion_to_use.requires_target:
                    return {
                        "messages": [{"role": "user",
                                      "content": "Your chosen potion must have a target(targetIndex can't be -1),"
                                                 " please regenerate your answer!"}]
                    }



        # 获取当前turn的所有action
        previous_rounds_info = '['
        for item in list(self.previous_rounds_info):

            tmp_turn = item['turn']
            tmp_floor = item['floor']
            if tmp_turn == turn and tmp_floor == floor:
                tmp_str = f"{{ turn:{tmp_turn},operation:{item['operation']} }}"
                previous_rounds_info += (tmp_str + '\n')
        previous_rounds_info += ']'



        # 添加round信息到队列
        available_monsters = [monster for monster in monsters if
                              monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]

        card_to_play = None
        if isinstance(self.card_Index, int) and 0 <= self.card_Index < len(hand):
            card_to_play = hand[self.card_Index]
        potion_to_use = None
        if isinstance(self.potion_index, int) and 0 <= self.potion_index < len(potion):
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



        # 生成humanM
        template_string ="""       
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
            notice='',
            deck_analysis=deck_analysis,
            potion=get_lists_str(potion)
        )
        self.humanM = messages[0].content



        # 生成systemM
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
        outputFormat = output_parser.get_format_instructions()
        system_msg = f"""You are an AI designed to play *Slay the Spire* as {self.role} and make optimal card choices during combat. 
### Basic Combat Rules:
- Each turn begins with **MAX_ENERGY** and a hand drawn from the Draw Pile.
- You can only play cards in your **Hand Pile**. Each card has an energy cost.
- Each call to this AI corresponds to **one action only**:  
  - Play **one card** (if affordable),  
  - Use **one potion**, or  
  - **End the turn**.
- Potions can be used at any time but are one-time use.
- You cannot play multiple cards or use multiple potions in one call.

### Deck Analysis:
An overview of your current deck is provided to inform synergy-aware decisions.

### Combat situation:
You are given real-time combat info:
- **Floor**: 'floor'
- **Turn Number**: 'turn_number'
- **Current HP**: 'current_hp' / 'max_hp'
- **Block**: 'block'
- **Energy Available**: 'energy'
- **Relics**: [ "Relic Name" ]
- **Enemy List**:  
  Each enemy: `"enemy_name(current_hp, intent, block, [status_effects])"`  
  - `intent`: e.g. `"attack(9*2)"`, `"buff"`, `"debuff"`
- **Draw Pile**: [ "Card Name" ]
- **Discard Pile**: [ "Card Name" ]
- **Player Status**: [ "status_name" ]  
- **Potions**: [ "potion_name(is_target_required)" ]

### Previous Actions (This Turn):
A list of prior actions during this turn:  
[ 
  {{ turn: int, operation: str }}, //first action in this turn
  {{ turn: int, operation: str }} // Example: {{ turn: 3, operation: choose card 'Defend'  }}
  .......
  {{ turn: int, operation: str }}, // last action in this turn
]

### Notice:
Extra information or special effects you should be aware of (e.g., upcoming massive attacks, special relic interactions).

### Hand Pile (Cards to Choose From):
List of playable cards this turn:  
[ "card_name(card_cost, is_target_required, card_type)" ]  
Where:
- `card_cost`: int
- `is_target_required`: true/false
- `card_type`: "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"

### Response Format:
{outputFormat}

### Explanation Guidelines:
Your explanation should briefly justify your decision using the following structure:
-- State current threat or opportunity
-- Explain card/potion effect and why it's chosen
-- Mention why other options were suboptimal (if relevant)

"""
        self.sys_prompt = system_msg

        #生成dataset
        ai_m = f"""
```json
{{
    "action":"{self.action}",
    "cardIndex":{self.card_Index},
    "potionIndex":{self.potion_index},
    "targetIndex":{self.target_index},
    "explanation":"{self.explanation}"
}}
```
"""
        item = {
            "conversations": []
        }
        sys = {
            "role": "system",
            "content": self.sys_prompt,
        }
        human = {
            "role": "user",
            "content": self.humanM,
        }
        ai = {
            "role": "assistant",
            "content": ai_m
        }
        item["conversations"].append(sys)
        item["conversations"].append(human)
        item["conversations"].append(ai)
        user_name = "human"
        if 0 <= floor <= 16:
            with open(fr'C:\Users\32685\Desktop\spirecomm\dataset\dataset_{user_name}_act1.jsonl', 'a',
                      encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif 16 < floor <= 33:
            with open(fr'C:\Users\32685\Desktop\spirecomm\dataset\dataset_{user_name}_act2.jsonl', 'a',
                      encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(fr'C:\Users\32685\Desktop\spirecomm\dataset\dataset_{user_name}_act3.jsonl', 'a',
                      encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(r'C:\Users\32685\Desktop\spirecomm\output\battle_agent_gui.txt', 'a') as file:
            file.write('--------------round start-------------------------\n')
            # file.write("System:\n" + self.battle_agent_sys_prompt + '\n')
            file.write("Human:\n"+human["content"]+'\n')
            file.write("AI:\n" + ai["content"] + '\n')
            file.write('--------------round end-------------------------\n')



    def debug_invoke(self):
        self.invoke(
            floor=2,
            turn=2,
            current_hp=60,
            max_hp=75,
            block=5,
            energy=2,
            relics=[Relic(name="Anchor",relic_id="Anchor"), Relic(name="Lantern",relic_id="Lantern")],
            hand=[Card(name="Defend",has_target=False,card_id='',card_type='',rarity=''), Card(name="Strike",has_target=True,card_id='',card_type='',rarity=''),],
            monsters=["Gremlin (HP: 30)", "Sentry (HP: 40)"],
            drawPile=["Strike", "Defend"],
            discardPile=["Dualcast"],
            powers=["Strength+2"],
            orbs=["Lightning"],
            deck_analysis="Contains basic strikes and defends.",
            potion=[Potion(requires_target=False,name="Strength Potion",can_use='',potion_id="Strength Potion",can_discard=''), Potion(requires_target=True,name="Dexterity Potion",can_use='',potion_id="Dexterity Potion",can_discard='')],
            room="Elite",
            config=None
        )

def run_gui():
    root = tk.Tk()
    app = BattleAgentGUI(root,battle_rounds_info=deque(maxlen=5),role="THE DEFECT")
    root.mainloop()
    # 创建并启动GUI线程

if __name__ == "__main__":


    try:
        gui_process = multiprocessing.Process(target=run_gui)
        gui_process.start()
    except Exception as e:
        print(e)
    print("main")
    # root.mainloop()
