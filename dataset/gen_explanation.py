import copy
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # cnm


# 自定义你的修改逻辑
def modify_assistant_message(messages: []) -> str:
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)

    t_messages = copy.deepcopy(messages)
    t_messages.append({"role": "user", "content": "add explanation to your response,"
                                                  "limit your explanation to 100 words."})
    response = llm.invoke(t_messages)
    print(f"content is:\n{response.content}\n\n")
    explanation = response.content
    return explanation



sys = """You are an AI designed to play *Slay the Spire* and make optimal card choices during combat. 
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
  { turn: int, operation: str }, //first action in this turn
  { turn: int, operation: str } // Example: { turn: 3, operation: choose card 'Defend'  }
  .......
  { turn: int, operation: str }, // last action in this turn
]

### Notice:
Extra information or special effects you should be aware of (e.g., upcoming massive attacks, special relic interactions).

### Hand Pile (Cards to Choose From):
List of playable cards this turn:  
[ "card_name(card_cost, is_target_required, card_type, card_index)" ]  
Where:
- `card_cost`: int
- `is_target_required`: true/false
- `card_type`: "ATTACK", "SKILL", "POWER", "CURSE", "STATUS"
- `card_index`: int

### Response Format:
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
	"action": string  // return 'card' if you choose one card to play, return 'end' if you decide to end the turnreturn 'potion' if you choose one potion to use.
	"cardIndex": Int  // The index of the card you choose from Hand Pile(Start with 0); if you don't choose a card, just return -1
	"potionIndex": Int  // The index of the potion you choose(Start with 0); if you don't choose a potion, just return -1
	"targetIndex": Int  // The index of your target in enemy list(Start with 0); if your card's attribute 'is_card_has_target' is False, just return -1.If your potion's attribute 'is_potion_has_target' is False, just return -1.
	"explanation": string  // Explanation of why you take the action.
}
```

### Explanation Guidelines:
Your explanation should briefly justify your decision using the following structure:
-- State your choice with [],e.q: [I choose card 'Strike'], [I choose to end turn],[I choose to use potion 'xxPotion']
-- Explain card/potion effect and why it's chosen
-- Mention why other options were suboptimal (if relevant)
"""
input_file = "dataset_human_act2.jsonl"
output_file = "dataset_human_act2_modified.jsonl"
start_line = 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for index,line, in enumerate(fin):
        if index < start_line-1:
            break
        item = json.loads(line)
        messages = item.get("conversations", [])

        # 遍历找到 role 为 "assistant" 的消息
        for message in messages:
            if message.get("role") == "system":
                message["content"] = sys

            if message.get("role") == "assistant":
                message["role"] = "ai"
                response_text = message["content"]
                # print(f"ori text is:\n{response_text}\n")
                message["content"] = modify_assistant_message(messages)
                message["role"] = "assistant"

        # 写入修改后的数据
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"data {index} write over.")
        # break

print("处理完成，已保存至", output_file)
