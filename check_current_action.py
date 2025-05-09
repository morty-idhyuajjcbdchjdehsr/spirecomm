from collections import Counter, defaultdict


def count_current_actions(file_path):
    action_counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('self.game.current_action:'):
                action = line.strip().split('self.game.current_action:')[1]
                action_counter[action] += 1

    print("current_action 的种类和出现次数：")
    for action, count in action_counter.items():
        print(f"{action}: {count}")

def parse_log(file_path):
    # 用于存储每个 action 对应的 num_cards 计数
    action_card_stats = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as file:
        num_cards = None
        current_action = None

        for line in file:
            line = line.strip()
            if line.startswith('self.game.screen.num_cards:'):
                num_cards = int(line.split('self.game.screen.num_cards:')[1])
            elif line.startswith('self.game.current_action:'):
                current_action = line.split('self.game.current_action:')[1]
                if current_action is not None and num_cards is not None:
                    action_card_stats[current_action].append(num_cards)

    # 打印统计结果
    for action, cards in action_card_stats.items():
        print(f"Action: {action}")
        print(f"  出现次数: {len(cards)}")
        print(f"  num_cards 分布: {dict(Counter(cards))}")
        print()

    return action_card_stats

# 用你的文件路径替换下面的路径
# count_current_actions(r'C:\Users\32685\Desktop\spirecomm\hand_select_situation.txt')
parse_log(r'C:\Users\32685\Desktop\spirecomm\hand_select_situation.txt')