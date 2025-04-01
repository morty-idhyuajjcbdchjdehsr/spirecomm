import itertools
from datetime import datetime
import logging
import sys

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass


if __name__ == "__main__":
    # 配置 logging
    logging.basicConfig(
        filename=r'C:\Users\32685\Desktop\spirecomm\error_log.txt',  # 输出到的文件名
        level=logging.ERROR,  # 记录的最低级别
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )

    try:
        # 可能引发异常的代码
        with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\mapInfo.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\battle_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\choose_card_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\action_list.txt', 'w') as file:
            file.write('--------------tracking---------------\n')

        agent = SimpleAgent()
        coordinator = Coordinator()
        coordinator.signal_ready()

        coordinator.register_command_error_callback(agent.handle_error)
        coordinator.register_state_change_callback(agent.get_next_action_in_game_new)
        coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

        # Play games forever, cycling through the various classes
        for chosen_class in itertools.cycle(PlayerClass):
            agent.change_class(chosen_class)
            agent.init_llm_env()
            agent.init_common_llm()
            agent.init_battle_llm()
            agent.init_choose_card_llm()
            agent.init_make_map_choice_llm()
            seed = "2ZK5PHFXAGB3X" # 重锤开，二层圆顶
            seed = "3FR420LZN9M7H" # 35层，力量战
            seed = "16G2XGIZWIVPY" # 44层，鸡煲
            seed = "16QXPYKRH7U5W" # 50层，毒贼
            seed = "2IEMKEY2CBQAZ" # 33层，鸡煲，鸟居钛合金棒

            result = coordinator.play_one_game(chosen_class)
            with open(r'C:\Users\32685\Desktop\spirecomm\results.txt', 'a') as file:
                if result:
                    file.write(f"win as {chosen_class} at {datetime.now()}\n")
                else:
                    file.write(f"lose as {chosen_class} at {datetime.now()} at floor {agent.game.floor}\n")

        # while True:
        #     agent.change_class(PlayerClass.WATCHER)
        #
        #     agent.init_llm_env()
        #     agent.init_common_llm()
        #     agent.init_battle_llm()
        #     agent.init_choose_card_llm()
        #     agent.init_make_map_choice_llm()
        #
        #     result = coordinator.play_one_game(PlayerClass.WATCHER)
        #     with open(r'C:\Users\32685\Desktop\spirecomm\results.txt', 'a') as file:
        #         if result:
        #             file.write(f"win as {PlayerClass.WATCHER} at {datetime.now()}\n")
        #         else:
        #             file.write(f"lose as {PlayerClass.WATCHER} at {datetime.now()} at floor {agent.game.floor}\n")
    except Exception as e:
        # 将错误信息记录到文件
        logging.error("An error occurred: %s\n\n\n\n\n", str(e), exc_info=True)

