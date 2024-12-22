import itertools
import datetime
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
    while True:
        try:
            # 可能引发异常的代码
            with open(r'C:\Users\32685\Desktop\spirecomm\output.txt', 'w') as file:
                file.write('--------------tracking---------------\n')
            with open(r'C:\Users\32685\Desktop\spirecomm\mapInfo.txt', 'w') as file:
                file.write('--------------tracking---------------\n')
            agent = SimpleAgent()
            coordinator = Coordinator()
            coordinator.signal_ready()
            agent.init_llm_env()
            agent.init_battle_llm()
            agent.init_choose_card_llm()
            agent.init_make_map_choice_llm()
            coordinator.register_command_error_callback(agent.handle_error)
            coordinator.register_state_change_callback(agent.get_next_action_in_game)
            coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

            # Play games forever, cycling through the various classes
            # for chosen_class in itertools.cycle(PlayerClass):
            #     agent.change_class(chosen_class)
            #     result = coordinator.play_one_game(chosen_class)
            #     with open(r'C:\Users\32685\Desktop\spirecomm\results.txt', 'a') as file:
            #         if result:
            #             file.write(f"win as {chosen_class} at {datetime.now()}\n")
            #         else:
            #             file.write(f"lose as {chosen_class} at {datetime.now()} at floor {agent.game.floor}\n")
            while True:
                agent.change_class(PlayerClass.DEFECT)
                result = coordinator.play_one_game(PlayerClass.DEFECT)
            with open(r'C:\Users\32685\Desktop\spirecomm\results.txt', 'a') as file:
                if result:
                    file.write(f"win as {PlayerClass.DEFECT} at {datetime.now()}\n")
                else:
                    file.write(f"lose as {PlayerClass.DEFECT} at {datetime.now()} at floor {agent.game.floor}\n")
        except Exception as e:
            # 将错误信息记录到文件
            logging.error("An error occurred: %s\n\n\n\n\n", str(e), exc_info=True)
