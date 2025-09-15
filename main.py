import itertools
import multiprocessing
from datetime import datetime
import logging
import sys

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.agent import SimpleAgent
from spirecomm.spire.character import PlayerClass
import tkinter as tk

is_to_use_gui = False #是否启用GUI
if is_to_use_gui:
    root = tk.Tk()
else:
    root = None
def run_gui():
    root.mainloop()

if __name__ == "__main__":
    # 配置 logging
    logging.basicConfig(
        filename=r'C:\Users\32685\Desktop\spirecomm\output\error_log.txt',  # 输出到的文件名
        level=logging.ERROR,  # 记录的最低级别
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )

    try:
        # 可能引发异常的代码
        with open(r'C:\Users\32685\Desktop\spirecomm\output\output.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\mapInfo.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\battle_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\choose_card_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\action_list.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\simple_grid_choice_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\event_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\hand_select_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\shop_select_agent.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\output\battle_agent_gui.txt', 'w') as file:
            file.write('--------------tracking---------------\n')
        with open(r'C:\Users\32685\Desktop\spirecomm\dataset\dataset_buffer_act1.jsonl','w') as file:
            pass
        with open(r'C:\Users\32685\Desktop\spirecomm\dataset\dataset_buffer_act2.jsonl','w') as file:
            pass


        agent = SimpleAgent(is_to_use_gui=is_to_use_gui,root=root)
        coordinator = Coordinator(is_to_use_gui=is_to_use_gui)
        coordinator.signal_ready()

        coordinator.register_command_error_callback(agent.handle_error)
        coordinator.register_state_change_callback(agent.get_next_action_in_game_new)
        coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

        # Play games forever, cycling through the various classes
        for chosen_class in itertools.cycle(PlayerClass):
            agent.change_class(chosen_class)
            agent.init_llm_env()
            if is_to_use_gui:
                gui_process = multiprocessing.Process(target=run_gui)
                gui_process.start()

            seed = "2ZK5PHFXAGB3X" # 重锤开，二层圆顶
            seed = "3FR420LZN9M7H" # 35层，力量战
            seed = "16G2XGIZWIVPY" # 44层，鸡煲
            seed = "16QXPYKRH7U5W" # 50层，毒贼
            seed = "2IEMKEY2CBQAZ" # 33层，鸡煲，鸟居钛合金棒
            seed = "55DIXCQA169G8" # 33层，战士自残流, 有肉
            seed = "1ASP5QUI90TT8" # 45层，毒贼，催化剂，双瓶中基米
            seed = "IJGDFL933EQJ" # 肉
            seed = "3UUE1ZMQ7E2T" # 45层，战士
            seed = "1B2WCU633TTY6" # 换4农合开
            seed = "15U1RLGLSPKXT" # 战士农种
            seed = "LSIYNLGZDQJU" # 肉
            seed = "2336LW8KMWIIT" #白嫖精英
            seed = "576PMLJ8RWVQV" #金冠开
            seed = "3I0F0ZCZD9KUC" #50层，肉
            seed = "LPMPVD2QKA4C" #50层，农局
            seed = "4N56AHZCZ1YMZ" #50层，硫磺战
            seed = "39976S0CU4QPB" #50层，战士

            result = coordinator.play_one_game(chosen_class)
            with open(r'C:\Users\32685\Desktop\spirecomm\output\results.txt', 'a') as file:
                if result:
                    file.write(f"win as {chosen_class} at {datetime.now()}\n")
                else:
                    file.write(f"lose as {chosen_class} at {datetime.now()} at floor {agent.game.floor}\n")


        while True:
            p_class = PlayerClass.DEFECT
            agent.change_class(p_class)
            agent.init_llm_env()
            if is_to_use_gui:
                gui_process = multiprocessing.Process(target=run_gui)
                gui_process.start()


            result = coordinator.play_one_game(p_class)
            with open(r'C:\Users\32685\Desktop\spirecomm\output\results.txt', 'a') as file:
                if result:
                    file.write(f"win as {p_class} at {datetime.now()}\n")
                else:
                    file.write(f"lose as {p_class} at {datetime.now()} at floor {agent.game.floor}\n")
    except Exception as e:
        # 将错误信息记录到文件
        logging.error("An error occurred: %s\n\n\n\n\n", str(e), exc_info=True)

