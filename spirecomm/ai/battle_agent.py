from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
import os
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

os.environ["OPENAI_API_KEY"] = "sk-Nxr5VkCGRNruaDUzUZz3uCkKUtMvg0u3V7uiXJhJSbo0wAIp"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    floor: int
    turn: int
    current_hp:int
    max_hp:int
    block:int
    energy:int
    relics:str
    hand:str
    monsters:str
    drawPile:str
    discardPile:str
    powers:str
    orbs:str


class BattleAgent:
    def __init__(self,role="DEFECT",llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)):
        self.role= role
        self.llm = llm

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

        graph_builder.add_edge(START, "LLM")
        graph_builder.add_conditional_edges("LLM", self.router1, ["Tool", END])
        graph_builder.add_edge("Tool", "LLM")

        self.graph = graph_builder.compile()



    def llm_1(self,state: State):

        outputFormat = self.battle_output_parser.get_format_instructions()
        system_msg = f"""
                            You are an AI designed to play *Slay the Spire* as the role {self.role} and make optimal card choices during combat. 
                            On each turn, you need to choose one card to play or decide to end the turn. 
                            Given the following context, make your choice for this specific call.

                            Context:
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

                            Goal:
                            take a action to maximize your chances of winning the combat:
                                Choose one card from Hand pile to play, or decide to end the turn.              

                            General Guidelines:
                            - before you choose a card, please figure out your combat strategy first based on the enemy you encounter.
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
                            - exhausting the status card is good for you, do it when you can.

                            Attention:
                            - Before giving your response,please check the chosen card's attribute 'is_card_has_target', 
                              if it's True,then you need to appoint a target for the card, which means 'targetIndex' 
                              in your response shouldn't be -1.
                            - Provide reasoning for your card choice based on the current context.

                            Response format:
                            {outputFormat}
                            """

        messages = [{"role": "system", "content": system_msg}] + state["messages"]
        return {"messages": [self.llm.invoke(messages)]}


    def router1(self,state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "Tool"
        else:
            return END


    def invoke(self,floor:int,turn: int,
    current_hp:int,
    max_hp:int,
    block:int,
    energy:int,
    relics:str,
    hand:str,
    monsters:str,
    drawPile:str,
    discardPile:str,
    powers:str,
    orbs:str):
        template_string = """ You are currently in a combat, and the below is the context:
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

                            now take your action and give the response.
                        """
        template1 = ChatPromptTemplate.from_template(template_string)
        messages = template1.format_messages(
            floor=floor,
            turn=turn,
            hp=f"{current_hp}/{max_hp}",
            block=block,
            energy=energy,
            relics=relics,
            hand=hand,
            monsters=monsters,
            drawPile=drawPile,
            discardPile=discardPile,
            pStatus=powers,
            # output_format=outputFormat,
            orbs=orbs
        )
        state = State(messages=messages,turn = turn,current_hp=current_hp,max_hp=max_hp,
                      block=block,energy=energy,relics=relics,hand=hand,
                      monsters=monsters,drawPile=drawPile,discardPile=discardPile,
                      powers=powers,orbs=orbs)
        result = self.graph.invoke(state)

        return result

if __name__ =="__main__":

    agent = BattleAgent()
    responses = agent.invoke(
        floor=1,
        turn=1,
        current_hp=75,
        max_hp=75,
        block=0,
        energy=3,
        relics='[ Cracked Core ]',
        hand='[ Defend(1,False), Zap(1,False), Strike(1,True), Dualcast(1,False), Strike(1,True) ]',
        monsters='[ JawWorm( 42/42 ,Intent.DEBUG,0,[  ]) ]',
        drawPile='[ Strike(1,True), Defend(1,False), Defend(1,False), Defend(1,False) ]',
        discardPile='[]',
        powers='[]',
        orbs='[ Lightning, Orb Slot, Orb Slot ]',
    )
    for response in responses["messages"]:
        print(type(response).__name__+" "+response.content.__str__())
        print("\n\n")

