# src\graph.py
from src.logger import getLogger
from dotenv import load_dotenv

from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from state import SharedState
from subgraph_color import color_agent
from subgraph_speed import speed_agent

logger = getLogger(__name__)
load_dotenv()

# 1️⃣  Supervisor with two workers
supervisor = create_supervisor(
    agents=[color_agent, speed_agent],
    model=ChatOpenAI(model="gpt-4o-mini"),
    prompt=(
        "You manage two specialists:\n"
        "• color_agent – knows the car’s colour\n"
        "• speed_agent – knows the car’s speed\n\n"
        "No matter what the user says, delegate the task, first delegate with "
        "`transfer_to_color_agent`, wait, then delegate with "
        "`transfer_to_speed_agent`, wait, and finally summarise."
        "Your goal is to obtain the color and the speed from the specialists and then combine them."
        "Don't directly answer the user other than the final summary. Instead, use `transfer_to_color_agent` and `transfer_to_speed_agent` until you have both information."
    ),
    include_agent_name="inline",
    add_handoff_back_messages=True,
    state_schema=SharedState,
).compile(name="supervisor")

# 2️⃣  Node functions
def ensure_defaults(state: SharedState):
    return {
        "halfSentence": "The car is ",
        "color": "",
        "speed": "slow",
        "fullSentence": "",
        "remaining_steps": 15,
    }
def assemble(state: SharedState):
    logger.debug(f"[assemble] entry state: {state!r}")
    color = state.get("color", "").strip()
    speed = state.get("speed", "").strip()

    if not color:
        logger.error("[assemble] missing color in state, raising")
        raise ValueError("assemble(): ‘color’ must be non-empty")
    if not speed:
        logger.error("[assemble] missing speed in state, raising")
        raise ValueError("assemble(): ‘speed’ must be non-empty")

    sentence = f"{state['halfSentence']}{color} and {speed}"
    logger.debug(f"[assemble] built sentence: {sentence!r}")
    return {
        "fullSentence": sentence,
        "messages": state["messages"]
        + [SystemMessage(content=f"combined into '{sentence}'")],
    }

# 3️⃣  Parent graph
parent = StateGraph(SharedState)
parent.add_node("init", ensure_defaults)
parent.add_node("delegate", supervisor)
parent.add_node("assemble", assemble)

parent.add_edge(START, "init")
parent.add_edge("init", "delegate")
parent.add_edge("delegate", "assemble")
graph = parent.compile(name="parent_graph")

# 4️⃣  Demo run
if __name__ == "__main__":
    init = {
        "messages": [{"role": "user", "content": "Describe the car."}],
        "halfSentence": "The car is ",
        "color": "",
        "speed": "",
        "fullSentence": "",
        "remaining_steps": 5,
    }
    logger.info(f"Starting graph.stream with init: {init!r}")
    result = graph.invoke(init)          # ← run once
    print("FINAL STATE →", result)
