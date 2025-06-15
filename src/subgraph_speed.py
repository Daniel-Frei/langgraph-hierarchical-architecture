# src\subgraph_speed.py
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from state import SharedState
from tools import set_state, ask_user, get_state

logging.getLogger(__name__).setLevel(logging.DEBUG)

_SYSTEM_PROMPT = (
    "You are a car-speed expert.\n"
    "First call the `get_state` tool with {\"key\": \"speed\"} to see if a "
    "speed adjective is already stored.\n"
    "• If the returned value is non-empty, your job is done \n"
    "`set_state` with that same value and finish."
    "• otherwise, call the `ask_user` tool to ask the user: "
    "\"What colour should the car be?\".\n"
    "After the user replies, call `set_state` with:\n"
    "  {\"key\": \"color\", \"value\": \"<their answer>\", "
    "\"msg_key_in_state\": \"messagesColor\"}"
)


def ask_for_speed(state: SharedState):
    """LLM node that asks the speed specialist to pick a word and call the tool."""
    logging.debug("[speed_agent.ask_for_speed] entry state: %r", state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([set_state, ask_user, get_state])
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state.get("messagesSpeed", [])

    ai: AIMessage = llm.invoke(messages)
    ai.name = "speed_agent"
    logging.debug("[speed_agent.ask_for_speed] LLM returned: %r", ai)
    return {"messagesSpeed": [ai]}


def return_msg(state: SharedState):
    """Once the adjective has been stored, inform the supervisor thread."""
    if not state.get("speed"):
        # Still waiting for a value – keep the loop alive silently.
        return {}

    logging.debug("[speed_agent.return_msg] Adding message for supervisor")
    public = AIMessage(content="speed_agent has chosen the speed: " + state.get("speed"), name="speed_agent")
    return {
        "messages": [public],          # supervisor thread
        "messagesSpeed": [public],
    }

def make_tools_router(messages_key: str = "messages"):
    def _router(state):
        return tools_condition(state, messages_key=messages_key)
    return _router


# ── build the mini‑graph ─────────────────────────────────────────
builder = StateGraph(SharedState)
builder.add_node("llm", ask_for_speed)
builder.add_node("tools", ToolNode([get_state, set_state, ask_user], messages_key="messagesSpeed"))
builder.add_node("returnMsg", return_msg)

builder.add_edge(START, "llm")
builder.add_edge("tools", "llm")
builder.add_conditional_edges("llm",
    make_tools_router("messagesColor"),
    {"tools": "tools", END: "returnMsg"},
)
builder.add_edge("returnMsg", END)

speed_agent = builder.compile(name="speed_agent")
