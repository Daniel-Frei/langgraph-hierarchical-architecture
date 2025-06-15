# src/subgraph_color.py
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from state import SharedState
from tools import set_state, ask_user

logging.getLogger(__name__).setLevel(logging.DEBUG)

_SYSTEM_PROMPT = (
    "You are a car-colour information collector.\n"
    "First, call the `ask_user` tool to ask: "
    "\"What colour should the car be? (single-word paint colour)\".\n"
    "After the user replies, call `set_state` with:\n"
    "  {\"key\": \"color\", \"value\": \"<their answer>\", "
    "\"msg_key_in_state\": \"messagesColor\"}"
)

def ask_for_colour(state: SharedState):
    logging.debug("[color_agent.ask_for_colour] entry state: %r", state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([set_state, ask_user])
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state.get("messagesColor", [])
    ai: AIMessage = llm.invoke(messages)
    ai.name = "color_agent"
    logging.debug("[color_agent.ask_for_colour] LLM returned: %r", ai)
    return {"messagesColor": [ai]}

def return_msg(state: SharedState):
    if not state.get("color"):
        # nothing chosen yet → keep the agent loop alive
        return {}
    logging.debug("[color_agent.return_msg] Adding message for supervisor")
    public = AIMessage(content="color_agent has chosen the colour: " + state.get("color"),
                       name="color_agent")
    return {
        "messages": [public],                              # supervisor thread
        "messagesColor":[public],
    }


def check_state(state: SharedState):
    logging.debug("[color_agent.check_state] color state: %r", state.get("color"))
    return not bool(state.get("color"))

def make_tools_router(messages_key: str = "messages"):
    def _router(state):
        return tools_condition(state, messages_key=messages_key)
    return _router


# ── build the mini-graph ─────────────────────────────────────────
builder = StateGraph(SharedState)
builder.add_node("llm", ask_for_colour)
builder.add_node("tools", ToolNode([set_state, ask_user], messages_key="messagesColor"),
)
builder.add_node("returnMsg", return_msg)

builder.add_edge(START, "llm")
builder.add_edge("tools", "llm")
builder.add_conditional_edges("llm",
    make_tools_router("messagesColor"),
    {"tools": "tools", END: "returnMsg"},
)
builder.add_edge("returnMsg", END)

color_agent = builder.compile(name="color_agent")