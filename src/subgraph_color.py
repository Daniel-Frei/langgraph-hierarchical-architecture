# src/subgraph_color.py
import logging
import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from state.main_state import SharedState
from tools import make_set_state, make_ask_user, make_get_state

logging.getLogger(__name__).setLevel(logging.DEBUG)

_SYSTEM_PROMPT = (
    "You are a car-colour information collector.\n"
    "First call the `get_state` tool with {\"key\": \"color\"} to see if a "
    "color adjective is already stored.\n"
    "• If the returned value is non-empty, your job is done \n"
    "• otherwise, call the `ask_user_color` tool to ask the user: "
    "\"What colour should the car be?\".\n"
    "After the user replies, call `set_state_color` with:\n"
    "  {\"key\": \"color\", \"value\": \"<their answer>\"}"
)

ask_user_color = make_ask_user("messagesColor")
set_state_color = make_set_state("messagesColor", state_schema=SharedState)
get_state_color = make_get_state(state_schema=SharedState)


def ask_for_colour(state: SharedState):
    logging.debug("[color_agent.ask_for_colour] entry state: %r", state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([set_state_color, ask_user_color, get_state_color])
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state.messagesColor
    ai: AIMessage = llm.invoke(messages)
    ai.name = "color_agent"
    logging.debug("[color_agent.ask_for_colour] LLM returned: %r", ai)
    return {"messagesColor": [ai]}

def return_msg(state: SharedState):
    if not state.color:
        # nothing chosen yet → keep the agent loop alive
        return {}
    logging.debug("[color_agent.return_msg] Adding message for supervisor")
    public = AIMessage(content="color_agent has chosen the colour: " + state.color,
                        name="color_agent")
    return {
        "messages": [public],                              # supervisor thread
        "messagesColor":[public],
    }


def check_state(state: SharedState):
    logging.debug("[color_agent.check_state] color state: %r", state.color)
    return not bool(state.color)

def make_tools_router(messages_key: str = "messages"):
    def _router(state: SharedState):
        # --- NEW DIAGNOSTICS ------------------------------------
        msgs = getattr(state, messages_key)
        last = msgs[-1] if msgs else None
        logging.debug(
            "[router:%s] last type=%s has_attr.tool_calls=%s content=%s",
            messages_key,
            type(last).__name__ if last else None,
            hasattr(last, "tool_calls"),
            pprint.pformat(last.tool_calls if hasattr(last, "tool_calls") else None),
        )
        branch = tools_condition({"__dummy__": True, messages_key: msgs},   # KEEP objects intact
                                 messages_key=messages_key)
        logging.debug("[router:%s] tools_condition → %s", messages_key, branch)
        return branch
    return _router


# ── build the mini-graph ─────────────────────────────────────────
builder = StateGraph(SharedState)
builder.add_node("llm", ask_for_colour)
builder.add_node("tools", ToolNode([set_state_color, ask_user_color, get_state_color], messages_key="messagesColor"),
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