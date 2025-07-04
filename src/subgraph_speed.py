# src\subgraph_speed.py
import logging
import pprint
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from state.main_state import SharedState
from tools import make_set_state, make_ask_user, get_state

logging.getLogger(__name__).setLevel(logging.DEBUG)

_SYSTEM_PROMPT = (
    "You are a car-speed expert.\n"
    "First call the `get_state` tool with {\"key\": \"speed\"} to see if a "
    "speed adjective is already stored.\n"
    "• If the returned value is non-empty, your job is done \n"
    "• otherwise, call the `ask_user_speed` tool to ask the user: "
    "\"What speed should the car be?\".\n"
    "After the user replies, call `set_speed_state` with:\n"
    "  {\"key\": \"speed\", \"value\": \"<their answer>\"}"
)

ask_user_speed = make_ask_user("messagesSpeed", "ask_user_speed")
set_speed_state = make_set_state("messagesSpeed", "set_speed_state")

def ask_for_speed(state: SharedState):
    """LLM node that asks the speed specialist to pick a word and call the tool."""
    logging.debug("[speed_agent.ask_for_speed] entry state: %r", state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([set_speed_state, ask_user_speed, get_state])
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state.messagesSpeed

    ai: AIMessage = llm.invoke(messages)
    ai.name = "speed_agent"
    logging.debug("[speed_agent.ask_for_speed] LLM returned: %r", ai)
    return {"messagesSpeed": [ai]}


def return_msg(state: SharedState):
    """Once the adjective has been stored, inform the supervisor thread."""
    if not state.speed:
        # Still waiting for a value – keep the loop alive silently.
        return {}

    logging.debug("[speed_agent.return_msg] Adding message for supervisor")
    public = AIMessage(content="speed_agent has chosen the speed: " + state.speed, name="speed_agent")
    return {
        "messages": [public],          # supervisor thread
        "messagesSpeed": [public],
    }

def make_tools_router(messages_key: str = "messages"):
    def _router(state: SharedState):
        msgs = getattr(state, messages_key)

        last = msgs[-1] if msgs else None
        logging.debug(
            "[router:%s] last type=%s has_attr.tool_calls=%s content=%s",
            messages_key,
            type(last).__name__ if last else None,
            hasattr(last, "tool_calls"),
            pprint.pformat(getattr(last, "tool_calls", None)),
        )

        branch = tools_condition({ "__dummy__": True, messages_key: msgs },
                                 messages_key = messages_key)
        logging.debug("[router:%s] tools_condition → %s", messages_key, branch)
        return branch
    return _router


# ── build the mini‑graph ─────────────────────────────────────────
builder = StateGraph(SharedState)
builder.add_node("llm", ask_for_speed)
builder.add_node("tools", ToolNode([get_state, set_speed_state, ask_user_speed], messages_key="messagesSpeed"))
builder.add_node("returnMsg", return_msg)

builder.add_edge(START, "llm")
builder.add_edge("tools", "llm")
builder.add_conditional_edges("llm",
    make_tools_router("messagesSpeed"),
    {"tools": "tools", END: "returnMsg"},
)
builder.add_edge("returnMsg", END)

speed_agent = builder.compile(name="speed_agent")
