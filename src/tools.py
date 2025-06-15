# src/tools.py
import logging
from typing import Optional
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt
from langgraph.prebuilt     import InjectedState

from state import SharedState

logging.getLogger(__name__).setLevel(logging.DEBUG)


@tool(
    "get_state",
    description=(
        "Read a value from shared state (short-term memory).\n"
        "Arguments:\n  key – which field to read ('color', 'speed', …)\n"
        "Returns the stored value or an empty string if the key is missing."
    ),
)
def get_state(
    key: str,
    state: Annotated[SharedState, InjectedState],        # ← magic injection
) -> str:
    """
    Read-only helper: expose the current value of ``state[key]`` to the LLM.
    Does NOT modify the graph state.
    """
    value = state.get(key, "")
    logging.debug("[get_state] key=%r  value=%r", key, value)
    return value

@tool(
    "set_state",
    description=(
        "Write a value into shared state.\n"
        "Arguments:\n  key   – which field to set ('color' or 'speed')\n"
        "  value – the word to store"
        "  msg_key_in_state  – (optional) the state‐field to append messages to (defaults to 'messages')"
    ),
)
def set_state(
    key: str,
    value: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    msg_key_in_state: Optional[str] = None,
) -> Command:
    """
    Generic setter: update ``state[key] = value`` (key is normally
    'color' or 'speed') and add a ToolMessage for traceability.
    """
    # use the passed‐in msg_key_in_state, or fall back to "messages"
    msg_key = msg_key_in_state or "messages"
    logging.debug(
        "[set_state] called with key=%r  value=%r  id=%r  msg_key=%r  msg_key_in_state=%r",
        key,
        value,
        tool_call_id,
        msg_key,
        msg_key_in_state,
    )
    cmd = Command(
        update={
            key: value,
            msg_key: [
                ToolMessage(
                    content=f"{key.capitalize()} set to {value}",
                    name="set_state",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
    logging.debug("[set_state] returning Command.update: %r", cmd.update)
    return cmd

@tool(
    "ask_user",
    description=(
        "Ask the end-user a question and pause execution until they reply. "
        "Arguments:\n  prompt – the question to show the user"
    ),
)
def ask_user(
    prompt: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Pause the graph, surface `prompt` to the front-end, and resume with
    the user’s answer.  The answer is returned to the LLM in a ToolMessage,
    so the LLM can decide what to do next (e.g. call `set_state`).
    """
    user_reply = interrupt(prompt)                 # ← blocks until resume
    return Command(                                #  update private thread
        update={
            "messagesColor": [
                ToolMessage(
                    name="ask_user",
                    tool_call_id=tool_call_id,
                    content=user_reply,
                )
            ]
        }
    )