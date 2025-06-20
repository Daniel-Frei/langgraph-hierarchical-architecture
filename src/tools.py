# src/tools.py
import logging
from typing import Optional, Annotated

from langchain_core.messages import ToolMessage, AIMessage
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


def _set_state_impl(
    *,
    key: str,
    value: str,
    tool_call_id: str,
    msg_key_in_state: Optional[str] = None,
) -> Command:
    msg_key = msg_key_in_state or "messages"
    return Command(
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


def make_set_state(msg_key: str, name: str):
    @tool(
        name,
        description=(
            "Write a value into shared state.\n"
            "Arguments:\n  key   – which field to set ('color' or 'speed')\n"
            "  value – the word to store"
            "  msg_key_in_state  – (optional) the state‐field to append messages to (defaults to 'messages')"
        ),
    )
    def _wrapper(
        key: str,
        value: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        return _set_state_impl(
            key=key,
            value=value,
            tool_call_id=tool_call_id,
            msg_key_in_state=msg_key,
        )

    return _wrapper

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


def _ask_user_impl(
    prompt: str,
    tool_call_id: str,
    msg_key_in_state: str,
    state: SharedState | None = None,
) -> Command:
    msg_key = msg_key_in_state or "messages"
    if state is not None:
        state.setdefault(msg_key, []).append(
            AIMessage(content=prompt, name="assistant")
        )
    user_reply = interrupt(prompt)
    return Command(
        update={
            msg_key: [
                ToolMessage(
                    name="ask_user",
                    tool_call_id=tool_call_id,
                    content=user_reply,
                )
            ]
        }
    )


def make_ask_user(msg_key: str, name: str):
    @tool(name, description=f"Ask the user; replies are saved in '{msg_key}'.")
    def _wrapper(
        prompt: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        return _ask_user_impl(prompt, tool_call_id, msg_key)
    return _wrapper


# @tool(
#     "ask_user",
#     description=(
#         "Ask the end-user a question and pause execution until they reply. "
#         "Arguments:\n  prompt – the question to show the user"
#     ),
# )
# def ask_user(
#     prompt: str,
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     msg_key_in_state: Optional[str] = None,
#     state: Annotated[SharedState, InjectedState] | None = None,
# ) -> Command:
#     """
#     Pause the graph, surface `prompt` to the front-end, and resume with
#     the user’s answer.  The answer is returned to the LLM in a ToolMessage,
#     so the LLM can decide what to do next (e.g. call `set_state`).
#     """
#     # 1) Show the prompt to the user as a normal assistant message
#     msg_key = msg_key_in_state or "messages"
#     if state is not None:
#         state.setdefault(msg_key, []).append(
#             AIMessage(content=prompt, name="assistant")
#         )

#     # 2) Stop and wait for the reply (no text in the modal itself)
#     user_reply = interrupt(prompt)

#     msg_key = msg_key_in_state or "messages"
#     logging.debug("[ask_user]  msg_key=%r", msg_key)

#     return Command(
#         update={
#             msg_key: [
#                  ToolMessage(
#                      name="ask_user",
#                      tool_call_id=tool_call_id,
#                      content=user_reply,
#                  )
#              ]
#          }
#      )