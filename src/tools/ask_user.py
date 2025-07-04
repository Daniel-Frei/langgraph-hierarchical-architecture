# graph\src\tools\ask_user.py
from typing import Annotated

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt

from state.main_state import SharedState
from src.logger.logger import getLogger

logger = getLogger(__name__)

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
    logger.info("[ask_user] prompt=%r  msg_key=%s", prompt, msg_key)
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
    @tool(
        name,
        description=(
        """Ask the human user a specific question and return their response.

        Use this tool whenever you need to collect new information from the user.
        It sends a prompt string to the user and pauses execution until the user replies.

        Args:
            prompt: A clear, concise question to solicit user input about one or more schema fields.

        Returns:
            The user's answer as raw text.
        """
        ),
    )
    def _wrapper(
        prompt: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        return _ask_user_impl(prompt, tool_call_id, msg_key)
    return _wrapper