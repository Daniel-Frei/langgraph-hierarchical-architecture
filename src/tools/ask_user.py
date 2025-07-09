# graph\src\tools\ask_user.py
from typing import Annotated, Any

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt

from src.logger.logger import getLogger

logger = getLogger(__name__)

def make_ask_user(
    msg_key: str | None = None,
    name: str | None = None,
):
    """
    Create a pause-and-ask tool.

    Parameters
    ----------
    msg_key : str, optional
        State attribute that stores the conversation thread
        (e.g. ``"messagesColor"``).
    name : str, optional
        Public name shown to the LLM.

    """
    tool_name = name or "ask_user"
    @tool(
        tool_name,
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
    def _ask_user_impl(
        prompt: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Any | None = None,          # injected automatically
    ) -> Command:
        # 1️ Guard/validate
        if not isinstance(prompt, str):
            logger.error("[ask_user] prompt must be str, got %s", type(prompt))
            return Command(
                update={
                    (msg_key or "messages"): [
                        ToolMessage(
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            content="ERROR: prompt argument must be a string.",
                        )
                    ]
                }
            )

        # 2️ Figure out which list to append to
        actual_msg_key = msg_key or "messages"

        # 2️  Store the assistant prompt in the thread (if a state object exists)
        if state is not None:
            thread = getattr(state, actual_msg_key, None)
            if thread is None:
                setattr(state, actual_msg_key, [])
                thread = getattr(state, actual_msg_key)
            thread.append(AIMessage(content=prompt, name="assistant"))

        logger.info("[ask_user] prompt=%r  msg_key=%s", prompt, actual_msg_key)
        user_reply = interrupt(prompt)

        # 3️  Return a ToolMessage
        return Command(
            update={
                actual_msg_key: [
                    ToolMessage(
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        content=user_reply,
                    )
                ]
            }
        )

    return _ask_user_impl