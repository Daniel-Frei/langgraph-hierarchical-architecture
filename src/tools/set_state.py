# src\tools\set_state.py
from __future__ import annotations

from typing import Annotated, get_type_hints
from pydantic import ValidationError

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from state.main_state import SharedState

from src.logger.logger import getLogger

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper – validate the write against the global SupervisorState schema
# ---------------------------------------------------------------------------
def _validated_kv(key: str, value: str) -> dict:
    """
    Return `{key: value}` if the pair is compatible with `SupervisorState`,
    otherwise raise `ValidationError`.
    """
    SharedState.model_validate({key: value})
    return {key: value}


# def _set_state_impl(
#     *,
#     key: str,
#     value: str,
#     tool_call_id: str,
#     msg_key_in_state: Optional[str] = None,
# ) -> Command:
#     msg_key = msg_key_in_state or "messages"
#     logger.info("[set_state] key=%s value=%s (overwrite)", key, value)
#     return Command(
#         update={
#             key: value,
#             msg_key: [
#                 ToolMessage(
#                     content=f"{key.capitalize()} set to {value} [overwriting previous value if any]",
#                     name="set_state",
#                     tool_call_id=tool_call_id,
#                 )
#             ],
#         }
#     )


def make_set_state(msg_key: str, name: str):
    """
    Parameters
    ----------
    msg_key : str
        Where the resulting ToolMessage(s) should be appended in the *caller’s*
        local state (e.g. "messagesInitQuestions").  Falls back to "messages".
    name : str
        Public name exposed to the LLM.

    The returned tool **never** mutates the live state object directly – it
    only returns a `Command(update=…)` describing the desired change.
    """

    @tool(
        name,
        description=(
            """Write a value in the shared state.

            Arguments
            ---------
            key : str
                The exact field to set (e.g. ``"color"`` or ``"speed"``).
            value : str
                The value to store. **If the key already exists, its previous
                content is replaced by this new value.**

            Returns
            -------
            • On success: confirmation message.
            • On schema error: an *ERROR* message so the agent can retry.

            Notes
            -----
            • **Destructive** – any prior value at ``state[key]`` is lost.
            • Be aware of the current state value before using the tool and overwriting the current value to avoid data loss.
            """
        ),
    )
    def _set_state(
        key: str,
        value: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        # Where to store the confirmation/error message
        target_msg_key = msg_key or "messages"

        if key not in get_type_hints(SharedState):
            logger.error("[set_state] Unknown state field: %s", key)
            return Command(
                update={
                    target_msg_key: [
                        ToolMessage(
                            content=(
                                f"ERROR: '{key}' is not a valid field in the state."
                            ),
                            name="set_state",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        try:
            update_dict = _validated_kv(key, value)
            logger.info(
                "[set_state] SUCCESS  %s ← %s  (msg_key=%s)",
                key,
                value,
                target_msg_key,
            )

            # Success – confirmation message
            update_dict[target_msg_key] = [
                ToolMessage(
                    content=f"{key} updated.",
                    name="set_state",
                    tool_call_id=tool_call_id,
                )
            ]
            cmd = Command(update=update_dict)
            logger.info("[set_state] Command returned: %s", cmd)
            return cmd

        except ValidationError as err:
            # Failure – feed the error back to the LLM
            logger.warning("[set_state] schema error: %s", err)
            return Command(
                update={
                    target_msg_key: [
                        ToolMessage(
                            content=f"ERROR: {err.errors()[0]['msg']}",
                            name="set_state",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

    return _set_state