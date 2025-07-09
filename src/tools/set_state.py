# src\tools\set_state.py
from __future__ import annotations

from typing import Annotated, Type, get_type_hints
from pydantic import BaseModel, ValidationError

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from src.logger.logger import getLogger

logger = getLogger(__name__)


def make_set_state(
    msg_key: str | None = None,
    name: str | None = None,
    state_schema: Type[BaseModel] | None = None,
):
    """
    Parameters
    ----------
    msg_key : str, optional
        Where the resulting ToolMessage(s) should be appended (e.g. "messagesInitQuestions"). Defaults to "messages".
    name : str, optional
        Public name exposed to the LLM. Defaults to **"set_state"**.
    state_schema : pydantic.BaseModel **required**
        Schema used for validation.  **Must** be provided; absence raises
        during build so runtime never crashes.
    """

    def _validated_kv(key: str, value: str) -> dict[str, str]:
        state_schema.model_validate({key: value})
        return {key: value}

    # ---- build-time guard --------------------------------------------------
    if state_schema is None:
        logger.critical("[set_state] No state_schema provided – aborting build.")
        raise ValueError(
            "make_set_state(...) requires a `state_schema` argument."
        )

    tool_name      = name or "set_state"
    target_msg_key = msg_key or "messages"

    @tool(
        tool_name,
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

        # ---- runtime validation -------------------------------------------
        if not isinstance(key, str):
            logger.error("[set_state] key must be str, got %s", type(key))
            return Command(
                update={
                    target_msg_key: [
                        ToolMessage(
                            content="ERROR: ‘key’ must be a string.",
                            name="set_state",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        if not isinstance(value, str):
            logger.error("[set_state] value must be str, got %s", type(value))
            return Command(
                update={
                    target_msg_key: [
                        ToolMessage(
                            content="ERROR: ‘value’ must be a string.",
                            name="set_state",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        if key not in get_type_hints(state_schema):
            logger.error("[set_state] Unknown state field: %s", key)
            return Command(
                update={
                    target_msg_key: [
                        ToolMessage(
                            content=(
                                f"ERROR: '{key}' is not a valid field in the state."
                            ),
                             name=tool_name,
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
                    name=tool_name,
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
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

    return _set_state