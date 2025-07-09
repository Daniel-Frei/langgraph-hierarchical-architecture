# graph\src\tools\get_state.py
from typing import Annotated, Any, Type
from pydantic import BaseModel

from langchain_core.tools import tool
from langgraph.prebuilt     import InjectedState
from src.logger.logger import getLogger

logger = getLogger(__name__)

def make_get_state(
    state_schema: Type[BaseModel] | None = None,
    name: str | None = None,
):
    """
    Factory that returns a read-only *get_state* tool bound to `state_schema`.

    Parameters
    ----------
    state_schema : pydantic.BaseModel **required**
        Schema used for validation / field existence checks.
    name : str, optional
        Public name of the tool. Defaults to **"get_state"**.
    """

    # ---- build-time guard --------------------------------------------------
    if state_schema is None:
        logger.critical(
            "[get_state] No state_schema provided – aborting build."
        )
        raise ValueError(
            "make_get_state(...) requires a `state_schema` argument."
        )

    tool_name = name or "get_state"

    @tool(
        tool_name,
        description=(
            """Read any value from the state (short-term memory).

            Arguments
            ---------
            key : str
                The  entry you want to inspect
                (e.g. ``"initial_questions"`` or ``"StudyObjectives"``).

            Returns
            -------
            str  – the stored value, or an empty string when the key is absent.

            Notes
            -----
            • **Read-only** – this tool never mutates state.
            """
        ),
    )
    def _get_state(
        key: str,
        state: Annotated[Any, InjectedState],    # concrete class injected at run time
    ) -> str:
        # ---- runtime validation -------------------------------------------
        if not isinstance(key, str):
            logger.error("[get_state] key must be str, got %s", type(key))
            return "ERROR: ‘key’ argument must be a string."

        value = getattr(state, key, "")
        logger.info("[get_state] key=%r  value=%r", key, value)
        return value

    return _get_state