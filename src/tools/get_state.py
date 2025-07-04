# graph\src\tools\get_state.py
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt     import InjectedState
from state.main_state import SharedState
from src.logger.logger import getLogger

logger = getLogger(__name__)

@tool(
    "get_state",
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
def get_state(
    key: str,
    state: Annotated[SharedState, InjectedState],        # ← magic injection
) -> str:
    """
    Read-only helper: expose the current value of ``state[key]`` to the LLM.
    Does NOT modify the graph state.
    """
    value = getattr(state, key, "")
    logger.info("[get_state] key=%r  value=%r", key, value)
    return value