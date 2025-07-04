# src\state\main_state.py

"""
Pydantic-based graph-state schema.

Switching from a TypedDict to a Pydantic model gives us
• run-time validation of inputs coming into each node
• IDE auto-completion with attribute access (dot syntax)
• stricter “extra = forbid” behaviour to surface typos early
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import AnyMessage                # AnyMessage ⇢ serialises cleanly
from langgraph.graph.message import add_messages


class SharedState(BaseModel):
    """Global state shared by the supervisor and both sub-graphs."""
    model_config = ConfigDict(extra="forbid")                  # reject unknown keys

    # ── conversation buffers ───────────────────────────────────────────────
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    messagesColor: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    messagesSpeed: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

    # ── working memory ─────────────────────────────────────────────────────
    halfSentence: Optional[str] = None
    color:        Optional[str] = None
    speed:        Optional[str] = None
    fullSentence: Optional[str] = None

    # ── ReAct bookkeeping ──────────────────────────────────────────────────
    remaining_steps: int = 0
