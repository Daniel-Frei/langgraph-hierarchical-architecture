# src/state.py
from typing import List
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SharedState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    halfSentence: str
    color: str
    speed: str
    fullSentence: str
    remaining_steps: int
    # subgraphs state
    messagesColor: Annotated[List[BaseMessage], add_messages]
    messagesSpeed: Annotated[List[BaseMessage], add_messages]