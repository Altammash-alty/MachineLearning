from langgraph import StateGraph
from typing import TypedDict
from Pydantic import BaseModel , Field


class conversation_history(BaseModel):
    messsages : Annotated[list[BaseMessage]]
    