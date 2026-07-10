from langgraph import StateGraph
from typing import TypedDict
from Pydantic import BaseModel , Field


class evaluation_schema(BaseModel):
    feedback : str = Field(description = "")
    