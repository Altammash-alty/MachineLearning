from langgraph import StateGraph
from typing import TypedDict
from Pydantic import BaseModel , Field
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage

class conversation_history(BaseModel):
    """BaseMessage gives the flexibility to the user to store the conversation whether it is human message , AI or System """
    messages : Annotated[list[BaseMessage]]
    