from langgraph import StateGraph , START 
from typing import TypedDict,Annotated
from Pydantic import BaseModel , Field
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages

class conversation_history(BaseModel):
    """BaseMessage gives the flexibility to the user to store the conversation whether it is human message , AI or System """
    messages : Annotated[list[BaseMessage],add_messages]
    