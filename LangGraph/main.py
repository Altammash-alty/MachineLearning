from langgraph import StateGraph , START , END
from typing import TypedDict,Annotated
from Pydantic import BaseModel , Field
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFaceEndpoint

class conversation_history(BaseModel):
    """BaseMessage gives the flexibility to the user to store the conversation whether it is human message , AI or System """
    messages : Annotated[list[BaseMessage],add_messages]

repo_id = ""
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="conversation",
    temperature=1.2
)

def chat_node(State:conversation_history):
    messages = State['messages']

    
graph = StateGraph(conversation_history)

graph.add_node('chat_node',chat_node)