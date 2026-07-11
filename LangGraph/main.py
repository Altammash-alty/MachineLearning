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

def chat_node(state:conversation_history):
    messages = state['messages']
    response=llm.invoke(messages)
    return {"messages":response}


graph = StateGraph(conversation_history)


checkpointer=MemorySaver()


graph.add_node('chat_node',chat_node)
graph.add_edge(START,chat_node)
graph.add_edge(chat_node,END)

workflow = graph.compile(checkpointer=checkpointer)

initial_state={
    "message":[HumanMessage(content="what is my name")]
}

final_state = workflow.invoke(initial_state)
print(final_state['messages'])



thread_id='1'
while True :
    user_message=input("What is your  query ?")
    if user_message.strip().lower() in ["quit","bye","exit"]:
        break

    
    for message.chunk , metadata in  workflow.stream(
        {"messages":HumanMessage(content=user_message)},config={
        "configurable":{
            "thread_id":thread_id
        }   },
            stream_mode='messages')):

    for messsage.chunk , metadata in workflow.stream :

        print(response['messages'][-1].content)