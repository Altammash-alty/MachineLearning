from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate , MessagesPlaceholder



load_dotenv()
#chat template
chat_message=[
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variables="   chat_history"),
    HumanMessage(content="{query}"),
]
chat_history=[]
#load chat  history 
with open('chat_history','r') as f:
    chat_history.extend(f.readlines())
#create prompt
prompt= chat_history.invoke

template = PromptTemplate(
    template="",
    input_variables="",
    validate_template=True
)
chat_template = ChatPromptTemplate(
    [
        ("system", "{system}"),
        ("human", "{human}"),
        ("ai", "{ai}"),
    ]
)
repo_id = "meta-llama/Llama-3.1-8B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=256,
)


prompt = template.invoke(
    userinput1="",
    uerinput2="",
    userinput3=""

)

result=llm.invoke(prompt)
print(result.content)

